//! The auth-gate dataflow.
//!
//! The engine underneath stage 5: a forward must-analysis over one MIR body
//! that propagates whether an authorization check has already been passed. It
//! is built on `rustc_mir_dataflow`, which supplies the worklist, the
//! reverse-postorder iteration and — decisively — the correct treatment of
//! unwind and cleanup edges: those are not ordinary predecessors of a sink and
//! must not dilute the must-state.
//!
//! Which program points count as a check is not decided here; [`crate::gating`]
//! derives them from the comparisons of stage 3 and from the call graph, and
//! passes them in.

use std::collections::HashSet;
use std::fmt;

use rustc_middle::mir::{
    AggregateKind, BasicBlock, Body, Location, Rvalue, Statement, StatementKind, Terminator,
    TerminatorEdges,
};
use rustc_middle::ty::TyCtxt;
use rustc_mir_dataflow::{Analysis, Forward, JoinSemiLattice, fmt::DebugWithContext};

use crate::cosmwasm::is_result_def;

/// Whether an authorization check has been observed on every path to a
/// program point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthState {
    Unchecked,
    Checked,
}

impl AuthState {
    pub fn of(checked: bool) -> Self {
        if checked {
            AuthState::Checked
        } else {
            AuthState::Unchecked
        }
    }

    pub fn is_checked(self) -> bool {
        self == AuthState::Checked
    }
}

impl JoinSemiLattice for AuthState {
    fn join(&mut self, other: &Self) -> bool {
        // A meet in the intuitive reading: only a point that is checked on
        // both incoming paths stays checked.
        let merged = match (*self, *other) {
            (AuthState::Checked, AuthState::Checked) => AuthState::Checked,
            _ => AuthState::Unchecked,
        };
        if merged != *self {
            *self = merged;
            true
        } else {
            false
        }
    }
}

impl<C> DebugWithContext<C> for AuthState {
    fn fmt_with(&self, _ctxt: &C, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Forward must-analysis propagating [`AuthState`] through a body.
///
/// Built on `rustc_mir_dataflow`, which supplies the worklist, the
/// reverse-postorder iteration and — decisively — the correct treatment of
/// unwind and cleanup edges: those are not ordinary predecessors of a sink and
/// must not dilute the must-state.
struct AuthGate {
    /// Program points that count as an authorization check.
    check_locations: HashSet<Location>,
    /// State at the function entry: `Unchecked` for a standalone analysis,
    /// `Checked` once an interprocedural context says every caller checks
    /// first.
    entry_checked: AuthState,
}

impl<'tcx> Analysis<'tcx> for AuthGate {
    type Domain = AuthState;
    type Direction = Forward;

    const NAME: &'static str = "auth_gate";

    /// Neutral element of the join. The optimistic initialization is lowered
    /// to `Unchecked` at every join point that has an unchecked predecessor.
    fn bottom_value(&self, _body: &Body<'tcx>) -> AuthState {
        AuthState::Checked
    }

    fn initialize_start_block(&self, _body: &Body<'tcx>, state: &mut AuthState) {
        *state = self.entry_checked;
    }

    fn apply_primary_statement_effect(
        &self,
        state: &mut AuthState,
        _statement: &Statement<'tcx>,
        location: Location,
    ) {
        if self.check_locations.contains(&location) {
            *state = AuthState::Checked;
        }
    }

    fn apply_primary_terminator_effect<'mir>(
        &self,
        state: &mut AuthState,
        terminator: &'mir Terminator<'tcx>,
        location: Location,
    ) -> TerminatorEdges<'mir, 'tcx> {
        if self.check_locations.contains(&location) {
            *state = AuthState::Checked;
        }
        terminator.edges()
    }
}

/// Result of the gating analysis for one body.
pub struct Gating {
    /// State on entry to each basic block, indexed by block number.
    block_states: Vec<AuthState>,
    /// State joined over all `Ok`-return points: `Checked` exactly when the
    /// function checks on every path that returns successfully.
    always: AuthState,
}

impl Gating {
    /// State on entry to `block`.
    pub fn at(&self, block: BasicBlock) -> AuthState {
        self.block_states
            .get(block.as_usize())
            .copied()
            .unwrap_or(AuthState::Unchecked)
    }

    pub fn always(&self) -> AuthState {
        self.always
    }
}

/// Runs the gating analysis on one body.
pub fn solve<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    check_locations: HashSet<Location>,
    entry_checked: AuthState,
    ok_points: &[Location],
) -> Gating {
    let analysis = AuthGate {
        check_locations,
        entry_checked,
    };

    let results = analysis.iterate_to_fixpoint(tcx, body, None);
    let mut cursor = results.into_results_cursor(body);

    let mut block_states = Vec::with_capacity(body.basic_blocks.len());
    for (block, _) in body.basic_blocks.iter_enumerated() {
        cursor.seek_before_primary_effect(Location {
            block,
            statement_index: 0,
        });
        block_states.push(*cursor.get());
    }

    // A function without any successful return cannot vouch for a caller, so
    // the absence of `Ok`-return points is `Unchecked` rather than the neutral
    // element of the join.
    let mut always = AuthState::Checked;
    for location in ok_points {
        cursor.seek_before_primary_effect(*location);
        always.join(cursor.get());
    }
    if ok_points.is_empty() {
        always = AuthState::Unchecked;
    }

    Gating {
        block_states,
        always,
    }
}

/// The program points at which a function assigns its successful return value,
/// `_0 = Ok(..)`. They are the reference points of the must-analysis: a check
/// has to be passed before every one of them.
pub fn ok_return_points<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) -> Vec<Location> {
    let mut points = Vec::new();

    for (block, data) in body.basic_blocks.iter_enumerated() {
        for (statement_index, stmt) in data.statements.iter().enumerate() {
            let StatementKind::Assign(assign) = &stmt.kind else {
                continue;
            };
            let (place, rvalue) = assign.as_ref();
            if place.local.as_usize() != 0 || !place.projection.is_empty() {
                continue;
            }
            let Rvalue::Aggregate(kind, _) = rvalue else {
                continue;
            };
            if let AggregateKind::Adt(def_id, variant_index, ..) = kind.as_ref() {
                // `Result::Ok` is variant 0.
                if variant_index.as_usize() == 0 && is_result_def(tcx, *def_id) {
                    points.push(Location {
                        block,
                        statement_index,
                    });
                }
            }
        }
    }

    points
}
