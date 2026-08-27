//! Stage 6 — final evaluation.
//!
//! For every write to an authorization-relevant storage item — the *sink* —
//! two questions are answered and combined into the verdict:
//!
//! * is the written value attacker-controlled (taint, [`crate::taint`])?
//! * is the access gated by an effective check on every path (gating,
//!   [`crate::gating`])?
//!
//! A tainted write without a gate is a reportable vulnerability; every other
//! combination is benign and is reported with the reason it is benign.

use std::collections::{HashMap, HashSet};

use rustc_middle::mir::{Body, Local, Location, TerminatorKind};
use rustc_middle::ty::{TyCtxt, TyKind};
use rustc_span::def_id::DefId;

use crate::auth_gate::AuthState;
use crate::call_graph::{CallSite, callee_locations};
use crate::cosmwasm::is_storage_write_fn;
use crate::gating::GatingIndex;
use crate::mir_util::{callee_def_id, operand_local};
use crate::storage_inventory::{ConstDefIndex, ConstTypeIndex, resolve_storage_item};
use crate::taint::{self, TaintFacts};

/// A write to an authorization-relevant storage item.
#[derive(Debug, Clone)]
pub struct SensitiveSink {
    pub location: Location,
    /// Name of the storage constant written to, for example `OWNER`.
    pub symbolic_name: String,
}

/// An attacker-controlled value flowing into a sink. The taint originates from
/// a function parameter of address type, i.e. a field of the `ExecuteMsg` the
/// caller supplied.
#[derive(Debug, Clone)]
pub struct TaintSource {
    /// The parameter local the tainted value originates from.
    pub param_local: Local,
    /// Normalized type of that parameter, for example `Addr` or `Vec<String>`.
    pub param_ty: String,
}

/// The verdict for one sink.
#[derive(Debug, Clone)]
pub struct AccessControlFinding {
    pub sink: SensitiveSink,
    /// `Some` if the written value is attacker-controlled.
    pub taint: Option<TaintSource>,
    /// `true` if an authorization check precedes the sink on every path.
    pub gated: bool,
}

impl AccessControlFinding {
    /// A reportable vulnerability: a tainted write that no check gates.
    pub fn is_vulnerability(&self) -> bool {
        self.taint.is_some() && !self.gated
    }

    /// Short label for the report.
    pub fn verdict(&self) -> &'static str {
        if self.is_vulnerability() {
            "VULN "
        } else if self.gated {
            "gated"
        } else {
            "clean"
        }
    }
}

/// Everything stage 6 needs beyond the body under evaluation.
pub struct EvaluationContext<'a, 'tcx> {
    pub const_types: &'a ConstTypeIndex<'tcx>,
    pub gating: &'a GatingIndex,
    pub always_checks: &'a HashMap<DefId, bool>,
    pub return_taint_params: &'a HashMap<DefId, HashSet<usize>>,
    pub entry_checked: &'a HashMap<DefId, bool>,
    /// Closures that check the sender on every path to an `Ok`-return.
    pub checking_closures: &'a HashSet<DefId>,
}

/// Evaluates all sinks of one function.
pub fn evaluate_function<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    def_id: DefId,
    call_sites: &[CallSite],
    auth_item_names: &HashSet<&str>,
    ctx: &EvaluationContext<'_, 'tcx>,
) -> Vec<AccessControlFinding> {
    if auth_item_names.is_empty() {
        return Vec::new();
    }

    let const_defs = ConstDefIndex::build(tcx, body);
    let sinks = find_sensitive_sinks(tcx, body, auth_item_names, &const_defs, ctx.const_types);
    if sinks.is_empty() {
        return Vec::new();
    }

    let taint_by_sink = find_taint_sources(tcx, body, &sinks, call_sites, ctx, &const_defs);

    let entry = AuthState::of(ctx.entry_checked.get(&def_id).copied().unwrap_or(false));
    let gating = ctx
        .gating
        .solve_for(tcx, body, def_id, call_sites, ctx.always_checks, entry);

    sinks
        .into_iter()
        .map(|sink| {
            let location = sink.location;
            // Gated either by a check in this body, or — for the
            // `Item::update(store, |s| { if sender != s.owner {..}; Ok(s) })`
            // idiom — by an always-checking guard inside the closure the write
            // call itself receives.
            let gated = gating
                .as_ref()
                .is_some_and(|g| g.at(location.block).is_checked())
                || sink_gated_by_closure(tcx, body, location, ctx.checking_closures);
            AccessControlFinding {
                taint: taint_by_sink.get(&location).cloned(),
                sink,
                gated,
            }
        })
        .collect()
}

/// Finds the writes to authorization-relevant storage items.
fn find_sensitive_sinks<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    auth_item_names: &HashSet<&str>,
    const_defs: &ConstDefIndex,
    const_types: &ConstTypeIndex<'tcx>,
) -> Vec<SensitiveSink> {
    let mut sinks = Vec::new();

    for (block, bb_data) in body.basic_blocks.iter_enumerated() {
        let TerminatorKind::Call { func, args, .. } = &bb_data.terminator().kind else {
            continue;
        };
        if !callee_def_id(tcx, body, func).is_some_and(|d| is_storage_write_fn(tcx, d)) {
            continue;
        }

        // Argument 0 is the receiver, i.e. the storage item written to.
        let Some(receiver) = args.first().and_then(|a| operand_local(&a.node)) else {
            continue;
        };
        let (symbolic_name, _) = resolve_storage_item(tcx, body, receiver, const_defs, const_types);

        if auth_item_names.contains(symbolic_name.as_str()) {
            sinks.push(SensitiveSink {
                location: Location {
                    block,
                    statement_index: bb_data.statements.len(),
                },
                symbolic_name,
            });
        }
    }

    sinks
}

/// Determines, per sink, whether the value it writes is attacker-controlled.
fn find_taint_sources<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    sinks: &[SensitiveSink],
    call_sites: &[CallSite],
    ctx: &EvaluationContext<'_, 'tcx>,
    const_defs: &ConstDefIndex,
) -> HashMap<Location, TaintSource> {
    let seed = taint::address_seed(tcx, body);
    let facts = TaintFacts::build(tcx, body, const_defs, ctx.const_types);
    let callee_at = callee_locations(call_sites);
    let result = taint::propagate_taint_forward(
        tcx,
        body,
        &seed,
        &facts,
        &callee_at,
        ctx.return_taint_params,
    );

    let mut sources = HashMap::new();

    for sink in sinks {
        // The written value is always the last argument of the write call:
        //   Item::save(self, store, data)      -> data
        //   Map::save(self, store, key, data)  -> data
        let TerminatorKind::Call { args, .. } =
            &body.basic_blocks[sink.location.block].terminator().kind
        else {
            continue;
        };
        let Some(value_local) = args.last().and_then(|a| operand_local(&a.node)) else {
            continue;
        };

        if result.is_tainted(value_local) {
            let param_local = result.origin_of(value_local);
            let param_ty = taint::origin_type(body, param_local);

            crate::debug::tainted_sink(body, &result, sink, value_local, param_local, &param_ty);

            sources.insert(
                sink.location,
                TaintSource {
                    param_local,
                    param_ty,
                },
            );
        } else {
            crate::debug::untainted_sink(body, &result, sink, value_local, &seed);
        }
    }

    sources
}

/// Returns whether the write call at `location` passes a closure argument that
/// is an always-checking sender guard.
fn sink_gated_by_closure<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    location: Location,
    checking_closures: &HashSet<DefId>,
) -> bool {
    if checking_closures.is_empty() {
        return false;
    }
    let TerminatorKind::Call { args, .. } = &body.basic_blocks[location.block].terminator().kind
    else {
        return false;
    };

    args.iter()
        .any(|a| match a.node.ty(&body.local_decls, tcx).kind() {
            TyKind::Closure(closure_def, _) => {
                crate::debug::closure_sink(tcx, location, *closure_def, checking_closures);
                checking_closures.contains(closure_def)
            }
            _ => false,
        })
}
