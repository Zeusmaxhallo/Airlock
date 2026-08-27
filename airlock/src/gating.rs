//! Stage 5 — gating summaries.
//!
//! Decides which program points are protected by an authorization check, and
//! turns that into two interprocedural summaries over the call graph:
//!
//! * [`compute_always_checks`] — which functions check on every path from
//!   their entry to a successful return. A call to such a function is itself a
//!   check in the caller.
//! * [`compute_entry_checked`] — which functions are entered only from an
//!   already checked context, so a write inside them is gated even though
//!   their own body contains no comparison.
//!
//! What counts as a check within one body is decided by two complementary
//! criteria, both derived from the comparisons of stage 3:
//!
//! * **divergence** ([`effective_guard_locations`]) — a comparison whose
//!   result feeds a branch of which at least one arm abandons the success path;
//! * **edge sensitivity** ([`authorized_gate_locations`]) — the blocks that can
//!   only be reached through authorizing comparison edges, which is what
//!   recovers a disjunctive guard such as
//!   `if sender != owner && sender != manager { return Err(..) }`.
//!
//! The dataflow that propagates the result through a body lives in
//! [`crate::auth_gate`].

use std::collections::{HashMap, HashSet};

use rustc_hir::def::DefKind;
use rustc_middle::mir::{
    BasicBlock, BinOp, Body, Local, Location, Rvalue, StatementKind, TerminatorKind,
};
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::DefId;

use crate::auth_gate::{self, AuthState, Gating};
use crate::call_graph::{CallGraph, CallSite};
use crate::cosmwasm::{self, is_forwarding_glue_fn};
use crate::mir_util::{blocks_reaching, callee_def_id, operand_local};
use crate::sender_comparisons::{SenderComparison, comparison_result_local};

/// The parts of a function's gating that do not change while the
/// interprocedural fixpoints iterate.
///
/// The `Ok`-return points and the guards a function derives from its own
/// comparisons depend only on the body and on stage 3's result, both of which
/// are fixed by the time stage 5 starts. Computing them once — rather than per
/// fixpoint round, and again for the second fixpoint — keeps the repeated
/// reachability analysis out of the iteration.
pub struct GatingFacts {
    ok_points: Vec<Location>,
    own_guards: HashSet<Location>,
}

impl GatingFacts {
    fn build<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>, comparisons: &[SenderComparison]) -> Self {
        let ok_points = auth_gate::ok_return_points(tcx, body);
        if comparisons.is_empty() {
            return GatingFacts {
                ok_points,
                own_guards: HashSet::new(),
            };
        }

        let ok_blocks: HashSet<BasicBlock> = ok_points.iter().map(|l| l.block).collect();
        let alias = build_comparison_alias_map(tcx, body);

        let mut own_guards = effective_guard_locations(body, comparisons, &ok_blocks, &alias);
        own_guards.extend(authorized_gate_locations(body, comparisons, &alias));

        GatingFacts {
            ok_points,
            own_guards,
        }
    }

    /// All check locations for one analysis run: the function's own guards
    /// plus every call site whose callee checks on all paths.
    fn check_locations(
        &self,
        call_sites: &[CallSite],
        always_checks: &HashMap<DefId, bool>,
    ) -> HashSet<Location> {
        let mut locations = self.own_guards.clone();
        for cs in call_sites {
            if always_checks.get(&cs.callee).copied().unwrap_or(false) {
                locations.insert(cs.location);
            }
        }
        locations
    }
}

/// The gating facts of every analysed body, keyed by function.
pub struct GatingIndex {
    facts: HashMap<DefId, GatingFacts>,
}

impl GatingIndex {
    pub fn build(tcx: TyCtxt<'_>, fn_comparisons: &HashMap<DefId, Vec<SenderComparison>>) -> Self {
        let mut facts = HashMap::new();
        for (&def_id, comparisons) in fn_comparisons {
            let Some(body) = cosmwasm::body_of(tcx, def_id) else {
                continue;
            };
            facts.insert(def_id, GatingFacts::build(tcx, body, comparisons));
        }
        GatingIndex { facts }
    }

    fn get(&self, def_id: DefId) -> Option<&GatingFacts> {
        self.facts.get(&def_id)
    }

    /// Runs the gating analysis of `def_id` under an assumed entry state.
    /// Used by stage 6, which needs the state at each sink rather than only
    /// the summary.
    pub fn solve_for<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        body: &Body<'tcx>,
        def_id: DefId,
        call_sites: &[CallSite],
        always_checks: &HashMap<DefId, bool>,
        entry_checked: AuthState,
    ) -> Option<Gating> {
        let facts = self.get(def_id)?;
        let checks = facts.check_locations(call_sites, always_checks);
        Some(auth_gate::solve(
            tcx,
            body,
            checks,
            entry_checked,
            &facts.ok_points,
        ))
    }
}

/// A comparison only counts as an *enforcing* guard if its result feeds a
/// `SwitchInt` of which at least one arm abandons the success path entirely:
/// from that target no `Ok`-return is reachable any more, because of an early
/// `return Err(..)`, a `?` on a fallible check, or a panic. A comparison whose
/// arms all rejoin the success path — fee logic keyed on the sender, say —
/// enforces nothing and must not count.
fn effective_guard_locations<'tcx>(
    body: &Body<'tcx>,
    comparisons: &[SenderComparison],
    ok_blocks: &HashSet<BasicBlock>,
    alias: &HashMap<Local, Local>,
) -> HashSet<Location> {
    let mut result_to_loc: HashMap<Local, Location> = HashMap::new();
    for cmp in comparisons {
        if let Some(result) = comparison_result_local(body, cmp) {
            result_to_loc.insert(result, cmp.location);
        }
    }
    if result_to_loc.is_empty() {
        return HashSet::new();
    }

    // "Does this arm still reach a successful return?" as a single backward
    // reachability instead of a forward search per branch target.
    let can_reach_ok = blocks_reaching(body, ok_blocks);

    let mut effective = HashSet::new();
    for data in body.basic_blocks.iter() {
        let TerminatorKind::SwitchInt { discr, targets } = &data.terminator().kind else {
            continue;
        };
        let Some(discr_local) = operand_local(discr) else {
            continue;
        };
        let matched = resolve_alias_chain(alias, discr_local)
            .into_iter()
            .find_map(|l| result_to_loc.get(&l).copied());
        let Some(location) = matched else {
            continue;
        };

        let has_aborting_arm = targets
            .all_targets()
            .iter()
            .any(|t| !can_reach_ok.contains(t));
        if has_aborting_arm {
            effective.insert(location);
        }
    }

    effective
}

/// Edge-sensitive gating for disjunctive checks.
///
/// A check such as `if sender != owner && sender != manager { return Err(..) }`
/// lets the authorized caller — owner *or* manager — through a short-circuit
/// bypass edge of the `&&`, not through a single dominating guard location.
/// The location-based criterion therefore misses it: the first comparison has
/// no aborting arm, since both of its arms can still reach an `Ok`-return, and
/// the owner's path would arrive at the storage write as `Unchecked`.
///
/// Those gates are recovered by reasoning about *edges*. For every equality
/// comparison feeding a boolean `SwitchInt` the authorizing edge is
/// determined — for `!=` the false edge, for `==` the true edge, both meaning
/// `sender == principal` — and the entry of a block is marked whenever *all*
/// of its predecessors are authorizing edges.
///
/// That "all predecessors authorized" condition is what keeps the marking
/// sound: a block reachable by even one unauthorized path is never marked, so
/// no vulnerability can be hidden — a misclassification can only ever fail
/// towards a false positive.
fn authorized_gate_locations<'tcx>(
    body: &Body<'tcx>,
    comparisons: &[SenderComparison],
    alias: &HashMap<Local, Local>,
) -> HashSet<Location> {
    let mut result_to_op: HashMap<Local, BinOp> = HashMap::new();
    for cmp in comparisons {
        if matches!(cmp.op, BinOp::Eq | BinOp::Ne) {
            if let Some(result) = comparison_result_local(body, cmp) {
                result_to_op.insert(result, cmp.op);
            }
        }
    }
    if result_to_op.is_empty() {
        return HashSet::new();
    }

    let mut authorized_edges: HashSet<(BasicBlock, BasicBlock)> = HashSet::new();
    for (block, data) in body.basic_blocks.iter_enumerated() {
        let TerminatorKind::SwitchInt { discr, targets } = &data.terminator().kind else {
            continue;
        };
        // Genuine boolean branches only. A multi-way discriminant switch over
        // an enum is not an equality guard and must not be classified here.
        if targets.all_targets().len() != 2 {
            continue;
        }
        let Some(discr_local) = operand_local(discr) else {
            continue;
        };
        let matched = resolve_alias_chain(alias, discr_local)
            .into_iter()
            .find_map(|l| result_to_op.get(&l).copied());
        let Some(op) = matched else {
            continue;
        };
        // In a boolean `SwitchInt` value 0 is the false edge and `otherwise`
        // the true edge. `sender == principal` is the authorizing outcome, so
        // it is the false edge for `!=` and the true edge for `==`.
        let authorized_succ = match op {
            BinOp::Ne => targets.target_for_value(0),
            BinOp::Eq => targets.otherwise(),
            _ => continue,
        };
        authorized_edges.insert((block, authorized_succ));
    }
    if authorized_edges.is_empty() {
        return HashSet::new();
    }

    // Forward must-analysis over blocks: a block is authorized exactly when
    // every path from the entry to it crosses an authorizing comparison edge.
    //
    // Marking only the immediate successor of each edge would not suffice: an
    // `Option` comparison such as `Some(sender) != cfg.owner` leaves a
    // temporary that drop elaboration frees on the taken edge, which inserts a
    // drop-and-goto block between the switch and the real convergence block.
    // Propagating the property through such forwarding blocks recovers those
    // gates without weakening the soundness condition.
    let entry = BasicBlock::from_usize(0);
    let predecessors = body.basic_blocks.predecessors();
    // Greatest fixpoint: optimistically `true` everywhere but the entry, then
    // monotonically lowered as unauthorized paths are discovered.
    let mut authorized = vec![true; body.basic_blocks.len()];
    authorized[entry.as_usize()] = false;

    loop {
        let mut changed = false;
        for (block, _) in body.basic_blocks.iter_enumerated() {
            if block == entry {
                continue;
            }
            let preds = &predecessors[block];
            let new_value = !preds.is_empty()
                && preds
                    .iter()
                    .all(|&p| authorized_edges.contains(&(p, block)) || authorized[p.as_usize()]);
            if new_value != authorized[block.as_usize()] {
                authorized[block.as_usize()] = new_value;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    body.basic_blocks
        .iter_enumerated()
        .filter(|(block, _)| *block != entry && authorized[block.as_usize()])
        .map(|(block, _)| Location {
            block,
            statement_index: 0,
        })
        .collect()
}

/// Maps each local to the local it is a copy of, so that a `SwitchInt`
/// discriminant can be traced back to the local holding a comparison result.
///
/// Three kinds of step are recorded: `Use`/`UnaryOp` copies, which cover `!`;
/// `Discriminant` reads, through which `x?` and `match res { .. }` switch; and
/// calls to forwarding glue, because `x?` routes the checked `Result` through
/// a call to `Try::branch` before switching on its discriminant. Without the
/// last two no fallible check — a library authorization helper above all —
/// would ever resolve to its comparison.
fn build_comparison_alias_map<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) -> HashMap<Local, Local> {
    let mut alias: HashMap<Local, Local> = HashMap::new();

    for data in body.basic_blocks.iter() {
        for stmt in data.statements.iter() {
            let StatementKind::Assign(assign) = &stmt.kind else {
                continue;
            };
            let (lhs, rvalue) = assign.as_ref();
            let source = match rvalue {
                Rvalue::Use(op, _) | Rvalue::UnaryOp(_, op) => operand_local(op),
                Rvalue::Discriminant(place) => Some(place.local),
                _ => None,
            };
            if let Some(source) = source {
                alias.insert(lhs.local, source);
            }
        }

        if let TerminatorKind::Call {
            func,
            args,
            destination,
            ..
        } = &data.terminator().kind
        {
            let is_glue =
                callee_def_id(tcx, body, func).is_some_and(|d| is_forwarding_glue_fn(tcx, d));
            if is_glue {
                if let Some(arg) = args.first().and_then(|a| operand_local(&a.node)) {
                    alias.insert(destination.local, arg);
                }
            }
        }
    }

    alias
}

/// Follows the alias chain starting at `start`, inclusive, stopping at the
/// first cycle. Returns every local the discriminant may alias, so that a
/// comparison result is recovered through copies and glue alike.
fn resolve_alias_chain(alias: &HashMap<Local, Local>, start: Local) -> Vec<Local> {
    let mut chain = vec![start];
    let mut seen = HashSet::new();
    seen.insert(start);
    let mut current = start;
    while let Some(&next) = alias.get(&current) {
        if !seen.insert(next) {
            break;
        }
        chain.push(next);
        current = next;
    }
    chain
}

/// Determines, for every function of the call graph, whether it performs an
/// authorization check on every path from its entry to a successful return.
///
/// Solved as a least fixpoint over the call graph: a call to an
/// already-checking callee counts as a check, which may in turn make its
/// caller always-checking. The monotone iteration from the pessimistic initial
/// value `false` also terminates in the presence of recursion.
pub fn compute_always_checks(
    tcx: TyCtxt<'_>,
    call_graph: &CallGraph,
    gating: &GatingIndex,
) -> HashMap<DefId, bool> {
    let mut summary: HashMap<DefId, bool> =
        call_graph.nodes().iter().map(|&n| (n, false)).collect();

    loop {
        let mut changed = false;

        for &f in call_graph.nodes() {
            if summary.get(&f).copied().unwrap_or(false) {
                continue;
            }
            let (Some(body), Some(facts)) = (cosmwasm::body_of(tcx, f), gating.get(f)) else {
                continue;
            };

            let checks = facts.check_locations(call_graph.call_sites_in(f), &summary);
            let solved =
                auth_gate::solve(tcx, body, checks, AuthState::Unchecked, &facts.ok_points);

            if solved.always().is_checked() {
                summary.insert(f, true);
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    summary
}

/// Determines, for every function of the call graph, whether it is entered
/// only after an authorization check has already happened in the caller.
///
/// The value for `f` is the meet over all call sites `C -> f` of the gating
/// state at that call site in `C`, analysed under `C`'s own context. The root
/// is an external entry point and therefore never entry-checked.
///
/// Solved as a greatest fixpoint: the optimistic initial value `true` for all
/// non-root functions is lowered monotonically as ungated call paths are
/// discovered, which terminates and handles recursion.
pub fn compute_entry_checked(
    tcx: TyCtxt<'_>,
    call_graph: &CallGraph,
    gating: &GatingIndex,
    always_checks: &HashMap<DefId, bool>,
) -> HashMap<DefId, bool> {
    let root = call_graph.root();
    let mut entry: HashMap<DefId, bool> =
        call_graph.nodes().iter().map(|&n| (n, n != root)).collect();

    loop {
        let mut changed = false;
        let mut contribution: HashMap<DefId, bool> = HashMap::new();

        for &caller in call_graph.nodes() {
            let (Some(body), Some(facts)) = (cosmwasm::body_of(tcx, caller), gating.get(caller))
            else {
                continue;
            };
            let call_sites = call_graph.call_sites_in(caller);

            let checks = facts.check_locations(call_sites, always_checks);
            let context = AuthState::of(entry.get(&caller).copied().unwrap_or(false));
            let solved = auth_gate::solve(tcx, body, checks, context, &facts.ok_points);

            for cs in call_sites {
                let gated_here = solved.at(cs.location.block).is_checked();
                let acc = contribution.entry(cs.callee).or_insert(true);
                *acc = *acc && gated_here;
            }
        }

        for (&callee, &gated) in &contribution {
            if callee == root {
                continue;
            }
            if entry.get(&callee).copied().unwrap_or(true) != gated {
                entry.insert(callee, gated);
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    entry.insert(root, false);
    entry
}

/// Closures whose body crosses an `info.sender` check on every path to an
/// `Ok`-return.
///
/// This covers the `Item::update` / `Map::update` authorization idiom, where
/// guard and write both live inside the update closure:
///
/// ```ignore
/// STATE.update(store, |mut s| {
///     if info.sender != s.owner { return Err(..) };
///     s.field = ..;
///     Ok(s)
/// })
/// ```
///
/// The update call is a storage write in the *parent*, but its guard is in the
/// closure, so the gating of the parent body alone would not see it. The same
/// dataflow is therefore applied to each closure body.
pub fn closures_always_checking(
    tcx: TyCtxt<'_>,
    fn_comparisons: &HashMap<DefId, Vec<SenderComparison>>,
    gating: &GatingIndex,
    always_checks: &HashMap<DefId, bool>,
) -> HashSet<DefId> {
    let mut out = HashSet::new();

    for (&def_id, comparisons) in fn_comparisons {
        if comparisons.is_empty() || !matches!(tcx.def_kind(def_id), DefKind::Closure) {
            continue;
        }
        let (Some(body), Some(facts)) = (cosmwasm::body_of(tcx, def_id), gating.get(def_id)) else {
            continue;
        };
        // Only `Result`-returning closures — the update actions — qualify. A
        // boolean predicate closure has no `Ok(..)` and is handled by the
        // `Option` combinator recognition of stage 3.
        if facts.ok_points.is_empty() {
            crate::debug::closure_skipped(tcx, def_id, comparisons.len());
            continue;
        }

        let checks = facts.check_locations(&[], always_checks);
        let check_count = checks.len();
        let solved = auth_gate::solve(tcx, body, checks, AuthState::Unchecked, &facts.ok_points);
        crate::debug::closure_result(
            tcx,
            def_id,
            comparisons.len(),
            facts.ok_points.len(),
            check_count,
            solved.always(),
        );

        if solved.always().is_checked() {
            out.insert(def_id);
        }
    }

    out
}
