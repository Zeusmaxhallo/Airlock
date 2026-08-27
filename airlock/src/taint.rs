//! Taint propagation — attacker-controlled values.
//!
//! Access control only matters where the written value comes from the message
//! the caller sent. This module answers that question in two parts:
//!
//! * [`propagate_taint_forward`] is the intraprocedural forward taint over a
//!   single body, seeded with the address-typed parameters of the handler;
//! * [`compute_return_taint_params`] is the interprocedural summary of
//!   stage 5: which parameter positions of a function reach its return value,
//!   so that a call to a helper forwards taint precisely instead of
//!   conservatively tainting everything it touches.
//!
//! Storage is modelled per item rather than through the `deps` handle: a value
//! read from storage is attacker-controlled only if that *same* item was
//! written with tainted data earlier in the handler. Tainting the handle
//! instead would let an unrelated write — an IBC transfer with a user-supplied
//! recipient, say — contaminate every later load.

use std::collections::{HashMap, HashSet};

use rustc_middle::mir::{
    AggregateKind, Body, BorrowKind, Local, Location, Operand, Rvalue, StatementKind,
    TerminatorKind,
};
use rustc_middle::ty::{Ty, TyCtxt, TyKind};
use rustc_span::{def_id::DefId, sym};

use crate::call_graph::{CallGraph, callee_locations};
use crate::cosmwasm::{
    self, is_cosmwasm_addr, is_framework_ty, is_message_info_ty, is_result_def, is_result_ty,
    is_std_string, is_storage_load_fn, is_storage_write_fn,
};
use crate::mir_util::{callee_def_id, normalize_ty_str, operand_local, rvalue_locals};
use crate::storage_inventory::{ConstDefIndex, ConstTypeIndex, resolve_storage_def_id};

/// Result of a taint propagation over one body.
pub struct TaintResult {
    /// Every local reached by the taint.
    tainted: HashSet<Local>,
    /// For each tainted local, the seed it originates from.
    origin: HashMap<Local, Local>,
}

impl TaintResult {
    pub fn is_tainted(&self, local: Local) -> bool {
        self.tainted.contains(&local)
    }

    /// The seed a tainted local originates from; the local itself if it is
    /// not part of a chain.
    pub fn origin_of(&self, local: Local) -> Local {
        self.origin.get(&local).copied().unwrap_or(local)
    }
}

/// The parts of a body the taint propagation needs but that do not depend on
/// the seed, so they can be built once and reused across the seeds of a
/// function and across the rounds of the interprocedural fixpoint.
pub struct TaintFacts {
    /// For each reference local, the local it borrows.
    borrow_of: HashMap<Local, Local>,
    /// The reference locals that are mutable borrows, through which a callee
    /// can write back into the caller.
    mut_borrow: HashSet<Local>,
    /// The storage constant behind each load or write receiver.
    item_def_of: HashMap<Local, Option<DefId>>,
}

impl TaintFacts {
    pub fn build<'tcx>(
        tcx: TyCtxt<'tcx>,
        body: &Body<'tcx>,
        const_defs: &ConstDefIndex,
        const_types: &ConstTypeIndex<'tcx>,
    ) -> Self {
        let (borrow_of, mut_borrow) = build_borrow_maps(body);

        let mut item_def_of: HashMap<Local, Option<DefId>> = HashMap::new();
        for bb_data in body.basic_blocks.iter() {
            let TerminatorKind::Call { func, args, .. } = &bb_data.terminator().kind else {
                continue;
            };
            let is_storage_op = callee_def_id(tcx, body, func)
                .is_some_and(|d| is_storage_write_fn(tcx, d) || is_storage_load_fn(tcx, d));
            if !is_storage_op {
                continue;
            }
            let Some(receiver) = args.first().and_then(|a| operand_local(&a.node)) else {
                continue;
            };
            item_def_of.entry(receiver).or_insert_with(|| {
                resolve_storage_def_id(tcx, body, receiver, const_defs, const_types)
            });
        }

        TaintFacts {
            borrow_of,
            mut_borrow,
            item_def_of,
        }
    }
}

/// Maps each reference local to the local it borrows, recording whether the
/// borrow is mutable.
fn build_borrow_maps<'tcx>(body: &Body<'tcx>) -> (HashMap<Local, Local>, HashSet<Local>) {
    let mut borrow_of: HashMap<Local, Local> = HashMap::new();
    let mut mut_borrow: HashSet<Local> = HashSet::new();

    for bb_data in body.basic_blocks.iter() {
        for stmt in bb_data.statements.iter() {
            let StatementKind::Assign(assign) = &stmt.kind else {
                continue;
            };
            let (lhs, rvalue) = assign.as_ref();
            match rvalue {
                Rvalue::Ref(_, kind, place) => {
                    borrow_of.insert(lhs.local, place.local);
                    if matches!(kind, BorrowKind::Mut { .. }) {
                        mut_borrow.insert(lhs.local);
                    }
                }
                // A raw pointer is treated conservatively as a mutable borrow.
                Rvalue::RawPtr(_, place) => {
                    borrow_of.insert(lhs.local, place.local);
                    mut_borrow.insert(lhs.local);
                }
                // A reborrow through a copy or move of an existing reference.
                Rvalue::Use(Operand::Copy(src) | Operand::Move(src), _) => {
                    if let Some(base) = borrow_of.get(&src.local).copied() {
                        borrow_of.insert(lhs.local, base);
                        if mut_borrow.contains(&src.local) {
                            mut_borrow.insert(lhs.local);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    (borrow_of, mut_borrow)
}

/// Forward taint fixpoint over one body, starting from `seed`.
pub fn propagate_taint_forward<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    seed: &HashSet<Local>,
    facts: &TaintFacts,
    callee_at: &HashMap<Location, DefId>,
    return_taint_params: &HashMap<DefId, HashSet<usize>>,
) -> TaintResult {
    let mut tainted: HashSet<Local> = HashSet::new();
    let mut origin: HashMap<Local, Local> = HashMap::new();

    for &local in seed {
        tainted.insert(local);
        origin.insert(local, local);
    }
    if tainted.is_empty() {
        return TaintResult { tainted, origin };
    }

    // Storage items written with tainted data, and the seed that data came
    // from. `unknown_item` is the conservative fallback for a written item
    // that could not be resolved to a constant: a later load is then tainted
    // as well, which errs towards a false positive and never towards a missed
    // vulnerability.
    let mut tainted_items: HashMap<DefId, Local> = HashMap::new();
    let mut unknown_item: Option<Local> = None;

    let mark = |local: Local,
                src: Local,
                tainted: &mut HashSet<Local>,
                origin: &mut HashMap<Local, Local>|
     -> bool {
        let newly = tainted.insert(local);
        origin.entry(local).or_insert(src);
        newly
    };

    loop {
        let mut changed = false;

        for (block, bb_data) in body.basic_blocks.iter_enumerated() {
            for stmt in bb_data.statements.iter() {
                let StatementKind::Assign(assign) = &stmt.kind else {
                    continue;
                };
                let (lhs, rvalue) = assign.as_ref();
                if let Some(src) = rvalue_locals(rvalue)
                    .into_iter()
                    .find(|l| tainted.contains(l))
                    .and_then(|l| origin.get(&l).copied())
                {
                    changed |= mark(lhs.local, src, &mut tainted, &mut origin);
                }
            }

            let TerminatorKind::Call {
                func,
                args,
                destination,
                ..
            } = &bb_data.terminator().kind
            else {
                continue;
            };

            let location = Location {
                block,
                statement_index: bb_data.statements.len(),
            };
            let arg_locals: Vec<Local> =
                args.iter().filter_map(|a| operand_local(&a.node)).collect();

            // Per argument, the seed its taint came from — directly, or through
            // the local it borrows.
            let arg_taint: Vec<Option<Local>> = arg_locals
                .iter()
                .map(|l| {
                    if tainted.contains(l) {
                        origin.get(l).copied().or(Some(*l))
                    } else {
                        facts.borrow_of.get(l).and_then(|base| {
                            tainted
                                .contains(base)
                                .then(|| origin.get(base).copied().unwrap_or(*base))
                        })
                    }
                })
                .collect();
            let first_src = arg_taint.iter().copied().flatten().next();

            let callee = callee_def_id(tcx, body, func);

            if callee.is_some_and(|d| is_storage_write_fn(tcx, d)) {
                // A storage write persists its *data* argument — the last one —
                // into the item named by argument 0. If that data is tainted,
                // the item is remembered as tainted; the storage handle itself
                // is deliberately left clean so unrelated later loads stay so.
                let data_src = args
                    .last()
                    .and_then(|a| operand_local(&a.node))
                    .filter(|l| tainted.contains(l))
                    .map(|l| origin.get(&l).copied().unwrap_or(l));
                if let Some(src) = data_src {
                    let item = args
                        .first()
                        .and_then(|a| operand_local(&a.node))
                        .and_then(|l| facts.item_def_of.get(&l).copied().flatten());
                    match item {
                        Some(def_id) => {
                            if !tainted_items.contains_key(&def_id) {
                                tainted_items.insert(def_id, src);
                                changed = true;
                            }
                        }
                        None => {
                            if unknown_item.is_none() {
                                unknown_item = Some(src);
                                changed = true;
                            }
                        }
                    }
                }
            } else if callee.is_some_and(|d| is_storage_load_fn(tcx, d)) {
                // A loaded value is tainted only where this item — or an
                // unresolved written one — holds attacker data. It never
                // inherits taint from the storage handle it was read through.
                let load_src = args
                    .first()
                    .and_then(|a| operand_local(&a.node))
                    .and_then(|l| facts.item_def_of.get(&l).copied().flatten())
                    .and_then(|def_id| tainted_items.get(&def_id).copied())
                    .or(unknown_item);
                if let Some(src) = load_src {
                    changed |= mark(destination.local, src, &mut tainted, &mut origin);
                }
            } else {
                // An ordinary call. Where a return-taint summary exists, only
                // the parameters that actually reach the return value forward
                // their taint; otherwise any tainted argument does.
                let dest_src = match callee_at
                    .get(&location)
                    .and_then(|callee| return_taint_params.get(callee))
                {
                    Some(flow) => arg_taint
                        .iter()
                        .enumerate()
                        .find_map(|(i, s)| if flow.contains(&i) { *s } else { None }),
                    None => first_src,
                };
                if let Some(src) = dest_src {
                    changed |= mark(destination.local, src, &mut tainted, &mut origin);
                }

                // A callee that receives a mutable borrow may write tainted
                // data back into the borrowed local.
                if let Some(src) = first_src {
                    for l in arg_locals.iter() {
                        if facts.mut_borrow.contains(l) {
                            if let Some(base) = facts.borrow_of.get(l).copied() {
                                changed |= mark(base, src, &mut tainted, &mut origin);
                            }
                        }
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }

    TaintResult { tainted, origin }
}

/// Per-function inputs of the return-taint fixpoint, resolved once rather
/// than rebuilt in every round.
struct TaintCandidate<'tcx> {
    def_id: DefId,
    body: &'tcx Body<'tcx>,
    facts: TaintFacts,
    callee_at: HashMap<Location, DefId>,
    return_locals: HashSet<Local>,
    /// Parameter positions eligible as taint sources, i.e. everything but
    /// framework plumbing.
    params: Vec<usize>,
}

/// Determines, for every function of the call graph, the parameter positions
/// whose taint reaches the function's return value.
///
/// Solved as a least fixpoint: a summary that grows may make a caller's
/// parameter reach its own return value, until nothing changes any more.
pub fn compute_return_taint_params<'tcx>(
    tcx: TyCtxt<'tcx>,
    call_graph: &CallGraph,
    const_types: &ConstTypeIndex<'tcx>,
) -> HashMap<DefId, HashSet<usize>> {
    let mut candidates = Vec::new();
    let mut summary: HashMap<DefId, HashSet<usize>> = HashMap::new();

    for &f in call_graph.nodes() {
        let Some(body) = cosmwasm::body_of(tcx, f) else {
            continue;
        };
        summary.insert(f, HashSet::new());

        let const_defs = ConstDefIndex::build(tcx, body);
        candidates.push(TaintCandidate {
            def_id: f,
            body,
            facts: TaintFacts::build(tcx, body, &const_defs, const_types),
            callee_at: callee_locations(call_graph.call_sites_in(f)),
            return_locals: return_value_locals(tcx, body),
            params: (0..body.arg_count)
                .filter(|&i| !is_framework_param(tcx, body, Local::from_usize(i + 1)))
                .collect(),
        });
    }

    loop {
        let mut changed = false;

        for candidate in &candidates {
            let mut flow = summary.get(&candidate.def_id).cloned().unwrap_or_default();

            for &i in &candidate.params {
                if flow.contains(&i) {
                    continue;
                }
                let seed = HashSet::from([Local::from_usize(i + 1)]);
                let result = propagate_taint_forward(
                    tcx,
                    candidate.body,
                    &seed,
                    &candidate.facts,
                    &candidate.callee_at,
                    &summary,
                );
                if candidate
                    .return_locals
                    .iter()
                    .any(|&rl| result.is_tainted(rl))
                {
                    flow.insert(i);
                    changed = true;
                }
            }

            summary.insert(candidate.def_id, flow);
        }

        if !changed {
            break;
        }
    }

    summary
}

/// The locals that make up a function's *successful* return value: the
/// operands of `_0 = Ok(..)` for a `Result`-returning function, `_0` itself
/// otherwise.
fn return_value_locals<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) -> HashSet<Local> {
    let return_place = Local::from_usize(0);
    let mut locals = HashSet::new();

    if !is_result_ty(tcx, body.local_decls[return_place].ty) {
        locals.insert(return_place);
        return locals;
    }

    for data in body.basic_blocks.iter() {
        for stmt in data.statements.iter() {
            let StatementKind::Assign(assign) = &stmt.kind else {
                continue;
            };
            let (place, rvalue) = assign.as_ref();
            if place.local != return_place || !place.projection.is_empty() {
                continue;
            }
            let Rvalue::Aggregate(kind, operands) = rvalue else {
                continue;
            };
            if let AggregateKind::Adt(def_id, variant_index, ..) = kind.as_ref() {
                // `Result::Ok` is variant 0.
                if variant_index.as_u32() == 0 && is_result_def(tcx, *def_id) {
                    locals.extend(operands.iter().filter_map(operand_local));
                }
            }
        }
    }

    // A tail-call return such as `fn h(x) -> Result<T> { inner(x) }` writes
    // `_0` directly from a call and never builds an `Ok(..)` aggregate. Falling
    // back to `_0` over-approximates by including error flows, which errs in
    // the conservative direction.
    if locals.is_empty() {
        locals.insert(return_place);
    }

    locals
}

/// Parameters excluded from taint seeding: framework plumbing such as `Deps`,
/// `Env` or `dyn Storage`.
///
/// `MessageInfo` is deliberately *not* excluded even though it is a framework
/// type. It carries `info.sender`, and helpers that take the whole `info` and
/// return sender-derived data are exactly the flows the sender taint needs.
fn is_framework_param<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>, local: Local) -> bool {
    let ty = body.local_decls[local].ty;
    !is_message_info_ty(tcx, ty) && is_framework_ty(tcx, ty)
}

/// The taint seed of a handler: its address-typed parameters, plus the locals
/// that read an address out of an address-bearing parameter.
pub fn address_seed<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) -> HashSet<Local> {
    let mut seed = HashSet::new();

    // Address-typed parameters are the direct sources. The framework types
    // need no exclusion here: none of them mentions an address, because
    // `MessageInfo`, `Deps` and `Env` carry their addresses in fields rather
    // than in type arguments.
    for idx in 1..=body.arg_count {
        let local = Local::from_usize(idx);
        if ty_mentions_address(tcx, body.local_decls[local].ty) {
            seed.insert(local);
        }
    }

    for data in body.basic_blocks.iter() {
        for stmt in data.statements.iter() {
            let StatementKind::Assign(assign) = &stmt.kind else {
                continue;
            };
            let (lhs, rvalue) = assign.as_ref();

            let source = match rvalue {
                Rvalue::Use(Operand::Copy(p) | Operand::Move(p), _) | Rvalue::Ref(_, _, p) => p,
                _ => continue,
            };

            // Only a *field read* out of a parameter, not the parameter itself
            // (already covered above).
            let root = source.local;
            let is_param = root.as_usize() >= 1 && root.as_usize() <= body.arg_count;
            if !is_param || source.projection.is_empty() {
                continue;
            }

            // Framework parameters are not attacker-controlled sources.
            if is_framework_param(tcx, body, root) {
                continue;
            }

            // A `&mut T` parameter is an in-out buffer for internal state — a
            // struct the caller loaded from storage and threads in for
            // mutation, as in `cfg: &mut Config` — never attacker input.
            // CosmWasm message data always arrives as owned leaf parameters,
            // never behind `&mut`, so reading an address field out of such a
            // struct must not seed attacker taint.
            if let TyKind::Ref(_, _, mutbl) = body.local_decls[root].ty.kind() {
                if mutbl.is_mut() {
                    continue;
                }
            }

            if ty_mentions_address(tcx, source.ty(&body.local_decls, tcx).ty) {
                seed.insert(lhs.local);
            }
        }
    }

    seed
}

/// Returns whether `ty` is, contains, or references an address type
/// (`cosmwasm_std::Addr` or `String`).
pub fn ty_mentions_address<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    match ty.kind() {
        // Bare `str`, behind a `&str` reference, is address-bearing text.
        TyKind::Str => true,
        TyKind::Ref(_, inner, _) => ty_mentions_address(tcx, *inner),
        TyKind::Slice(inner) | TyKind::Array(inner, _) => ty_mentions_address(tcx, *inner),
        TyKind::Tuple(tys) => tys.iter().any(|t| ty_mentions_address(tcx, t)),
        TyKind::Adt(adt_def, args) => {
            let did = adt_def.did();
            // `String` is matched by crate and item name first, with the
            // diagnostic item only as a fallback: the diagnostic-item lookup
            // does not resolve on the pinned toolchain, which would leave
            // `String` and `Option<String>` message fields carrying owner or
            // admin addresses unseeded.
            if is_std_string(tcx, did)
                || tcx.is_diagnostic_item(sym::String, did)
                || is_cosmwasm_addr(tcx, did)
            {
                return true;
            }
            args.types().any(|t| ty_mentions_address(tcx, t))
        }
        _ => false,
    }
}

/// The normalized type of a taint origin, for the report.
pub fn origin_type<'tcx>(body: &Body<'tcx>, local: Local) -> String {
    let ty = body.local_decls[local].ty;
    let base_ty = match ty.kind() {
        TyKind::Ref(_, inner, _) => *inner,
        _ => ty,
    };
    normalize_ty_str(&format!("{:?}", base_ty))
}
