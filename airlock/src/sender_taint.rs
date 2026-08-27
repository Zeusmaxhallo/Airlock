//! Stage 3, first half — where the sender flows.
//!
//! Determines which MIR locals carry `cosmwasm_std::MessageInfo::sender`, both
//! inside a single body and across the call graph.
//!
//! An intraprocedural taint alone misses every check in which the sender
//! arrives indirectly: as a function argument (`is_admin(&info.sender)`) or as
//! a closure upvar (`admins.iter().any(|a| a == info.sender)`). The bodies of
//! such helpers never mention `info` at all. [`seed_sender_taint`] therefore
//! computes a fixpoint over the call graph that pre-taints the parameters and
//! upvars a sender reaches, and [`compute_sender_locals`] runs the local
//! propagation on top of those seeds.

use std::collections::{HashMap, HashSet};

use rustc_middle::mir::{
    AggregateKind, Body, Local, Operand, Place, PlaceElem, Rvalue, StatementKind, TerminatorKind,
};
use rustc_middle::ty::{Ty, TyCtxt, TyKind};
use rustc_span::def_id::DefId;

use crate::call_graph::{CallGraph, CallSite};
use crate::cosmwasm::{self, is_cosmwasm_addr, is_forwarding_glue_fn, is_message_info};
use crate::mir_util::{callee_def_id, operand_local, strip_refs};

/// Pre-tainted entry points of a function, produced by the interprocedural
/// seeding over the call graph. Both kinds seed the intraprocedural taint
/// before it is computed for a body:
///
/// * `param_locals` — formal parameters (`_1`, `_2`, …) a caller passed a
///   sender-tainted argument to, as in `is_admin(&info.sender)`.
/// * `upvar_indices` — closure upvar indices whose captured value was
///   sender-tainted in the enclosing function, as in `|a| a == info.sender`.
///   Inside the closure the upvar is read as a field of the environment local.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SenderSeeds {
    pub param_locals: HashSet<Local>,
    pub upvar_indices: HashSet<usize>,
}

/// The bodies stage 3 analyses: the call-graph nodes plus every closure
/// created inside them, together with the seeds each of them was reached with.
pub struct SenderTaint {
    /// Analysable bodies in deterministic order.
    pub targets: Vec<DefId>,
    seeds: HashMap<DefId, SenderSeeds>,
}

impl SenderTaint {
    pub fn seeds_of(&self, def_id: DefId) -> SenderSeeds {
        self.seeds.get(&def_id).cloned().unwrap_or_default()
    }
}

/// Upper bound on the seeding iterations. The seed sets grow monotonically and
/// converge long before this in practice; the bound only guards against
/// pathological call graphs.
const MAX_SEED_ROUNDS: usize = 16;

/// Interprocedural sender-taint seeding over the call graph.
///
/// An intraprocedural taint alone misses every check in which the sender
/// arrives indirectly: as a function argument or as a closure upvar. This
/// computes a fixpoint — taint each body with its current seeds, then
/// propagate sender-tainted actuals into the callee's formal parameters and
/// sender-tainted captures into the closure's upvars — until the seeds stop
/// growing. Stage 3's detection pass then runs on stable seeds.
pub fn seed_sender_taint(tcx: TyCtxt<'_>, call_graph: &CallGraph) -> SenderTaint {
    let (targets, closure_captures) = collect_targets(tcx, call_graph);

    // Bodies and their call sites, resolved once instead of per round.
    let bodies: HashMap<DefId, &Body<'_>> = targets
        .iter()
        .filter_map(|&f| cosmwasm::body_of(tcx, f).map(|b| (f, b)))
        .collect();
    let mut sites_by_caller: HashMap<DefId, Vec<&CallSite>> = HashMap::new();
    for cs in call_graph.all_call_sites() {
        sites_by_caller.entry(cs.caller).or_default().push(cs);
    }

    let mut seeds: HashMap<DefId, SenderSeeds> = HashMap::new();
    let mut taint: HashMap<DefId, HashSet<Local>> = HashMap::new();

    for _ in 0..MAX_SEED_ROUNDS {
        for (&f, &body) in &bodies {
            let s = seeds.get(&f).cloned().unwrap_or_default();
            taint.insert(f, compute_sender_locals(tcx, body, &s));
        }

        let before = seeds.clone();

        // Argument wiring: a sender-tainted actual at position `i` taints the
        // callee's formal parameter `_{i+1}`, but only if the passed value has
        // a sender-compatible type. The gate is on the *caller's* argument
        // type, not on the callee's parameter: the callee may be generic
        // (`addr: impl AsRef<str>`), whose formal type is only a type
        // parameter, while the actual passed at the call site is concrete.
        // This keeps the taint out of non-address parameters such as a
        // `Vec<Asset>` while still seeding generic helpers like `is_admin`.
        for (caller, sites) in &sites_by_caller {
            let Some(caller_taint) = taint.get(caller) else {
                continue;
            };
            let caller_body = bodies.get(caller).copied();
            for cs in sites {
                for (i, arg) in cs.arg_locals.iter().enumerate() {
                    let Some(arg) = arg else { continue };
                    if !caller_taint.contains(arg) {
                        continue;
                    }
                    let sender_like = caller_body
                        .and_then(|b| b.local_decls.get(*arg))
                        .is_some_and(|d| ty_is_sender_like(tcx, d.ty));
                    if sender_like {
                        seeds
                            .entry(cs.callee)
                            .or_default()
                            .param_locals
                            .insert(Local::from_usize(i + 1));
                    }
                }
            }
        }

        // Capture wiring: a sender-tainted capture at index `k` taints upvar
        // `k` of the closure, again only for sender-compatible types.
        for (enclosing, closure_def, captures) in &closure_captures {
            let Some(caller_taint) = taint.get(enclosing) else {
                continue;
            };
            let caller_body = bodies.get(enclosing).copied();
            for (k, capture) in captures.iter().enumerate() {
                let Some(capture) = capture else { continue };
                if !caller_taint.contains(capture) {
                    continue;
                }
                let sender_like = caller_body
                    .and_then(|b| b.local_decls.get(*capture))
                    .is_some_and(|d| ty_is_sender_like(tcx, d.ty));
                if sender_like {
                    seeds
                        .entry(*closure_def)
                        .or_default()
                        .upvar_indices
                        .insert(k);
                }
            }
        }

        if seeds == before {
            break;
        }
    }

    SenderTaint { targets, seeds }
}

/// Determines the bodies stage 3 works on: the call-graph nodes plus every
/// closure created inside them, transitively. Closures are not call edges but
/// still have to be tainted and analysed. The second return value records, per
/// closure, the enclosing function and the caller locals captured as upvars.
fn collect_targets(
    tcx: TyCtxt<'_>,
    call_graph: &CallGraph,
) -> (Vec<DefId>, Vec<(DefId, DefId, Vec<Option<Local>>)>) {
    let mut targets: Vec<DefId> = call_graph.nodes().to_vec();
    let mut known: HashSet<DefId> = targets.iter().copied().collect();
    let mut closure_captures: Vec<(DefId, DefId, Vec<Option<Local>>)> = Vec::new();

    let mut worklist: Vec<DefId> = targets.clone();
    let mut scanned: HashSet<DefId> = HashSet::new();

    while let Some(f) = worklist.pop() {
        if !scanned.insert(f) {
            continue;
        }
        let Some(body) = cosmwasm::body_of(tcx, f) else {
            continue;
        };
        for (closure_def, captures) in find_closure_captures(body) {
            closure_captures.push((f, closure_def, captures));
            if closure_def.is_local() && known.insert(closure_def) {
                targets.push(closure_def);
                worklist.push(closure_def);
            }
        }
    }

    targets.retain(|&f| cosmwasm::is_analysable(tcx, f));
    (targets, closure_captures)
}

/// Finds every closure created in `body` through an aggregate, returning each
/// closure's `DefId` together with the caller locals captured as its upvars,
/// in upvar-field order (`None` for constant captures).
fn find_closure_captures<'tcx>(body: &Body<'tcx>) -> Vec<(DefId, Vec<Option<Local>>)> {
    let mut out = Vec::new();
    for data in body.basic_blocks.iter() {
        for stmt in &data.statements {
            let StatementKind::Assign(assign) = &stmt.kind else {
                continue;
            };
            if let Rvalue::Aggregate(kind, operands) = &assign.as_ref().1 {
                if let AggregateKind::Closure(closure_def, _) = &**kind {
                    out.push((*closure_def, operands.iter().map(operand_local).collect()));
                }
            }
        }
    }
    out
}

/// Collects all MIR locals that directly or indirectly originate from
/// `cosmwasm_std::MessageInfo::sender`, starting from the interprocedural
/// seeds.
pub fn compute_sender_locals<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    seeds: &SenderSeeds,
) -> HashSet<Local> {
    // Parameters a caller passed a sender-tainted argument to are tainted from
    // the first iteration on.
    let mut sender_locals: HashSet<Local> = seeds.param_locals.clone();

    // A single forward pass can miss definitions that appear after their use in
    // block order, which is common with closure captures, so the propagation
    // iterates until the set of tainted locals stops growing.
    loop {
        let start_len = sender_locals.len();

        for bb_data in body.basic_blocks.iter() {
            for stmt in &bb_data.statements {
                if let StatementKind::Assign(assign) = &stmt.kind {
                    let (lhs, rhs) = assign.as_ref();
                    if rvalue_is_sender_tainted(tcx, body, rhs, &sender_locals, seeds) {
                        sender_locals.insert(lhs.local);
                    }
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

            // Two kinds of call preserve the sender identity and therefore
            // forward the taint to their return value:
            //   * identity-preserving conversions (`as_str`, `to_string`,
            //     `addr_canonicalize`, …), and
            //   * forwarding glue (`?` via `Try::branch`/`from_residual`,
            //     `unwrap`, `ok_or`, `map_err`, …), which hands its payload
            //     straight through.
            // Without the glue the taint would die at the `?` of a fallible
            // conversion, so a guard such as
            //     if deps.api.addr_canonicalize(info.sender.as_str())? != cfg.owner
            // would not be recognised as a sender comparison at all.
            let callee = callee_def_id(tcx, body, func);
            let forwards = callee.is_some_and(|d| {
                is_identity_preserving(tcx.item_name(d).as_str()) || is_forwarding_glue_fn(tcx, d)
            });
            if forwards
                && args
                    .iter()
                    .filter_map(|a| operand_local(&a.node))
                    .any(|l| sender_locals.contains(&l))
            {
                sender_locals.insert(destination.local);
            }
        }

        if sender_locals.len() == start_len {
            break;
        }
    }

    sender_locals
}

/// Returns whether `rhs` passes a sender value on to its left-hand side,
/// either directly out of `info.sender` or derived from an already tainted
/// local.
fn rvalue_is_sender_tainted<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    rhs: &Rvalue<'tcx>,
    sender_locals: &HashSet<Local>,
    seeds: &SenderSeeds,
) -> bool {
    // A place carries the sender taint if it is `info.sender`, reads an
    // already tainted local, or reads a seeded closure upvar.
    let place_tainted = |place: &Place<'tcx>| {
        place_is_sender_field(tcx, body, place)
            || sender_locals.contains(&place.local)
            || place_reads_seeded_upvar(place, seeds)
    };
    match rhs {
        Rvalue::Use(Operand::Copy(place) | Operand::Move(place), _) => place_tainted(place),
        // `&info.sender`, or a reference to an already tainted local such as
        // the capture `&sender` of a closure.
        Rvalue::Ref(_, _, place) => place_tainted(place),
        // A raw pointer taints only where its target really is the sender.
        Rvalue::RawPtr(_, place) => place_tainted(place),
        // An aggregate (closure environment, tuple, struct) counts as tainted
        // when one of its fields carries the sender. That is what makes the
        // closure of `iter().any(|x| x == info.sender)` visible as a
        // sender-carrying argument at the call site.
        Rvalue::Aggregate(_, operands) => operands
            .iter()
            .filter_map(operand_local)
            .any(|l| sender_locals.contains(&l)),
        _ => false,
    }
}

/// Methods that preserve the sender identity, so their return value carries
/// the same taint as their receiver.
fn is_identity_preserving(name: &str) -> bool {
    matches!(
        name,
        "as_ref"
            | "as_str"
            | "as_bytes"
            | "to_string"
            | "to_owned"
            | "clone"
            | "deref"
            | "deref_mut"
            | "borrow"
            // The `cosmwasm_std::Api` address conversions return the *same*
            // identity in a different representation (`&str` to `Addr` to
            // `CanonicalAddr`), so the marking has to survive them: the common
            // dispatcher idiom
            //     if deps.api.addr_canonicalize(info.sender.as_str())? != config.owner
            // is otherwise not a sender check and everything it guards is
            // reported. This stays sound in both directions, because identity
            // preservation only forwards an *already* marked argument —
            // validating an attacker-supplied address still carries attacker
            // taint rather than creating a gate.
            | "addr_validate"
            | "addr_canonicalize"
            | "addr_humanize"
    )
}

/// Returns whether `place` reads `info.sender`.
fn place_is_sender_field<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>, place: &Place<'tcx>) -> bool {
    let local_ty = body.local_decls[place.local].ty;
    let base_ty = match local_ty.kind() {
        TyKind::Ref(_, inner, _) => *inner,
        _ => local_ty,
    };

    let TyKind::Adt(adt_def, _) = base_ty.kind() else {
        return false;
    };
    if !is_message_info(tcx, adt_def.did()) {
        return false;
    }

    // `MessageInfo` is a struct, so it has exactly one variant whose fields
    // resolve the projection's field index to a name.
    let Some(variant) = adt_def.variants().iter().next() else {
        return false;
    };
    place.projection.iter().any(|elem| match elem {
        PlaceElem::Field(field_idx, _) => variant
            .fields
            .get(field_idx)
            .is_some_and(|field| field.name.as_str() == "sender"),
        _ => false,
    })
}

/// Returns whether `place` reads a closure upvar that the interprocedural
/// seeding marked as sender-tainted.
///
/// Inside a closure body the environment is the first local and each captured
/// upvar is a field of it, accessed as `((*_1).k)` for a by-reference capture
/// or `(_1.k)` for a by-value one. Only closures ever receive `upvar_indices`,
/// so this never fires for an ordinary function.
fn place_reads_seeded_upvar(place: &Place<'_>, seeds: &SenderSeeds) -> bool {
    if seeds.upvar_indices.is_empty() || place.local != Local::from_u32(1) {
        return false;
    }
    place.projection.iter().any(|elem| match elem {
        PlaceElem::Field(field_idx, _) => seeds.upvar_indices.contains(&field_idx.as_usize()),
        _ => false,
    })
}

/// Returns whether `ty`, after stripping references, can carry a sender
/// identity: a CosmWasm address type, `String`, or `str`.
///
/// Used to type-gate the interprocedural seeds so that a non-address parameter
/// such as `Vec<Asset>` is never mistaken for the sender, which is the main
/// source of false positives in a naive argument and capture propagation.
pub fn ty_is_sender_like<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    match strip_refs(ty).kind() {
        TyKind::Str => true,
        TyKind::Adt(adt_def, _) => {
            let did = adt_def.did();
            is_cosmwasm_addr(tcx, did) || cosmwasm::item_name_is(tcx, did, "String")
        }
        _ => false,
    }
}
