//! Generic MIR helpers shared by several pipeline stages.
//!
//! Nothing in this module knows about CosmWasm; it only bridges recurring gaps
//! between the shape of MIR and the questions the analysis asks of it.

use std::collections::HashSet;

use rustc_middle::mir::{BasicBlock, Body, Local, Operand, Rvalue};
use rustc_middle::ty::{Ty, TyCtxt, TyKind};
use rustc_span::def_id::DefId;

/// Returns the local an operand reads, or `None` for constants.
pub fn operand_local(operand: &Operand<'_>) -> Option<Local> {
    match operand {
        Operand::Copy(place) | Operand::Move(place) => Some(place.local),
        _ => None,
    }
}

/// Returns the `DefId` of the callee if the call target is a direct function
/// definition. Indirect calls (function pointers, dynamic dispatch) cannot be
/// resolved statically and yield `None`.
pub fn callee_def_id<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    func: &Operand<'tcx>,
) -> Option<DefId> {
    match func.ty(&body.local_decls, tcx).kind() {
        TyKind::FnDef(def_id, _) => Some(*def_id),
        _ => None,
    }
}

/// Peels every reference layer off `ty`, e.g. `&&Addr` to `Addr`.
pub fn strip_refs<'tcx>(ty: Ty<'tcx>) -> Ty<'tcx> {
    let mut ty = ty;
    while let TyKind::Ref(_, inner, _) = ty.kind() {
        ty = *inner;
    }
    ty
}

/// Collects the locals an `Rvalue` reads, from both operands and source places.
pub fn rvalue_locals(rvalue: &Rvalue<'_>) -> Vec<Local> {
    let mut locals = Vec::new();
    let mut push_op = |op: &Operand<'_>| {
        if let Some(l) = operand_local(op) {
            locals.push(l);
        }
    };

    match rvalue {
        Rvalue::Use(op, _)
        | Rvalue::Repeat(op, _)
        | Rvalue::Cast(_, op, _)
        | Rvalue::UnaryOp(_, op) => push_op(op),
        // There is no `Rvalue::Len`: slice lengths appear as
        // `UnaryOp(PtrMetadata, ..)`, which the arm above already covers.
        Rvalue::Ref(_, _, place)
        | Rvalue::RawPtr(_, place)
        | Rvalue::Discriminant(place)
        | Rvalue::CopyForDeref(place) => locals.push(place.local),
        Rvalue::BinaryOp(_, operands) => {
            let (a, b) = operands.as_ref();
            push_op(a);
            push_op(b);
        }
        Rvalue::Aggregate(_, operands) => {
            for op in operands.iter() {
                push_op(op);
            }
        }
        _ => {}
    }

    locals
}

/// Returns every basic block from which at least one block in `targets` is
/// forward-reachable, `targets` themselves included.
///
/// Computed once per body by a single backward traversal, so that "can this
/// branch still reach a successful return?" becomes a set lookup instead of a
/// forward search per branch target.
pub fn blocks_reaching(body: &Body<'_>, targets: &HashSet<BasicBlock>) -> HashSet<BasicBlock> {
    let predecessors = body.basic_blocks.predecessors();
    let mut seen: HashSet<BasicBlock> = HashSet::new();
    let mut stack: Vec<BasicBlock> = targets.iter().copied().collect();

    while let Some(bb) = stack.pop() {
        if !seen.insert(bb) {
            continue;
        }
        for &pred in &predecessors[bb] {
            stack.push(pred);
        }
    }

    seen
}

/// Normalizes a formatted `rustc` type so that it is stable enough to be shown
/// in a report: lifetimes and erased-region markers are dropped, whitespace is
/// collapsed and the commas they leave behind are cleaned up
/// (`Item<'erased, Addr>` becomes `Item<Addr>`).
pub fn normalize_ty_str(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\'' {
            // Lifetime written with an apostrophe: 'erased, '_, 'a
            while let Some(&next) = chars.peek() {
                if next.is_alphanumeric() || next == '_' {
                    chars.next();
                } else {
                    break;
                }
            }
        } else if c == '{' {
            // Anonymous or erased region written without an apostrophe: {erased}
            while let Some(&next) = chars.peek() {
                chars.next();
                if next == '}' {
                    break;
                }
            }
        } else {
            result.push(c);
        }
    }

    let result = result
        .replace(", ,", ",")
        .replace("<,", "<")
        .replace(", >", ">")
        .replace(",>", ">");
    result.split_whitespace().collect::<Vec<_>>().join(" ")
}
