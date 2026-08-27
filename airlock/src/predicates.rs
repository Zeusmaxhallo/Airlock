//! Stage 3, second half — sender checks hidden behind a predicate.
//!
//! Not every authorization check is a comparison in the handler itself. The
//! dominant CosmWasm idioms put it behind a boolean helper
//! (`if !is_trusted(&info.sender, &config) { return Err(..) }`) or inside the
//! closure of an `Option` combinator
//! (`config.owner.map_or(true, |o| o != info.sender)`). In both cases the
//! comparison is found by stage 3 — but in the *helper's* body, while the
//! branch that enforces it sits in the caller.
//!
//! This module bridges that gap by synthesising a [`SenderComparison`] at the
//! call site, whose result local is the call's destination. The gating of
//! stage 5 then treats the branch on that result exactly like a direct
//! `if info.sender != owner` guard.

use std::collections::{HashMap, HashSet};

use rustc_middle::mir::{
    BinOp, Body, Local, Location, Operand, Rvalue, StatementKind, TerminatorKind,
};
use rustc_middle::ty::{TyCtxt, TyKind};
use rustc_span::def_id::DefId;

use crate::cosmwasm;
use crate::mir_util::{callee_def_id, operand_local};
use crate::sender_comparisons::{SenderComparison, comparison_result_local};

/// Functions and closures whose boolean result *is* an `info.sender` equality
/// comparison, keyed by that comparison's operator.
///
/// Consumed by [`option_predicate_comparisons`]: to decide whether an
/// `Option` combinator authorises, the polarity of the comparison inside its
/// closure has to be known.
pub fn sender_predicate_summary(
    tcx: TyCtxt<'_>,
    fn_comparisons: &HashMap<DefId, Vec<SenderComparison>>,
) -> HashMap<DefId, BinOp> {
    let mut out = HashMap::new();
    for (&def_id, comparisons) in fn_comparisons {
        if comparisons.is_empty() {
            continue;
        }
        let Some(body) = cosmwasm::body_of(tcx, def_id) else {
            continue;
        };
        let return_place = Local::from_usize(0);
        let feeders = return_value_feeders(body);
        for cmp in comparisons {
            if !matches!(cmp.op, BinOp::Eq | BinOp::Ne) {
                continue;
            }
            let Some(result) = comparison_result_local(body, cmp) else {
                continue;
            };
            if result == return_place || feeders.contains(&result) {
                out.insert(def_id, cmp.op);
                break;
            }
        }
    }
    out
}

/// The locals that flow into the return place in one hop, either directly
/// (`_0 = move _tmp`) or wrapped in an aggregate. The latter covers the very
/// common `fn is_owner(..) -> StdResult<bool>` shape, where the comparison
/// result is returned as `_0 = Ok(_tmp)`.
fn return_value_feeders<'tcx>(body: &Body<'tcx>) -> HashSet<Local> {
    let return_place = Local::from_usize(0);
    let mut feeders = HashSet::new();
    for data in body.basic_blocks.iter() {
        for stmt in data.statements.iter() {
            let StatementKind::Assign(assign) = &stmt.kind else {
                continue;
            };
            let (place, rvalue) = assign.as_ref();
            if place.local != return_place || !place.projection.is_empty() {
                continue;
            }
            match rvalue {
                Rvalue::Use(op, _) => feeders.extend(operand_local(op)),
                // `Ok(cmp)` / `Some(cmp)`: predicate helpers that return a
                // fallible bool and are called with `?`.
                Rvalue::Aggregate(_, operands) => {
                    feeders.extend(operands.iter().filter_map(operand_local))
                }
                _ => {}
            }
        }
    }
    feeders
}

/// Local functions that return a boolean *and* compare `info.sender`
/// somewhere in their body — the relaxed criterion behind
/// [`predicate_call_comparisons`].
///
/// [`sender_predicate_summary`] requires the comparison result to flow into
/// the return value, which only holds for single-expression predicates such as
/// `|o| o != sender`. Real authorization helpers usually branch instead:
///
/// ```ignore
/// pub fn is_trusted(sender: &Addr, config: &Config) -> bool {
///     let mut trusted = false;
///     if sender == config.owner { trusted = true; }   // a flag, not the return
///     ...
///     trusted
/// }
/// fn is_approved_or_owner(deps: Deps, spender: &Addr, id: &str) -> StdResult<bool> {
///     if owner == *spender { return Ok(true); }       // an early constant return
///     ...
/// }
/// ```
///
/// Requiring only "boolean return plus a sender comparison in the body" is
/// looser, but the decisive restriction sits on the caller side: the result
/// must feed a branch that abandons the success path. A boolean helper that
/// inspects the sender and whose result triggers an early `Err` is an
/// authorization check for all practical purposes, whereas one whose result
/// merely selects a fee never becomes a gate.
pub fn bool_predicate_fns(
    tcx: TyCtxt<'_>,
    fn_comparisons: &HashMap<DefId, Vec<SenderComparison>>,
) -> HashSet<DefId> {
    let mut out = HashSet::new();
    for (&def_id, comparisons) in fn_comparisons {
        if comparisons.is_empty() {
            continue;
        }
        let Some(body) = cosmwasm::body_of(tcx, def_id) else {
            continue;
        };
        let returns_bool = match body.local_decls[Local::from_usize(0)].ty.kind() {
            TyKind::Bool => true,
            // `Result<bool, E>` / `Option<bool>`: the payload is the first
            // type argument.
            TyKind::Adt(_, args) => args
                .types()
                .next()
                .is_some_and(|t| matches!(t.kind(), TyKind::Bool)),
            _ => false,
        };
        if returns_bool {
            out.insert(def_id);
        }
    }
    out
}

/// Recognises a direct call to a boolean sender-predicate helper (per
/// [`bool_predicate_fns`]) and records it as a detected sender check:
///
/// ```ignore
/// if !is_trusted(&info.sender, &config) { return Err(Unauthorized {}) }
/// if !is_approved_or_owner(deps.as_ref(), &info.sender, &token_id)? { .. }
/// require!(OWNER.is_owner(deps.as_ref(), &info.sender)?, Unauthorized);
/// ensure!(contract.is_owner_or_operator(deps.storage, info.sender.as_str())?, ..);
/// ```
///
/// The synthetic comparison uses a non-equality operator on purpose. The
/// edge-sensitive gating of stage 5 only considers `Eq`/`Ne` and therefore
/// skips these, which is required for soundness: for a boolean helper the
/// polarity is not recoverable — `if is_owner` and `if !is_owner` alias to the
/// same local once the negation is erased. Such a comparison may only drive
/// the divergence-based gating, which is polarity-independent because it
/// demands a branch that abandons the success path.
pub fn predicate_call_comparisons<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    bool_predicates: &HashSet<DefId>,
) -> Vec<SenderComparison> {
    let mut out = Vec::new();
    if bool_predicates.is_empty() {
        return out;
    }

    for (block, data) in body.basic_blocks.iter_enumerated() {
        let TerminatorKind::Call {
            func, destination, ..
        } = &data.terminator().kind
        else {
            continue;
        };
        let Some(callee) = callee_def_id(tcx, body, func) else {
            continue;
        };
        if !bool_predicates.contains(&callee) {
            continue;
        }
        out.push(SenderComparison {
            location: Location {
                block,
                statement_index: data.statements.len(),
            },
            compared_local: destination.local,
            op: BinOp::Ge,
            description: format!(
                "sender-predicate helper '{}': result branches as an authorization guard",
                tcx.item_name(callee)
            ),
        });
    }
    out
}

/// Recognises an `Option` combinator whose closure argument is a sender
/// predicate (per [`sender_predicate_summary`]) and synthesises a comparison
/// at the call.
///
/// Only the sound shapes are emitted: the value the combinator yields when the
/// principal is absent — the `None` case — must be the **non-authorising**
/// outcome, so that a missing owner can never silently authorise a write:
///
/// ```ignore
/// x.map_or(true,  |o| o != sender)   // None -> true  -> reject, op = Ne
/// x.is_none_or(   |o| o != sender)   // None -> true  -> reject, op = Ne
/// x.map_or(false, |o| o == sender)   // None -> false -> reject, op = Eq
/// x.is_some_and(  |o| o == sender)   // None -> false -> reject, op = Eq
/// ```
///
/// Any other pairing of default value and operator is skipped, which keeps the
/// rule strictly false-positive-reducing and never vulnerability-hiding.
pub fn option_predicate_comparisons<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    sender_predicate: &HashMap<DefId, BinOp>,
) -> Vec<SenderComparison> {
    let mut out = Vec::new();
    if sender_predicate.is_empty() {
        return out;
    }

    for (block, data) in body.basic_blocks.iter_enumerated() {
        let TerminatorKind::Call {
            func,
            args,
            destination,
            ..
        } = &data.terminator().kind
        else {
            continue;
        };
        let Some(callee) = callee_def_id(tcx, body, func) else {
            continue;
        };
        let combinator = tcx.item_name(callee);
        let combinator = combinator.as_str();
        if !matches!(combinator, "map_or" | "is_some_and" | "is_none_or") {
            continue;
        }

        // The closure argument, resolved to its sender-comparison operator.
        let mut op: Option<BinOp> = None;
        for arg in args.iter() {
            if let TyKind::Closure(closure_def, _) = arg.node.ty(&body.local_decls, tcx).kind() {
                if let Some(&o) = sender_predicate.get(closure_def) {
                    op = Some(o);
                }
            }
        }
        let Some(op) = op else {
            continue;
        };

        // The value the combinator produces when the principal is absent.
        let none_value = match combinator {
            "is_some_and" => false,
            "is_none_or" => true,
            // `map_or(default, f)`: the `bool`-typed argument.
            _ => {
                let mut default = None;
                for arg in args.iter() {
                    if matches!(arg.node.ty(&body.local_decls, tcx).kind(), TyKind::Bool) {
                        default = const_bool_operand(body, &arg.node);
                    }
                }
                let Some(default) = default else { continue };
                default
            }
        };

        // Soundness condition: `None` must yield the non-authorising outcome.
        // For `Ne` the authorising edge is the `false` result, so `None` has to
        // be `true`; for `Eq` it is the other way round.
        let sound = match op {
            BinOp::Ne => none_value,
            BinOp::Eq => !none_value,
            _ => false,
        };
        if !sound {
            continue;
        }

        out.push(SenderComparison {
            location: Location {
                block,
                statement_index: data.statements.len(),
            },
            // A nominal counterpart: the gating keys off the operator and the
            // comparison result — the call destination — not off this local.
            compared_local: destination.local,
            op,
            description: format!("Option::{} sender-predicate gate", combinator),
        });
    }
    out
}

/// Reads a constant `bool` from an operand, following one hop through a
/// `_x = const <bool>` assignment for the case where the literal was
/// materialised into a temporary, as the optimized MIR frequently does for
/// call arguments.
fn const_bool_operand<'tcx>(body: &Body<'tcx>, op: &Operand<'tcx>) -> Option<bool> {
    if let Some(c) = op.constant() {
        return c.const_.try_to_bool();
    }

    let local = operand_local(op)?;
    for data in body.basic_blocks.iter() {
        for stmt in data.statements.iter() {
            let StatementKind::Assign(assign) = &stmt.kind else {
                continue;
            };
            let (place, rvalue) = assign.as_ref();
            if place.local != local || !place.projection.is_empty() {
                continue;
            }
            if let Rvalue::Use(inner, _) = rvalue {
                if let Some(c) = inner.constant() {
                    return c.const_.try_to_bool();
                }
            }
        }
    }
    None
}
