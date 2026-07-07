use crate::utility;
use std::collections::HashSet;

use rustc_middle::{
    mir::{
        BinOp, Body, Local, Location, Operand, Place, PlaceElem, Rvalue, Statement, StatementKind,
        Terminator, TerminatorKind,
    },
    ty::{TyCtxt, TyKind},
};

use crate::storage_inventory::{self, StorageInventory};

#[derive(Debug, Clone)]
pub struct SenderComparison {
    /// The block and statement where the comparison occurs.
    pub location: Location,
    /// The local variable containing `info.sender` (either the left- or right-hand side).
    pub sender_local: Local,
    /// The local variable compared against `info.sender` — the potentially tainted variable.
    pub compared_local: Local,
    /// The comparison operator (Eq, Ne, etc.).
    pub op: BinOp,
    /// A human-readable description of the comparison.
    pub description: String,
}

pub fn analyze_function<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    fn_name: &str,
    storage_inventory: &mut StorageInventory,
) -> Vec<SenderComparison> {
    let comparisons = find_sender_comparisons(tcx, body);

    let function_name = fn_name.split("::").last().unwrap_or(fn_name);
    eprintln!("\nchecking function: '{}'",function_name);

    if comparisons.is_empty() {
        eprintln!("[1] info.sender comparisons: no");
    } else {
        eprintln!("[1] info.sender comparisons: {}", comparisons.len());
        for cmp in &comparisons{
            eprintln!("\t{:?} {}", cmp.location, cmp.description);
        }
    }
    

    comparisons
}
/// find all info.sender comparisons
fn find_sender_comparisons<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) -> Vec<SenderComparison> {
    let mut results = Vec::new();
    let sender_locals = collect_sender_locals(tcx, body);

    for (bb_idx, bb_data) in body.basic_blocks.iter_enumerated() {
        for (stmt_idx, stmt) in bb_data.statements.iter().enumerate() {
            let location = Location {
                block: bb_idx,
                statement_index: stmt_idx,
            };
            check_statement_comparison(stmt, &sender_locals, location, &mut results);
        }

        let term_location = Location {
            block: bb_idx,
            statement_index: bb_data.statements.len(),
        };
        check_terminator_comparison(
            tcx,
            body,
            bb_data.terminator(),
            &sender_locals,
            term_location,
            &mut results,
        );
    }

    results
}

/// Detects BinOp comparisons in MIR statements (primitive / Copy types).
fn check_statement_comparison(
    stmt: &Statement<'_>,
    sender_locals: &HashSet<Local>,
    location: Location,
    results: &mut Vec<SenderComparison>,
) {
    let StatementKind::Assign(assign) = &stmt.kind else {
        return;
    };

    let (lhs, rvalue) = assign.as_ref();

    let Rvalue::BinaryOp(op, operands) = rvalue else {
        return;
    };

    if !matches!(
        op,
        BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
    ) {
        return;
    }

    let (left, right) = operands.as_ref();
    let (Some(left_local), Some(right_local)) = (operand_local(left), operand_local(right)) else {
        return;
    };

    // "info.sender" can appear on either side; handle both cases symmetrically.
    for (sender, other, sender_is_left) in [
        (left_local, right_local, true),
        (right_local, left_local, false),
    ] {
        if !sender_locals.contains(&sender) {
            continue;
        }
        let description = if sender_is_left {
            format!(
                "BinOp {:?} info.sender ({:?}) {:?} {:?}",
                lhs.local, sender, op, other,
            )
        } else {
            format!(
                "BinOp {:?}: {:?} {:?} info.sender ({:?})",
                lhs.local, other, op, sender
            )
        };
        results.push(SenderComparison {
            location,
            sender_local: sender,
            compared_local: other,
            op: *op,
            description: description,
        });
    }
}

fn check_terminator_comparison<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    terminator: &Terminator<'tcx>,
    sender_locals: &HashSet<Local>,
    location: Location,
    results: &mut Vec<SenderComparison>,
) {
    let TerminatorKind::Call { func, args, .. } = &terminator.kind else {
        return;
    };

    let func_debug = format!("{:?}", func);
    let callee_name = utility::callee_def_id(tcx, body, func).map(|d| tcx.item_name(d).to_string());
    let arg_locals: Vec<Option<Local>> = args.iter().map(|a| operand_local(&a.node)).collect();

    match callee_name.as_deref() {
        Some(name @ ("eq" | "ne")) => {
            let op = if name == "eq" { BinOp::Eq } else { BinOp::Ne };
            check_eq_call(
                &func_debug,
                &arg_locals,
                sender_locals,
                location,
                op,
                results,
            );
        }
        Some("contains") => {
            check_contains_call(&func_debug, &arg_locals, sender_locals, location, results);
        }
        Some("any" | "find" | "position" | "filter") => {
            check_iter_search_call(&func_debug, &arg_locals, sender_locals, location, results);
        }
        _ => {}
    }
}

/// Pattern: PartialEq::eq / ne — args[0] and args[1] are the compared values.
fn check_eq_call(
    func_debug: &str,
    arg_locals: &[Option<Local>],
    sender_locals: &HashSet<Local>,
    location: Location,
    op: BinOp,
    results: &mut Vec<SenderComparison>,
) {
    let op_symbol = if op == BinOp::Ne { "!=" } else { "==" };
    for (i, arg_local_opt) in arg_locals.iter().enumerate() {
        if let Some(arg_local) = arg_local_opt {
            if sender_locals.contains(arg_local) {
                let other_idx = 1 - i;
                if let Some(Some(other_local)) = arg_locals.get(other_idx) {
                    results.push(SenderComparison {
                        location,
                        sender_local: *arg_local,
                        compared_local: *other_local,
                        op: op,
                        description: format!(
                            "Call {:?}: info.sender ({:?}) {} {:?}",
                            func_debug.chars().take(60).collect::<String>(),
                            arg_local,
                            op_symbol,
                            other_local,
                        ),
                    })
                }
            }
        }
    }
}

/// Pattern: contains(collection, needle)
///   args[0] = &collection (self)
///   args[1] = needle (the value being searched for)
///
/// The tainted element is the *collection* (args[0]), because it contains
/// values that are checked against the sender.
fn check_contains_call(
    func_debug: &str,
    arg_locals: &[Option<Local>],
    sender_locals: &HashSet<Local>,
    location: Location,
    results: &mut Vec<SenderComparison>,
) {
    let collection_local = arg_locals.get(0).copied().flatten();
    let needle_local = arg_locals.get(1).copied().flatten();

    let sender_is_needle = needle_local
        .map(|l| sender_locals.contains(&l))
        .unwrap_or(false);

    let sender_is_collection = collection_local
        .map(|l| sender_locals.contains(&l))
        .unwrap_or(false);

    if sender_is_needle {
        // Most common case: admins.contains(&info.sender)
        if let Some(collection) = collection_local {
            results.push(SenderComparison {
                location,
                sender_local: needle_local.unwrap(),
                compared_local: collection,
                op: BinOp::Eq,
                description: format!(
                    "contains() check '{}': collection ({:?}) contains info.sender ({:?})",
                    func_debug.chars().take(60).collect::<String>(),
                    collection,
                    needle_local.unwrap()
                ),
            });
        } else if sender_is_collection {
            // Unusual, but possible: sender.contains(&value)
            if let Some(needle) = needle_local {
                results.push(SenderComparison {
                    location,
                    sender_local: collection_local.unwrap(),
                    compared_local: needle,
                    op: BinOp::Eq,
                    description: format!(
                        "contains() check `{}`: sender-collection ({:?}) contains ({:?})",
                        func_debug.chars().take(60).collect::<String>(),
                        collection_local.unwrap(),
                        needle,
                    ),
                });
            }
        }
    }
}

/// Pattern: admins.iter().any(|a| a == &info.sender)
/// The sender comparison happens inside the closure; the call itself receives
/// the closure and iterator as arguments. If one of the arguments is a
/// sender_local (e.g., through an upvar), the call is recorded and the other
/// argument is interpreted as the iterator/collection.
fn check_iter_search_call(
    func_debug: &str,
    arg_locals: &[Option<Local>],
    sender_locals: &HashSet<Local>,
    location: Location,
    results: &mut Vec<SenderComparison>,
) {
    for (i, arg_local_opt) in arg_locals.iter().enumerate() {
        if let Some(arg_local) = arg_local_opt {
            if sender_locals.contains(arg_local) {
                let other_local_opt = arg_locals
                    .iter()
                    .enumerate()
                    .find(|(j, _)| *j != i)
                    .and_then(|(_, l)| *l);

                if let Some(other_local) = other_local_opt {
                    results.push(SenderComparison {
                        location,
                        sender_local: *arg_local,
                        compared_local: other_local,
                        op: BinOp::Eq,
                        description: format!(
                            "iter search '{}': info.sender ({:?}) in iterator over ({:?}) ",
                            func_debug.chars().take(60).collect::<String>(),
                            arg_local,
                            other_local
                        ),
                    });
                }
            }
        }
    }
}

/// Collects all MIR locals that directly or indirectly originate from `cosmwasm_std::MessageInfo.sender`
fn collect_sender_locals<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) -> HashSet<Local> {
    let mut sender_locals = HashSet::new();

    for (_, bb_data) in body.basic_blocks.iter_enumerated() {
        for stmt in &bb_data.statements {
            if let StatementKind::Assign(expr) = &stmt.kind {
                let (lhs, rhs) = expr.as_ref();

                match rhs {
                    Rvalue::Use(Operand::Copy(place), _) | Rvalue::Use(Operand::Move(place), _) => {
                        if place_is_sender_field(tcx, body, place) {
                            sender_locals.insert(lhs.local);
                        }
                    }
                    Rvalue::Ref(_, _, place) => {
                        if place_is_sender_field(tcx, body, place) {
                            sender_locals.insert(lhs.local);
                        }
                    }
                    Rvalue::RawPtr(_, _) => {
                        sender_locals.insert(lhs.local);
                    }
                    _ => {}
                }
                if let Rvalue::Use(op, _) = rhs {
                    if let Some(src_local) = operand_local(op) {
                        if sender_locals.contains(&src_local) {
                            sender_locals.insert(lhs.local);
                        }
                    }
                }
            }
        }
    }

    sender_locals
}

/// Checks if an operand is a local variable and returns it
fn operand_local(operand: &Operand<'_>) -> Option<Local> {
    match operand {
        Operand::Copy(place) | Operand::Move(place) => Some(place.local),
        _ => None,
    }
}

/// Check if Place is "info.sender"
fn place_is_sender_field<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>, place: &Place<'tcx>) -> bool {
    let local = place.local;
    let local_ty = body.local_decls[local].ty;

    // Determine the base type — dereference if necessary
    let base_ty = match local_ty.kind() {
        TyKind::Ref(_, inner, _) => *inner,
        _ => local_ty,
    };

    // Is the base type "MessageInfo"?
    let is_message_info = match base_ty.kind() {
        TyKind::Adt(adt_def, _) => {
            let path = tcx.def_path_str(adt_def.did());
            path == "cosmwasm_std::MessageInfo"
        }
        _ => false,
    };

    if !is_message_info {
        return false;
    }

    // Does the place contain a field projection to `sender`?
    for elem in place.projection.iter() {
        if let PlaceElem::Field(field_idx, _) = elem {
            // Resolve the field name using the ADT definition
            if let TyKind::Adt(adt_def, _) = base_ty.kind() {
                // `MessageInfo` is a struct with a single variant
                if let Some(variant) = adt_def.variants().iter().next() {
                    if let Some(field) = variant.fields.get(field_idx) {
                        let field_name = field.name.as_str();
                        if field_name == "sender" {
                            return true;
                        }
                    }
                }
            }
        }
    }

    false
}
