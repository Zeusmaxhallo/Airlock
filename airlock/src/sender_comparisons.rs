//! Stage 3, second half — where the sender is compared.
//!
//! Recognises the program points at which a sender-carrying local
//! ([`crate::sender_taint`]) is compared against another value. MIR expresses
//! such a comparison in several shapes, and each needs its own pattern:
//!
//! * a `BinaryOp` statement, for primitive and `Copy` types;
//! * a call to `PartialEq::eq`/`ne`, for `Addr` and other non-`Copy` types;
//! * a collection or iterator search (`contains`, `any`, `find`, …), where the
//!   sender is checked against a set of principals;
//! * a call into a library authorization helper whose own comparison lives in a
//!   dependency crate and is therefore unavailable as MIR.
//!
//! The counterpart of each comparison — the value the sender is checked
//! against — is what stage 4 traces back to a storage load. Comparisons hidden
//! behind a boolean helper or an `Option` combinator are added by
//! [`crate::predicates`].

use std::collections::HashSet;

use rustc_middle::mir::{
    BinOp, Body, Local, Location, Operand, Rvalue, Statement, StatementKind, Terminator,
    TerminatorKind,
};
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::DefId;

use crate::mir_util::{callee_def_id, operand_local};
use crate::sender_taint::{SenderSeeds, compute_sender_locals};

/// A detected comparison of `info.sender` against another value.
#[derive(Debug, Clone)]
pub struct SenderComparison {
    /// Block and statement at which the comparison occurs.
    pub location: Location,
    /// The value `info.sender` is compared against — the counterpart stage 4
    /// traces back to a storage load.
    pub compared_local: Local,
    /// The comparison operator.
    pub op: BinOp,
    /// Human-readable description for the report.
    pub description: String,
}

/// Finds every comparison of `info.sender` in one body.
pub fn find_sender_comparisons<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    seeds: &SenderSeeds,
) -> Vec<SenderComparison> {
    let mut results = Vec::new();
    let sender_locals = compute_sender_locals(tcx, body, seeds);

    for (block, bb_data) in body.basic_blocks.iter_enumerated() {
        for (statement_index, stmt) in bb_data.statements.iter().enumerate() {
            let location = Location {
                block,
                statement_index,
            };
            check_statement_comparison(stmt, &sender_locals, location, &mut results);
        }

        let location = Location {
            block,
            statement_index: bb_data.statements.len(),
        };
        check_terminator_comparison(
            tcx,
            body,
            bb_data.terminator(),
            &sender_locals,
            location,
            &mut results,
        );
    }

    results
}

/// Detects a `BinaryOp` comparison in a MIR statement, which is the shape
/// primitive and `Copy` types are compared in.
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

    // `info.sender` may stand on either side; both are handled symmetrically.
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
            compared_local: other,
            op: *op,
            description,
        });
    }
}

/// Detects the comparisons that MIR expresses as a call: the `PartialEq`
/// methods for non-`Copy` types such as `Addr`, the collection and iterator
/// searches, and the calls into library authorization helpers.
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
    let Some(callee) = callee_def_id(tcx, body, func) else {
        return;
    };

    let callee_name = tcx.item_name(callee);
    let callee_name = callee_name.as_str();
    let arg_locals: Vec<Option<Local>> = args.iter().map(|a| operand_local(&a.node)).collect();

    match callee_name {
        name @ ("eq" | "ne") => {
            let op = if name == "eq" { BinOp::Eq } else { BinOp::Ne };
            check_eq_call(func, &arg_locals, sender_locals, location, op, results);
        }
        "contains" => {
            check_contains_call(func, &arg_locals, sender_locals, location, results);
        }
        "any" | "find" | "position" | "filter" => {
            check_iter_search_call(func, &arg_locals, sender_locals, location, results);
        }
        _ => {}
    }

    // A call into a known library authorization helper whose actual comparison
    // lives in a dependency crate and is therefore unavailable as MIR. If
    // `info.sender` is passed to it, the call itself is what the analysis can
    // observe, and it is recorded as the detected check.
    if !callee.is_local() && is_auth_sink(callee_name) {
        check_auth_sink_call(
            tcx,
            callee,
            callee_name,
            &arg_locals,
            sender_locals,
            location,
            results,
        );
    }
}

/// Shortened rendering of the call target, for the report only.
fn func_label(func: &Operand<'_>) -> String {
    format!("{:?}", func).chars().take(60).collect()
}

/// Pattern `PartialEq::eq` / `ne`: the two arguments are the compared values.
fn check_eq_call(
    func: &Operand<'_>,
    arg_locals: &[Option<Local>],
    sender_locals: &HashSet<Local>,
    location: Location,
    op: BinOp,
    results: &mut Vec<SenderComparison>,
) {
    let op_symbol = if op == BinOp::Ne { "!=" } else { "==" };
    // `eq`/`ne` take exactly two operands; the counterpart is the other one.
    for (i, arg_local) in arg_locals.iter().enumerate().take(2) {
        let Some(arg_local) = arg_local else { continue };
        if !sender_locals.contains(arg_local) {
            continue;
        }
        if let Some(Some(other_local)) = arg_locals.get(1 - i) {
            results.push(SenderComparison {
                location,
                compared_local: *other_local,
                op,
                description: format!(
                    "Call {:?}: info.sender ({:?}) {} {:?}",
                    func_label(func),
                    arg_local,
                    op_symbol,
                    other_local,
                ),
            });
        }
    }
}

/// Pattern `contains(collection, needle)`, where argument 0 is the collection
/// and argument 1 the value searched for.
///
/// The counterpart of the comparison is the *collection*, because it holds the
/// values the sender is checked against.
fn check_contains_call(
    func: &Operand<'_>,
    arg_locals: &[Option<Local>],
    sender_locals: &HashSet<Local>,
    location: Location,
    results: &mut Vec<SenderComparison>,
) {
    let (Some(collection), Some(needle)) = (
        arg_locals.first().copied().flatten(),
        arg_locals.get(1).copied().flatten(),
    ) else {
        return;
    };
    if !sender_locals.contains(&needle) {
        return;
    }

    // The common shape: `admins.contains(&info.sender)`.
    results.push(SenderComparison {
        location,
        compared_local: collection,
        op: BinOp::Eq,
        description: format!(
            "contains() check '{}': collection ({:?}) contains info.sender ({:?})",
            func_label(func),
            collection,
            needle
        ),
    });
}

/// Pattern `admins.iter().any(|a| a == &info.sender)`.
///
/// The comparison itself happens inside the closure; the call receives the
/// closure and the iterator. When one argument carries the sender — through an
/// upvar of the closure environment — the call is recorded and the other
/// argument is read as the iterated collection.
fn check_iter_search_call(
    func: &Operand<'_>,
    arg_locals: &[Option<Local>],
    sender_locals: &HashSet<Local>,
    location: Location,
    results: &mut Vec<SenderComparison>,
) {
    for (i, arg_local) in arg_locals.iter().enumerate() {
        let Some(arg_local) = arg_local else { continue };
        if !sender_locals.contains(arg_local) {
            continue;
        }
        let other_local = arg_locals
            .iter()
            .enumerate()
            .find(|(j, _)| *j != i)
            .and_then(|(_, l)| *l);

        if let Some(other_local) = other_local {
            results.push(SenderComparison {
                location,
                compared_local: other_local,
                op: BinOp::Eq,
                description: format!(
                    "iter search '{}': info.sender ({:?}) in iterator over ({:?}) ",
                    func_label(func),
                    arg_local,
                    other_local
                ),
            });
        }
    }
}

/// Names of library functions that perform an authorization check internally,
/// comparing the caller against a stored owner or admin. Their body lives in a
/// dependency crate and is not available as MIR, so the *call* — with
/// `info.sender` among its arguments — is what counts as the check.
fn is_auth_sink(name: &str) -> bool {
    // Established helpers of cw-ownable and cw-controllers.
    if matches!(
        name,
        "assert_admin" | "assert_owner" | "assert_only_owner" | "update_ownership" | "is_admin"
    ) {
        return true;
    }

    // Project-specific authorization helpers living in a sibling workspace
    // crate. Those are cross-crate calls, so their MIR — and with it the sender
    // comparison — is unavailable and only the call itself is observable. They
    // are recognised by naming convention: an asserting or querying verb
    // combined with an authorization noun, as in `check_admin_privileges`,
    // `ensure_owner`, `only_operator`, `verify_permissions`, or the widespread
    // boolean predicates `OWNER.is_owner(deps, &info.sender)?` and
    // `is_approved_or_owner(..)`.
    //
    // The convention is deliberately narrow, so that a real vulnerability is
    // not masked:
    //   * `validate_*` is excluded, because in CosmWasm it overwhelmingly means
    //     *format* validation (`addr_validate`), not authorization;
    //   * reading verbs (`get_`, `load_`, `query_`) are not asserting verbs.
    // Two further conditions are enforced by the caller and keep the rule
    // sound: the call must receive `info.sender` as an argument, and its result
    // must feed a branch that abandons the success path — an ignored result
    // never gates, which the effective-guard criterion of stage 5 enforces.
    const VERBS: [&str; 7] = [
        "assert_", "check_", "ensure_", "require_", "verify_", "only_", "is_",
    ];
    const NOUNS: [&str; 7] = [
        "admin",
        "owner",
        "auth",
        "privilege",
        "permission",
        "operator",
        "minter",
    ];
    VERBS.iter().any(|v| name.starts_with(v)) && NOUNS.iter().any(|n| name.contains(n))
}

/// Records a call to a library authorization helper as a detected sender
/// check, provided `info.sender` is among its arguments.
///
/// The stored owner or admin it is compared against is internal to the
/// dependency, so there is no counterpart in caller space; the comparison
/// points at the sender itself and the description names the helper.
fn check_auth_sink_call(
    tcx: TyCtxt<'_>,
    callee: DefId,
    name: &str,
    arg_locals: &[Option<Local>],
    sender_locals: &HashSet<Local>,
    location: Location,
    results: &mut Vec<SenderComparison>,
) {
    let sender_arg = arg_locals
        .iter()
        .filter_map(|a| *a)
        .find(|l| sender_locals.contains(l));
    if let Some(sender_local) = sender_arg {
        results.push(SenderComparison {
            location,
            compared_local: sender_local,
            op: BinOp::Eq,
            description: format!(
                "auth-sink '{}::{}': info.sender ({:?}) checked against stored owner/admin",
                tcx.crate_name(callee.krate),
                name,
                sender_local
            ),
        });
    }
}

/// The local that holds the result of a comparison: the assigned place for a
/// statement comparison, the call destination for a comparison expressed as a
/// call.
pub fn comparison_result_local<'tcx>(body: &Body<'tcx>, cmp: &SenderComparison) -> Option<Local> {
    let block = &body.basic_blocks[cmp.location.block];
    if cmp.location.statement_index < block.statements.len() {
        match &block.statements[cmp.location.statement_index].kind {
            StatementKind::Assign(assign) => Some(assign.as_ref().0.local),
            _ => None,
        }
    } else {
        match &block.terminator().kind {
            TerminatorKind::Call { destination, .. } => Some(destination.local),
            _ => None,
        }
    }
}
