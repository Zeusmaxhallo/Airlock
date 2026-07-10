use crate::{
    call_graph::{CallGraph, CallSite}, storage_inventory::AuthStateVariable, utility::{self, callee_def_id, is_forwarding_glue_fn, is_result_def, is_storage_load_fn, normalize_ty_str},
};
use rustc_hir::{def::DefKind, def_id::DefId};
use rustc_middle::{
    mir::{
        AggregateKind, BasicBlock, BinOp, Body, Local, Location, Operand, Place, PlaceElem, Rvalue,
        Statement, StatementKind, Terminator, TerminatorEdges, TerminatorKind,
    },
    ty::{Ty, TyCtxt, TyKind},
};
use rustc_mir_dataflow::{Analysis, Forward, JoinSemiLattice, fmt::DebugWithContext};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Pre-tainted entry points of a function, produced by interprocedural
/// propagation over the call graph. Both kinds seed the intraprocedural
/// taint before it is computed for a body:
/// * `param_locals` — formal parameters (`_1`, `_2`, …) that a caller passed a
///   sender-tainted argument to (e.g. `is_admin(&info.sender)`).
/// * `upvar_indices` — closure upvar field indices whose captured value was
///   sender-tainted in the enclosing function (e.g. `|a| a == info.sender`,
///   where `info.sender` is captured). Inside the closure the upvar is read as
///   a field of the environment local `_1`.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SenderSeeds {
    pub param_locals: HashSet<Local>,
    pub upvar_indices: HashSet<usize>,
}

use crate::storage_inventory::StorageInventory;

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
    seeds: &SenderSeeds,
    storage_inventory: &mut StorageInventory,
) -> Vec<SenderComparison> {
    let comparisons = find_sender_comparisons(tcx, body, seeds);

    let function_name = fn_name.split("::").last().unwrap_or(fn_name);
    eprintln!("\nchecking function: '{}'", function_name);

    if comparisons.is_empty() {
        eprintln!("[1] info.sender comparisons: none");
    } else {
        eprintln!("[1] info.sender comparisons: {}", comparisons.len());
        for cmp in &comparisons {
            eprintln!("\t{:?} {}", cmp.location, cmp.description);
        }
    }

    let auth_vars = find_auth_state_variables(tcx, body, &comparisons);
    storage_inventory
        .auth_state_variables
        .extend(auth_vars.clone());
    storage_inventory.update_auth_state(&auth_vars);

    if auth_vars.is_empty() {
        eprintln!("[2] Auth-State-Variables: none");
    } else {
        eprintln!("[2] Auth-State-Variables: {}", auth_vars.len());
        for auth_var in &auth_vars {
            eprintln!(
                "\t{} (load @ {:?})",
                auth_var.symbolic_name, auth_var.load_location
            );
        }
    }

    comparisons
}
/// find all info.sender comparisons
fn find_sender_comparisons<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    seeds: &SenderSeeds,
) -> Vec<SenderComparison> {
    let mut results = Vec::new();
    let sender_locals = compute_sender_locals(tcx, body, seeds);

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
    let callee = utility::callee_def_id(tcx, body, func);
    let callee_name = callee.map(|d| tcx.item_name(d).to_string());
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

    // Auth-sink: a call to a known library authorization helper whose actual
    // comparison lives in a dependency crate (non-local) and is therefore not
    // available as MIR to analyse. If info.sender is passed to it, we record
    // the call itself as a detected access-control check. This closes the
    // recall gap for the dominant CosmWasm pattern — `cw_ownable::assert_owner`,
    // `cw_controllers::Admin::assert_admin`, `cw_ownable::update_ownership`, …
    if let (Some(callee), Some(name)) = (callee, callee_name.as_deref()) {
        if !callee.is_local() && is_auth_sink(name) {
            check_auth_sink_call(
                tcx,
                callee,
                name,
                &arg_locals,
                sender_locals,
                location,
                results,
            );
        }
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

/// Names of well-known CosmWasm library functions that perform an
/// authorization check internally (comparing the caller against a stored
/// owner/admin). Their body lives in a dependency crate and is not available
/// as MIR, so the *call* — with `info.sender` as an argument — is what we
/// treat as the check.
fn is_auth_sink(name: &str) -> bool {
    matches!(
        name,
        "assert_admin" | "assert_owner" | "assert_only_owner" | "update_ownership" | "is_admin"
    )
}

/// Records a call to a known authorization helper (see [`is_auth_sink`]) as a
/// detected sender check, provided `info.sender` is among its arguments. The
/// stored owner/admin it is compared against is internal to the dependency, so
/// there is no `compared_local` in caller space; the description names the sink.
fn check_auth_sink_call<'tcx>(
    tcx: TyCtxt<'tcx>,
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
        let krate = tcx.crate_name(callee.krate);
        results.push(SenderComparison {
            location,
            sender_local,
            compared_local: sender_local,
            op: BinOp::Eq,
            description: format!(
                "auth-sink '{}::{}': info.sender ({:?}) checked against stored owner/admin",
                krate, name, sender_local
            ),
        });
    }
}

/// Collects all MIR locals that directly or indirectly originate from
/// `cosmwasm_std::MessageInfo.sender`, seeded with any interprocedurally
/// pre-tainted parameters / closure upvars in `seeds`.
pub fn compute_sender_locals<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    seeds: &SenderSeeds,
) -> HashSet<Local> {
    // Start from the interprocedural seeds: parameters a caller passed a
    // sender-tainted argument to are tainted from the first iteration.
    let mut sender_locals: HashSet<Local> = seeds.param_locals.clone();

    // Fixed-point taint propagation: A single forward pass can miss cases
    // where the definition of a local appears after its use in block order
    // (common with closure captures). Therefore, iterate until the set of
    // sender-tainted locals no longer grows.
    loop {
        let start_len = sender_locals.len();

        for (_, bb_data) in body.basic_blocks.iter_enumerated() {
            for stmt in &bb_data.statements {
                if let StatementKind::Assign(expr) = &stmt.kind {
                    let (lhs, rhs) = expr.as_ref();
                    if rvalue_is_sender_tainted(tcx, body, rhs, &sender_locals, seeds) {
                        sender_locals.insert(lhs.local);
                    }
                }
            }

            // Identity-preserving calls (as_ref/as_str/to_string/clone/...)
            // propagate the sender taint to their return value. Example:
            // `let sender = info.sender.to_string();` -> `sender` becomes
            // tainted, making later checks such as
            // `iter().any(|v| *v == sender)` visible through closure captures.
            if let TerminatorKind::Call {
                func,
                args,
                destination,
                ..
            } = &bb_data.terminator().kind
            {
                let callee_name =
                    utility::callee_def_id(tcx, body, func).map(|d| tcx.item_name(d).to_string());
                if callee_name
                    .as_deref()
                    .map(is_identity_preserving)
                    .unwrap_or(false)
                {
                    let any_arg_tainted = args
                        .iter()
                        .filter_map(|a| operand_local(&a.node))
                        .any(|l| sender_locals.contains(&l));
                    if any_arg_tainted {
                        sender_locals.insert(destination.local);
                    }
                }
            }
        }

        if sender_locals.len() == start_len {
            break;
        }
    }

    sender_locals
}

/// True, wenn `rhs` einen Sender-Wert an sein LHS weitergibt: entweder direkt
/// aus `info.sender` oder abgeleitet aus einem bereits getainteten Local.
fn rvalue_is_sender_tainted<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    rhs: &Rvalue<'tcx>,
    sender_locals: &HashSet<Local>,
    seeds: &SenderSeeds,
) -> bool {
    // A place carries the sender taint if it is `info.sender`, reads an
    // already-tainted local, or reads a seeded closure upvar.
    let place_tainted = |place: &Place<'tcx>| {
        place_is_sender_field(tcx, body, place)
            || sender_locals.contains(&place.local)
            || place_reads_seeded_upvar(place, seeds)
    };
    match rhs {
        Rvalue::Use(Operand::Copy(place), _) | Rvalue::Use(Operand::Move(place), _) => {
            place_tainted(place)
        }
        // &info.sender ODER eine Referenz auf ein bereits getaintetes Local
        // (z. B. der Capture `&sender` einer Closure).
        Rvalue::Ref(_, _, place) => place_tainted(place),
        // RawPtr nur noch, wenn das Ziel tatsaechlich der Sender ist. Zuvor wurde
        // JEDER RawPtr unbedingt getaintet -> bekannte False-Positive-Quelle.
        Rvalue::RawPtr(_, place) => place_tainted(place),
        // Aggregat (Closure-Env, Tupel, Struct): enthaelt eines der Felder den
        // Sender, gilt das Aggregat als sender-getaintet. Dadurch wird die Closure
        // von `iter().any(|x| x == info.sender)` am Call-Site als sender-tragendes
        // Argument sichtbar und `check_iter_search_call` kann feuern.
        Rvalue::Aggregate(_, operands) => operands
            .iter()
            .filter_map(|op| operand_local(op))
            .any(|l| sender_locals.contains(&l)),
        _ => false,
    }
}

/// Methoden, die die Sender-Identitaet erhalten (Ref/Deref/String-Konversion).
/// Ihr Rueckgabewert traegt denselben Sender-Taint wie ihr Empfaenger.
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
    )
}

/// Checks if an operand is a local variable and returns it
fn operand_local(operand: &Operand<'_>) -> Option<Local> {
    match operand {
        Operand::Copy(place) | Operand::Move(place) => Some(place.local),
        _ => None,
    }
}

/// True if `place` reads a closure upvar that interprocedural propagation has
/// marked as sender-tainted. Inside a closure body the environment is the
/// first local (`_1`) and each captured upvar is a field of it — accessed as
/// `((*_1).k)` (by-ref capture) or `(_1.k)` (by-value). We treat a place as a
/// tainted-upvar read if its base is `_1` and it projects a field whose index
/// is in `seeds.upvar_indices`. Only closures ever receive `upvar_indices`, so
/// this never fires for ordinary functions.
fn place_reads_seeded_upvar(place: &Place<'_>, seeds: &SenderSeeds) -> bool {
    if seeds.upvar_indices.is_empty() {
        return false;
    }
    if place.local != Local::from_u32(1) {
        return false;
    }
    for elem in place.projection.iter() {
        if let PlaceElem::Field(field_idx, _) = elem {
            if seeds.upvar_indices.contains(&field_idx.as_usize()) {
                return true;
            }
        }
    }
    false
}

/// Finds every closure created in `body` via an aggregate, returning each
/// closure's `DefId` together with the caller locals captured as its upvars
/// (in upvar-field order; `None` for constant captures). Used by the
/// interprocedural pass to propagate sender taint from a captured value into
/// the corresponding upvar of the closure body.
pub fn find_closure_captures<'tcx>(body: &Body<'tcx>) -> Vec<(DefId, Vec<Option<Local>>)> {
    let mut out = Vec::new();
    for (_, bb_data) in body.basic_blocks.iter_enumerated() {
        for stmt in &bb_data.statements {
            if let StatementKind::Assign(assign) = &stmt.kind {
                let (_lhs, rvalue) = assign.as_ref();
                if let Rvalue::Aggregate(kind, operands) = rvalue {
                    if let AggregateKind::Closure(closure_def, _) = &**kind {
                        let caps = operands.iter().map(|op| operand_local(op)).collect();
                        out.push((*closure_def, caps));
                    }
                }
            }
        }
    }
    out
}

/// True if `ty` (after stripping references) is a type that can carry a sender
/// identity: `cosmwasm_std::Addr` / `CanonicalAddr`, `String`, or `str`. Used
/// to type-gate interprocedural seeds so that non-address parameters (e.g.
/// `Vec<Asset>`) are never mistaken for the sender — the main false-positive
/// source of naive argument/capture propagation.
pub fn ty_is_sender_like<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    let mut ty = ty;
    // Referenzen abtragen: &Addr, &&str, &String, ...
    while let TyKind::Ref(_, inner, _) = ty.kind() {
        ty = *inner;
    }
    match ty.kind() {
        TyKind::Str => true,
        TyKind::Adt(adt_def, _) => {
            let path = tcx.def_path_str(adt_def.did());
            path == "cosmwasm_std::Addr"
                || path == "cosmwasm_std::CanonicalAddr"
                || path == "String"
                || path.ends_with("::String")
        }
        _ => false,
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

pub fn find_auth_state_variables<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    comparisons: &[SenderComparison],
) -> Vec<AuthStateVariable> {
    let mut result = Vec::new();

    let block_map = build_block_local_source_map(tcx, body);

    for cmp in comparisons {
        let call_source = trace_to_load_call(
            tcx,
            cmp.compared_local,
            cmp.location.block,
            body,
            &block_map,
            24,
        );

        let (callee, arg_locals, load_location) = match call_source {
            Some(LocalSource::CallReturn {
                callee,
                arg_locals,
                location,
                ..
            }) => (callee, arg_locals, location),
            _ => continue,
        };

        // Only `cw_storage_plus` load operations yield an authorization state variable.
        if !callee.map_or(false, |d| is_storage_load_fn(tcx, d)) {
            continue;
        }

        let storage_item_local = match arg_locals.get(0).copied().flatten() {
            Some(l) => l,
            None => continue,
        };

        let (symbolic_name, storage_def_id) =
            find_storage_static_name(tcx, body, storage_item_local);

        result.push(AuthStateVariable {
            compared_local: cmp.compared_local,
            storage_item_local,
            symbolic_name,
            storage_def_id,
            load_location,
        });
    }

    result
}

#[derive(Debug, Clone)]
enum LocalSource {
    CopiedFrom(Local),
    CallReturn {
        func_debug: String,
        callee: Option<DefId>,
        arg_locals: Vec<Option<Local>>,
        location: Location,
    },
}
/// Builds a map of MIR locals to their value sources within each basic block.
fn build_block_local_source_map<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
) -> HashMap<(BasicBlock, Local), LocalSource> {
    let mut map = HashMap::new();

    for (bb_idx, bb_data) in body.basic_blocks.iter_enumerated() {
        for stmt in bb_data.statements.iter() {
            if let StatementKind::Assign(assign) = &stmt.kind {
                let (lhs, rhs) = assign.as_ref();
                let source = match rhs {
                    Rvalue::Use(Operand::Move(place), _) | Rvalue::Use(Operand::Copy(place), _) => {
                        Some(LocalSource::CopiedFrom(place.local))
                    }
                    Rvalue::Ref(_, _, place) => Some(LocalSource::CopiedFrom(place.local)),
                    Rvalue::RawPtr(_, place) => Some(LocalSource::CopiedFrom(place.local)),
                    Rvalue::Cast(_, op, _) => operand_local(op).map(LocalSource::CopiedFrom),
                    _ => None,
                };
                if let Some(src) = source {
                    map.insert((bb_idx, lhs.local), src);
                }
            }
        }

        if let TerminatorKind::Call {
            func,
            args,
            destination,
            ..
        } = &bb_data.terminator().kind
        {
            let location = Location {
                block: bb_idx,
                statement_index: bb_data.statements.len(),
            };

            map.insert(
                (bb_idx, destination.local),
                LocalSource::CallReturn {
                    func_debug: format!("{:?}", func),
                    callee: callee_def_id(tcx, body, func),
                    arg_locals: args.iter().map(|a| operand_local(&a.node)).collect(),
                    location,
                },
            );
        }
    }
    map
}

fn trace_to_load_call<'tcx>(
    tcx: TyCtxt<'tcx>,
    start_local: Local,
    start_block: BasicBlock,
    body: &Body<'tcx>,
    block_map: &HashMap<(BasicBlock, Local), LocalSource>,
    max_depth: usize,
) -> Option<LocalSource> {
    let mut predecessors: HashMap<BasicBlock, Vec<BasicBlock>> = HashMap::new();
    for (bb_idx, bb_data) in body.basic_blocks.iter_enumerated() {
        for succ in bb_data.terminator().successors() {
            predecessors.entry(succ).or_default().push(bb_idx);
        }
    }

    let mut queue: Vec<(BasicBlock, Local)> = vec![(start_block, start_local)];
    let mut visisted: HashSet<(BasicBlock, Local)> = HashSet::new();

    for _ in 0..max_depth {
        let (block, local) = match queue.pop() {
            Some(x) => x,
            None => return None,
        };

        if !visisted.insert((block, local)) {
            continue;
        }

        match block_map.get(&(block, local)) {
            Some(LocalSource::CallReturn {
                func_debug,
                callee,
                arg_locals,
                location,
            }) => {
                if callee.map_or(false, |d| is_forwarding_glue_fn(tcx, d)) {
                    // look through ?-/Deref-Glue -> follow first argument
                    if let Some(Some(next_local)) = arg_locals.get(0) {
                        queue.push((location.block, *next_local));
                        continue;
                    }
                    return None;
                }
                // real interesting Call
                return Some(LocalSource::CallReturn {
                    func_debug: func_debug.clone(),
                    callee: *callee,
                    arg_locals: arg_locals.clone(),
                    location: *location,
                });
            }
            Some(LocalSource::CopiedFrom(next_local)) => {
                // contiunue Search in this Block
                queue.push((block, *next_local));
            }
            None => {
                // not defined in this Block -> search all predecessors
                if let Some(preds) = predecessors.get(&block) {
                    for &pred in preds {
                        queue.push((pred, local));
                    }
                }
            }
        }
    }

    None
}

/// Resolves the storage constant a load receiver refers to.
///
/// Matches the receiver's type against the type of every local `const` item
/// via canonical type identity — both sides region-erased and compared as
/// interned `Ty`s. String comparison of formatted types is unreliable here:
/// MIR local types carry erased regions while `type_of` results carry
/// early-bound ones, and their `Debug`/`Display` renderings differ.
///
/// Returns the constant's name and `DefId` on success; otherwise falls back
/// to the normalized type string (display only, no `DefId`).
///
/// Inherent limit: two storage constants of the same type (e.g. two
/// `Item<Config>`) are indistinguishable by type — the first match wins.
fn find_storage_static_name<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    item_local: Local,
) -> (String, Option<DefId>) {
    let item_ty = body.local_decls[item_local].ty;

    // Peel all references — the load receiver is usually `&Item<T>`.
    let mut base_ty = item_ty;
    while let TyKind::Ref(_, inner, _) = base_ty.kind() {
        base_ty = *inner;
    }

    let base_ty_erased = tcx.erase_and_anonymize_regions(base_ty);

    for local_def_id in tcx.iter_local_def_id() {
        if !matches!(tcx.def_kind(local_def_id), DefKind::Const { .. }) {
            continue;
        }

        let def_id = local_def_id.to_def_id();
        let const_ty = tcx.type_of(def_id).skip_binder();

        if tcx.erase_and_anonymize_regions(const_ty) == base_ty_erased {
            return (tcx.item_name(def_id).to_string(), Some(def_id));
        }
    }

    (normalize_ty_str(&format!("{:?}", base_ty)), None)
}

/// Computes, for every function in the call graph, whether it performs
/// an authorization check on every path from its entry to a normal
/// return. Solved as a least fixpoint over the
/// call graph: a call to an already-checking callee counts as a check,
/// which may render further callers always-checking. The monotone
/// iteration from the pessimistic initial value `false` handles recursion.
pub fn compute_always_checks<'tcx>(
    tcx: TyCtxt<'tcx>,
    call_graph: &CallGraph,
    fn_comparisons: &HashMap<DefId, Vec<SenderComparison>>,
) -> HashMap<DefId, bool> {
    let mut summary: HashMap<DefId, bool> = call_graph.nodes.iter().map(|&n| (n, false)).collect();

    // `call_graph.call_sites` is keyed by CALLEE; gating needs the call
    // sites *inside* a function, i.e. grouped by caller — otherwise the
    // inserted locations would belong to a different body.
    let mut sites_by_caller: HashMap<DefId, Vec<&CallSite>> = HashMap::new();
    for cs in call_graph.call_sites.values().flatten() {
        sites_by_caller.entry(cs.caller).or_default().push(cs);
    }

    loop {
        let mut changed = false;

        for &f in &call_graph.nodes {
            if summary.get(&f).copied().unwrap_or(false)
                || !f.is_local()
                || !tcx.is_mir_available(f)
            {
                continue;
            }

            let body = tcx.optimized_mir(f);
            let comparisons = fn_comparisons.get(&f).cloned().unwrap_or_default();
            let sites = sites_by_caller.get(&f).map(|v| v.as_slice()).unwrap_or(&[]);

            let ok_points = ok_return_points(tcx, body);
            let ok_blocks: HashSet<BasicBlock> = ok_points.iter().map(|l| l.block).collect();

            let check_locations =
                gating_check_locations(tcx, body, &comparisons, sites, &summary, &ok_blocks);

            let (_in, always) =
                solve_auth_gating(tcx, body, check_locations, AuthState::Unchecked, &ok_points);

            if always == AuthState::Checked {
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

/// Builds the gating check locations: the *effective* sender-comparison
/// guards (branch-sensitive) plus call sites to always-checking callees.
fn gating_check_locations<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    comparisons: &[SenderComparison],
    call_sites: &[&CallSite],
    always_checks: &HashMap<DefId, bool>,
    ok_blocks: &HashSet<BasicBlock>,
) -> HashSet<Location> {
    let mut locations = effective_guard_locations(tcx, body, comparisons, ok_blocks);
    for cs in call_sites {
        if always_checks.get(&cs.callee).copied().unwrap_or(false) {
            locations.insert(cs.location);
        }
    }
    locations
}

/// A comparison only counts as an *enforcing* guard if its result feeds a
/// `SwitchInt` of which at least one branch aborts the Ok path entirely:
/// from that target no Ok-return assignment is reachable (early
/// `return Err(..)`, `?` on a Result-returning check, panic). A comparison
/// whose branches all rejoin the Ok path (e.g. fee logic keyed on the
/// sender) enforces nothing and must not count.
fn effective_guard_locations<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    comparisons: &[SenderComparison],
    ok_blocks: &HashSet<BasicBlock>,
) -> HashSet<Location> {
    let mut result_to_loc: HashMap<Local, Location> = HashMap::new();
    for cmp in comparisons {
        if let Some(res) = comparison_result_local(body, cmp) {
            result_to_loc.insert(res, cmp.location);
        }
    }
    if result_to_loc.is_empty() {
        return HashSet::new();
    }

    let mut alias: HashMap<Local, Local> = HashMap::new();
    for (_, data) in body.basic_blocks.iter_enumerated() {
        for stmt in data.statements.iter() {
            if let StatementKind::Assign(assign) = &stmt.kind {
                let (lhs, rvalue) = assign.as_ref();
                let src = match rvalue {
                    Rvalue::Use(op, _) | Rvalue::UnaryOp(_, op) => operand_local(op),
                    // `x?` / `match res {..}` switch on `discriminant(res)`;
                    // without this arm no Result-returning check (auth
                    // sinks!) ever resolves to its comparison.
                    Rvalue::Discriminant(place) => Some(place.local),
                    _ => None,
                };
                if let Some(s) = src {
                    alias.insert(lhs.local, s);
                }
            }
        }

        // `x?` routes the checked Result through a CALL to `Try::branch`
        // (`_cf = branch(move _r)`) before switching on
        // `discriminant(_cf)`. The statement scan above cannot see this —
        // alias through calls to identity-preserving glue as well, or no
        // non-local auth sink (`assert_admin(..)?`) ever resolves.
        if let TerminatorKind::Call {
            func,
            args,
            destination,
            ..
        } = &data.terminator().kind
        {
            let is_glue = utility::callee_def_id(tcx, body, func)
                .map_or(false, |d| is_forwarding_glue_fn(tcx, d));
            if is_glue {
                if let Some(arg) = args.get(0).and_then(|a| operand_local(&a.node)) {
                    alias.insert(destination.local, arg);
                }
            }
        }
    }
    let resolve = |start: Local| -> Vec<Local> {
        let mut chain = vec![start];
        let mut current = start;
        let mut seen = HashSet::new();
        seen.insert(start);
        while let Some(&next) = alias.get(&current) {
            if !seen.insert(next) {
                break;
            }
            chain.push(next);
            current = next;
        }
        chain
    };

    let mut effective = HashSet::new();
    for (_, data) in body.basic_blocks.iter_enumerated() {
        let TerminatorKind::SwitchInt { discr, targets } = &data.terminator().kind else {
            continue;
        };
        let Some(discr_local) = operand_local(discr) else {
            continue;
        };
        let matched = resolve(discr_local)
            .into_iter()
            .find_map(|l| result_to_loc.get(&l).copied());
        let Some(loc) = matched else {
            continue;
        };

        let all: Vec<BasicBlock> = targets.all_targets().to_vec();
        let has_aborting_arm = all.iter().any(|&t| {
            let reach = reachable_from(body, t);
            !ok_blocks.iter().any(|ob| reach.contains(ob))
        });
        if has_aborting_arm {
            effective.insert(loc);
        }
    }

    effective
}

fn comparison_result_local<'tcx>(body: &Body<'tcx>, cmp: &SenderComparison) -> Option<Local> {
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

/// Forward-reachable basic blocks from `start` (inclusive).
fn reachable_from<'tcx>(body: &Body<'tcx>, start: BasicBlock) -> HashSet<BasicBlock> {
    let mut seen = HashSet::new();
    let mut stack = vec![start];
    while let Some(bb) = stack.pop() {
        if !seen.insert(bb) {
            continue;
        }
        for succ in body.basic_blocks[bb].terminator().successors() {
            stack.push(succ);
        }
    }

    seen
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AuthState {
    Unchecked,
    Checked,
}

impl JoinSemiLattice for AuthState {
    fn join(&mut self, other: &Self) -> bool {
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

/// Forward must-analysis: propagates whether an authorization check has
/// been observed on every path from the entry to the current program
/// point. Built on `rustc_mir_dataflow`, which handles the worklist,
/// reverse-postorder iteration and — crucially — the correct treatment
/// of unwind/cleanup edges (these are not normal predecessors of a sink
/// and must not dilute the must-state).
struct AuthGateAnalysis {
    /// Exact locations of `info.sender` comparisons (Schritt 1) that are
    /// treated as authorization checks.
    check_locations: std::collections::HashSet<Location>,
    /// Auth state at the function entry. `Unchecked` for the standalone
    /// intraprocedural analysis; a propagated context once the gating
    /// becomes interprocedural (cf. docs/interprocedural-plan.md, §5.1).
    entry_checked: AuthState,
}

impl<'tcx> Analysis<'tcx> for AuthGateAnalysis {
    type Domain = AuthState;
    type Direction = Forward;

    const NAME: &'static str = "auth_gate";

    /// Bottom-Element (= neutrales Element des Join). Optimistische
    /// Initialisierung; Verbindungspunkte senken den Wert ggf. auf
    /// `Unchecked` (Available-Analysis-Schema).
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

fn solve_auth_gating<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    check_locations: HashSet<Location>,
    entry_checked: AuthState,
    ok_points: &[Location],
) -> (HashMap<BasicBlock, AuthState>, AuthState) {
    let analysis = AuthGateAnalysis {
        check_locations,
        entry_checked,
    };

    let results = analysis.iterate_to_fixpoint(tcx, body, None);
    let mut cursor = results.into_results_cursor(body);

    let mut in_state: HashMap<BasicBlock, AuthState> = HashMap::new();
    for (bb, _) in body.basic_blocks.iter_enumerated() {
        cursor.seek_before_primary_effect(Location {
            block: bb,
            statement_index: 0,
        });
        in_state.insert(bb, *cursor.get());
    }

    let mut always = AuthState::Checked;
    let mut counted = false;
    for loc in ok_points {
        cursor.seek_before_primary_effect(*loc);
        always.join(cursor.get());
        counted = true;
    }
    let always = if counted {
        always
    } else {
        AuthState::Unchecked
    };

    (in_state, always)
}

fn ok_return_points<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) -> Vec<Location> {
    let mut points = Vec::new();

    for (block, data) in body.basic_blocks.iter_enumerated(){
        for (statement_idx, stmt) in data.statements.iter().enumerate(){
            let StatementKind::Assign(assign) = &stmt.kind else {
                continue;
            };
            let (place, rvalue) = assign.as_ref();

            if place.local.as_usize() != 0 || !place.projection.is_empty(){
                continue;
            }

            if let Rvalue::Aggregate(kind, _ ) = rvalue {
                if let AggregateKind::Adt(def_id, variant_index, .. ) = kind.as_ref(){
                    // 'Result::Ok' is variant 0
                    if variant_index.as_usize() == 0 && is_result_def(tcx, *def_id){
                        points.push(Location { block, statement_index: statement_idx })
                    }
                }
            }
        }
    }


    points
}
