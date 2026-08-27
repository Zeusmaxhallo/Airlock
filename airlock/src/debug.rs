//! Opt-in diagnostic traces.
//!
//! These traces answer "why did this finding come out the way it did?" for a
//! single contract without touching the analysis itself: everything here is
//! purely observational and writes to stderr only when the corresponding
//! environment variable is set.
//!
//! * `AIRLOCK_DEBUG_TAINT` — per sink, whether its written value is tainted,
//!   and a backward trace of the definitions that led there, so the exact hop
//!   at which a taint chain starts or breaks becomes visible.
//! * `AIRLOCK_DEBUG_CLOSURE` — why a closure was or was not accepted as an
//!   always-checking guard, and which closure a write call received.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use rustc_middle::mir::{Body, Local, Location, StatementKind, TerminatorKind};
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::DefId;

use crate::auth_gate::AuthState;
use crate::evaluation::SensitiveSink;
use crate::mir_util::{operand_local, rvalue_locals};
use crate::taint::TaintResult;

/// How deep the backward trace of a sink value follows its definitions.
const MAX_TRACE_DEPTH: usize = 14;

fn enabled(var: &str, cache: &'static OnceLock<bool>) -> bool {
    *cache.get_or_init(|| std::env::var(var).is_ok())
}

fn taint_enabled() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    enabled("AIRLOCK_DEBUG_TAINT", &CACHE)
}

fn closure_enabled() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    enabled("AIRLOCK_DEBUG_CLOSURE", &CACHE)
}

/// A sink whose written value is tainted: names the value, the seed it came
/// from and its type, then traces the chain hop by hop.
pub fn tainted_sink<'tcx>(
    body: &Body<'tcx>,
    result: &TaintResult,
    sink: &SensitiveSink,
    value_local: Local,
    param_local: Local,
    param_ty: &str,
) {
    if !taint_enabled() {
        return;
    }
    eprintln!(
        "\t[dbg] TAINTED sink {} @ {:?}: value=_{}  origin=_{} : {}",
        sink.symbolic_name,
        sink.location,
        value_local.as_usize(),
        param_local.as_usize(),
        param_ty
    );
    trace_value(body, result, value_local);
}

/// A sink whose written value is not tainted although address-typed
/// parameters were seeded: traces the value back to the hop where the chain
/// breaks.
pub fn untainted_sink<'tcx>(
    body: &Body<'tcx>,
    result: &TaintResult,
    sink: &SensitiveSink,
    value_local: Local,
    seed: &HashSet<Local>,
) {
    if !taint_enabled() {
        return;
    }
    let seed_str: Vec<String> = seed.iter().map(|l| format!("_{}", l.as_usize())).collect();
    eprintln!(
        "\t[dbg] UNTAINTED sink {} @ {:?}: value=_{}  seed=[{}]",
        sink.symbolic_name,
        sink.location,
        value_local.as_usize(),
        seed_str.join(",")
    );
    trace_value(body, result, value_local);
}

/// Backward trace of the definitions of `value_local`, printing per hop
/// whether the local is tainted (`T`); a `*` marks a tainted read operand.
fn trace_value<'tcx>(body: &Body<'tcx>, result: &TaintResult, value_local: Local) {
    // Index every local to the rvalues and calls that define it.
    let mut defs: HashMap<Local, Vec<(String, Vec<Local>)>> = HashMap::new();
    for data in body.basic_blocks.iter() {
        for stmt in data.statements.iter() {
            if let StatementKind::Assign(assign) = &stmt.kind {
                let (lhs, rvalue) = assign.as_ref();
                defs.entry(lhs.local)
                    .or_default()
                    .push((format!("{:?}", rvalue), rvalue_locals(rvalue)));
            }
        }
        if let TerminatorKind::Call {
            func,
            args,
            destination,
            ..
        } = &data.terminator().kind
        {
            let reads: Vec<Local> = args.iter().filter_map(|a| operand_local(&a.node)).collect();
            defs.entry(destination.local)
                .or_default()
                .push((format!("call {:?}", func), reads));
        }
    }

    let mut seen = HashSet::new();
    let mut stack = vec![(value_local, 0usize)];
    while let Some((local, depth)) = stack.pop() {
        if depth > MAX_TRACE_DEPTH || !seen.insert(local) {
            continue;
        }
        let flag = if result.is_tainted(local) { 'T' } else { ' ' };
        let indent = "  ".repeat(depth);

        let Some(definitions) = defs.get(&local) else {
            eprintln!(
                "\t[dbg{}] {}_{} (parameter or undefined)",
                flag,
                indent,
                local.as_usize()
            );
            continue;
        };

        for (description, reads) in definitions {
            let reads_str: Vec<String> = reads
                .iter()
                .map(|r| {
                    format!(
                        "_{}{}",
                        r.as_usize(),
                        if result.is_tainted(*r) { "*" } else { "" }
                    )
                })
                .collect();
            eprintln!(
                "\t[dbg{}] {}_{} <- {}  reads=[{}]",
                flag,
                indent,
                local.as_usize(),
                description.chars().take(72).collect::<String>(),
                reads_str.join(",")
            );
            for &r in reads {
                stack.push((r, depth + 1));
            }
        }
    }
}

/// A closure that was not considered as a guard because it has no successful
/// return.
pub fn closure_skipped(tcx: TyCtxt<'_>, def_id: DefId, comparison_count: usize) {
    if !closure_enabled() {
        return;
    }
    eprintln!(
        "[closure-ac] SKIP {} — comps={} ok_points=0",
        tcx.def_path_str(def_id),
        comparison_count
    );
}

/// The gating outcome of a closure body.
pub fn closure_result(
    tcx: TyCtxt<'_>,
    def_id: DefId,
    comparison_count: usize,
    ok_point_count: usize,
    check_count: usize,
    always: AuthState,
) {
    if !closure_enabled() {
        return;
    }
    eprintln!(
        "[closure-ac] {} — comps={} ok_points={} checks={} always={:?}",
        tcx.def_path_str(def_id),
        comparison_count,
        ok_point_count,
        check_count,
        always
    );
}

/// The closure a write call received, and whether it is an accepted guard.
pub fn closure_sink(
    tcx: TyCtxt<'_>,
    location: Location,
    closure_def: DefId,
    checking_closures: &HashSet<DefId>,
) {
    if !closure_enabled() {
        return;
    }
    eprintln!(
        "[closure-ac] sink @ {:?} closure-arg {} in_ac={}",
        location,
        tcx.def_path_str(closure_def),
        checking_closures.contains(&closure_def)
    );
}
