//! Console output.
//!
//! Every line the tool writes to stderr is produced here, so that the analysis
//! modules stay free of formatting and the report format is defined in one
//! place. The output follows the pipeline: the storage inventory and the
//! per-function stage 1 to 4 results first, then the interprocedural summaries
//! `[3]` to `[5]`, then the findings.
//!
//! Lists are sorted before they are printed. The analysis stages iterate over
//! hash maps in places, so sorting is what makes two runs over the same crate
//! produce byte-identical output.

use std::collections::{HashMap, HashSet};

use rustc_middle::mir::BasicBlock;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::DefId;

use crate::auth_storage::AuthStateVariable;
use crate::evaluation::AccessControlFinding;
use crate::sender_comparisons::SenderComparison;
use crate::storage_inventory::StorageInventory;

/// Header printed once per analysed contract crate.
pub fn crate_header(crate_name: &str) {
    eprintln!("══ Analyze Contract-Crate: {crate_name} ══");
}

/// Stage 2: the entry point the call graph is built from.
pub fn entry_point(tcx: TyCtxt<'_>, root: DefId) {
    eprintln!("Found Execute Entry Point: {:?}", tcx.def_path_str(root));
}

pub fn no_entry_point() {
    eprintln!("No execute-Entry-Point, skipping analysis");
}

/// Stage 2: a call whose target cannot be resolved statically.
pub fn skipped_function_pointer(block: BasicBlock, caller: DefId) {
    eprintln!(
        "[call_graph] Skipping function pointer call at {:?} in {:?}",
        block, caller
    );
}

/// Stage 2: size of the reachable call graph.
pub fn call_graph_size(root: DefId, node_count: usize) {
    eprintln!(
        "[call_graph] Reachable functions from {:?}: {}",
        root, node_count
    );
}

/// Header of the per-function stage 3 and 4 output.
pub fn function_header(tcx: TyCtxt<'_>, def_id: DefId) {
    let path = tcx.def_path_str(def_id);
    let name = path.split("::").last().unwrap_or(&path);
    eprintln!("\nchecking function: '{}'", name);
}

/// Stage 3 result for one function.
pub fn sender_comparisons(comparisons: &[SenderComparison]) {
    if comparisons.is_empty() {
        eprintln!("[1] info.sender comparisons: none");
        return;
    }
    eprintln!("[1] info.sender comparisons: {}", comparisons.len());
    for cmp in comparisons {
        eprintln!("\t{:?} {}", cmp.location, cmp.description);
    }
}

/// Stage 4 result for one function.
pub fn auth_state_variables(auth_vars: &[AuthStateVariable]) {
    if auth_vars.is_empty() {
        eprintln!("[2] Auth-State-Variables: none");
        return;
    }
    eprintln!("[2] Auth-State-Variables: {}", auth_vars.len());
    for auth_var in auth_vars {
        eprintln!(
            "\t{} (load @ {:?})",
            auth_var.symbolic_name, auth_var.load_location
        );
    }
}

/// Stage 1 result, printed once stage 4 has marked the authorization-relevant
/// items.
pub fn storage_inventory(inventory: &StorageInventory) {
    eprintln!("══ Storage Inventory ══");
    for item in &inventory.items {
        let auth = if item.is_auth { "  [auth]" } else { "" };
        eprintln!(
            "Name: {}, DefId: {:?}, Type: {}, Kind: {:?},  {}",
            item.name, item.def_id, item.ty_string, item.kind, auth
        );
    }
}

/// Stage 5: functions that check the sender on every path.
pub fn always_checking(tcx: TyCtxt<'_>, always_checks: &HashMap<DefId, bool>) {
    let mut names: Vec<String> = always_checks
        .iter()
        .filter_map(|(&def_id, &ok)| ok.then(|| tcx.def_path_str(def_id)))
        .collect();
    names.sort();

    eprintln!(
        "\n[3] Always-checking functions: {}/{}",
        names.len(),
        always_checks.len()
    );
    for name in &names {
        eprintln!("\t{}", name);
    }
}

/// Stage 5: functions that forward parameter taint to their return value.
pub fn return_taint(tcx: TyCtxt<'_>, return_taint_params: &HashMap<DefId, HashSet<usize>>) {
    let mut flows: Vec<(String, Vec<usize>)> = return_taint_params
        .iter()
        .filter(|(_, params)| !params.is_empty())
        .map(|(&def_id, params)| {
            let mut positions: Vec<usize> = params.iter().copied().collect();
            positions.sort();
            (tcx.def_path_str(def_id), positions)
        })
        .collect();
    flows.sort();

    eprintln!(
        "\n[4] Return-tainted-by-param functions: {}/{}",
        flows.len(),
        return_taint_params.len()
    );
    for (name, positions) in &flows {
        let list = positions
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        eprintln!("\t{} <- param [{}]", name, list);
    }
}

/// Stage 5: functions entered only from an already checked context.
///
/// Local functions only. A node from a dependency crate — `String::clone`,
/// `addr_validate`, … — is vacuously entry-checked whenever all its call sites
/// happen to be gated, and would drown the list.
pub fn entry_checked(tcx: TyCtxt<'_>, entry_checked: &HashMap<DefId, bool>) {
    let mut names: Vec<String> = entry_checked
        .iter()
        .filter_map(|(&def_id, &ok)| (ok && def_id.is_local()).then(|| tcx.def_path_str(def_id)))
        .collect();
    names.sort();

    eprintln!(
        "\n[5] Entry-checked functions (local): {}/{}",
        names.len(),
        entry_checked.len()
    );
    for name in &names {
        eprintln!("\t{}", name);
    }
}

/// Stage 6: the findings, sorted by function and sink position.
pub fn findings(all: &mut Vec<(String, AccessControlFinding)>) {
    all.sort_by(|a, b| {
        a.0.cmp(&b.0).then(
            a.1.sink
                .location
                .block
                .as_usize()
                .cmp(&b.1.sink.location.block.as_usize()),
        )
    });

    let vulnerabilities = all.iter().filter(|(_, f)| f.is_vulnerability()).count();

    eprintln!(
        "\n=== Access-Control findings: {} sink(s), {} vulnerability(ies)",
        all.len(),
        vulnerabilities
    );
    for (function, finding) in all.iter() {
        let taint = match &finding.taint {
            Some(source) => format!(
                " <- tainted by _{} : {}",
                source.param_local.as_usize(),
                source.param_ty
            ),
            None => String::new(),
        };
        eprintln!(
            "\t[{}] {} writes {} @ {:?}{}",
            finding.verdict(),
            function,
            finding.sink.symbolic_name,
            finding.sink.location,
            taint
        );
    }
}
