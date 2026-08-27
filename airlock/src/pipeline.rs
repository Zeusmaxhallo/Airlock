//! The analysis pipeline.
//!
//! Runs the six stages in the order in which they depend on one another and
//! passes each stage's result on to the next. The stages themselves live in
//! their own modules; this module only wires them together and hands the
//! results to [`crate::report`].

use std::collections::HashMap;

use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::DefId;

use crate::auth_storage;
use crate::call_graph::CallGraph;
use crate::cosmwasm;
use crate::evaluation::{self, EvaluationContext};
use crate::gating::{self, GatingIndex};
use crate::predicates;
use crate::report;
use crate::sender_comparisons::{self, SenderComparison};
use crate::sender_taint;
use crate::storage_inventory::{ConstDefIndex, ConstTypeIndex, StorageInventory};
use crate::taint;

pub fn run(tcx: TyCtxt<'_>) {
    // ---- Stage 1: storage inventory -------------------------------------
    let mut inventory = StorageInventory::build(tcx);
    let const_types = ConstTypeIndex::build(tcx);

    // ---- Stage 2: entry point and call graph ----------------------------
    let Some(root) = cosmwasm::find_execute(tcx) else {
        report::storage_inventory(&inventory);
        report::no_entry_point();
        return;
    };
    report::entry_point(tcx, root);

    let call_graph = CallGraph::build_from_root(tcx, root);
    report::call_graph_size(root, call_graph.nodes().len());

    // ---- Stage 3: sender comparisons ------------------------------------
    // The seeding fixpoint first, so that every body is analysed with the
    // sender taint its callers pass in.
    let seeds = sender_taint::seed_sender_taint(tcx, &call_graph);

    let mut fn_comparisons: HashMap<DefId, Vec<SenderComparison>> = HashMap::new();
    for &f in &seeds.targets {
        let Some(body) = cosmwasm::body_of(tcx, f) else {
            continue;
        };
        report::function_header(tcx, f);

        let comparisons =
            sender_comparisons::find_sender_comparisons(tcx, body, &seeds.seeds_of(f));
        report::sender_comparisons(&comparisons);

        // ---- Stage 4: auth-storage marking ------------------------------
        let const_defs = ConstDefIndex::build(tcx, body);
        let auth_vars = auth_storage::find_auth_state_variables(
            tcx,
            body,
            &comparisons,
            &const_defs,
            &const_types,
        );
        report::auth_state_variables(&auth_vars);
        inventory.mark_auth(auth_vars.iter().filter_map(|v| v.storage_def_id));

        fn_comparisons.insert(f, comparisons);
    }

    report::storage_inventory(&inventory);

    // Sender checks that hide behind a predicate helper or an `Option`
    // combinator are added to the comparison map before stage 5 runs. They
    // have to be in place beforehand: a handler protected by a boolean helper
    // would otherwise count as gated itself without passing that protection on
    // to the helpers it calls — the shape of `only_authorized(..)?` followed by
    // a private writing helper.
    let sender_predicates = predicates::sender_predicate_summary(tcx, &fn_comparisons);
    let bool_predicates = predicates::bool_predicate_fns(tcx, &fn_comparisons);
    for &f in call_graph.nodes() {
        let Some(body) = cosmwasm::body_of(tcx, f) else {
            continue;
        };
        let mut synthetic = predicates::option_predicate_comparisons(tcx, body, &sender_predicates);
        synthetic.extend(predicates::predicate_call_comparisons(
            tcx,
            body,
            &bool_predicates,
        ));
        if !synthetic.is_empty() {
            fn_comparisons.entry(f).or_default().extend(synthetic);
        }
    }

    // ---- Stage 5: gating summaries --------------------------------------
    let gating_index = GatingIndex::build(tcx, &fn_comparisons);

    let always_checks = gating::compute_always_checks(tcx, &call_graph, &gating_index);
    let checking_closures =
        gating::closures_always_checking(tcx, &fn_comparisons, &gating_index, &always_checks);
    report::always_checking(tcx, &always_checks);

    let return_taint_params = taint::compute_return_taint_params(tcx, &call_graph, &const_types);
    report::return_taint(tcx, &return_taint_params);

    let entry_checked =
        gating::compute_entry_checked(tcx, &call_graph, &gating_index, &always_checks);
    report::entry_checked(tcx, &entry_checked);

    // ---- Stage 6: final evaluation --------------------------------------
    let auth_item_names = inventory.auth_item_names();
    let ctx = EvaluationContext {
        const_types: &const_types,
        gating: &gating_index,
        always_checks: &always_checks,
        return_taint_params: &return_taint_params,
        entry_checked: &entry_checked,
        checking_closures: &checking_closures,
    };

    let mut all_findings = Vec::new();
    for &f in call_graph.nodes() {
        let Some(body) = cosmwasm::body_of(tcx, f) else {
            continue;
        };
        let findings = evaluation::evaluate_function(
            tcx,
            body,
            f,
            call_graph.call_sites_in(f),
            &auth_item_names,
            &ctx,
        );
        if findings.is_empty() {
            continue;
        }
        let function_name = tcx.def_path_str(f);
        for finding in findings {
            all_findings.push((function_name.clone(), finding));
        }
    }

    report::findings(&mut all_findings);
}
