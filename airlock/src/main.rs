#![feature(rustc_private)]

mod analysis;
mod call_graph;
mod storage_inventory;
mod utility;

extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_mir_dataflow;
extern crate rustc_session;
extern crate rustc_span;

use rustc_driver::HandledOptions;
use rustc_hir::def_id::DefId;
use rustc_interface::Config;
use rustc_middle::mir::Local;
use rustc_middle::ty::TyCtxt;
use rustc_session::EarlyDiagCtxt;
use rustc_session::config::{self, ErrorOutputType, Input};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use crate::analysis::SenderComparison;
use crate::storage_inventory::StorageInventory;

/// rustc 1.98.0-nightly (c397dae80 2026-07-02)
const TOOLCHAIN: &str = "nightly-2026-07-04";

fn main() {
    rustc_driver::install_ice_hook("", |_| ());

    let mut args: Vec<String> = std::env::args().collect();

    let wrapper_mode = args
        .get(1)
        .map(|a| a.ends_with("rustc") || a.ends_with("rustc.exe"))
        .unwrap_or(false);

    if wrapper_mode {
        args.remove(1);

        let is_probe =
            args.iter().any(|a| a == "-") || args.iter().any(|a| a.starts_with("--print="));

        let is_dependency = args.iter().any(|a| a == "--cap-lints")
            && args
                .windows(2)
                .any(|w| w[0] == "--cap-lints" && w[1] == "allow");

        let is_relevant = !is_probe
            && !is_dependency
            && args.iter().any(|a| a == "--crate-type")
            && !args.iter().any(|a| a.contains("build_script"));

        if !is_relevant {
            run_real_rustc(&args[1..]);
        }

        let crate_name = args
            .windows(2)
            .find(|w| w[0] == "--crate-name")
            .map(|w| w[1].as_str())
            .unwrap_or("<unbekannt>");
        eprintln!("══ Analyze Contract-Crate: {crate_name} ══");
        run_analysis(&args);
        run_real_rustc(&args[1..]);
    } else {
        if args.len() < 2 {
            eprintln!("Usage: cosmwasm-access-checker <file.rs>");
            std::process::exit(1);
        }
        run_analysis(&args);
    }
}

fn run_real_rustc(args: &[String]) {
    let status = std::process::Command::new("rustup")
        .arg("run")
        .arg(TOOLCHAIN)
        .arg("rustc")
        .args(args)
        .status()
        .expect("Could not start rustup");
    std::process::exit(status.code().unwrap_or(1));
}

fn run_analysis(args: &Vec<String>) {
    let filepath = args
        .iter()
        .find(|a| a.ends_with(".rs"))
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            eprintln!("No .rs file found in arguments");
            std::process::exit(1);
        });

    let mut early_dcx = EarlyDiagCtxt::new(ErrorOutputType::default());

    let matches = match rustc_driver::handle_options(&early_dcx, args) {
        HandledOptions::Normal(m) => m,
        _ => std::process::exit(0),
    };

    let opts = config::build_session_options(&mut early_dcx, &matches);

    let config = Config {
        opts,
        crate_cfg: matches.opt_strs("cfg"),
        crate_check_cfg: matches.opt_strs("check-cfg"),
        input: Input::File(filepath),
        output_dir: None,
        output_file: None,
        ice_file: None,
        file_loader: None,
        lint_caps: Default::default(),
        psess_created: None,
        track_state: None,
        register_lints: None,
        override_queries: None,
        extra_symbols: Vec::new(),
        make_codegen_backend: None,
        using_internal_features: &rustc_driver::USING_INTERNAL_FEATURES,
    };

    rustc_interface::run_compiler(config, |compiler| {
        let krate = rustc_interface::parse(&compiler.sess);
        rustc_interface::create_and_enter_global_ctxt(compiler, krate, |tcx| {
            let storage_inventory = StorageInventory::build(tcx);

            let Some(root) = utility::find_execute(tcx) else {
                storage_inventory.print_inventory();
                eprintln!("No execute-Entry-Point, skipping analysis");
                return;
            };
            let call_graph = call_graph::CallGraph::build_from_root(tcx, root);
            // `find_auth_states` marks auth-state storage items (stage 2) and
            // returns the enriched inventory so later stages can read it.
            let (fn_comparisons, storage_inventory) =
                find_auth_states(tcx, &call_graph, storage_inventory);

            let always_checks = analysis::compute_always_checks(tcx, &call_graph, &fn_comparisons);

            // Sorted for deterministic output despite HashMap iteration order.
            let mut checking: Vec<String> = always_checks
                .iter()
                .filter_map(|(def_id, &ok)| ok.then(|| tcx.def_path_str(*def_id)))
                .collect();
            checking.sort();

            eprintln!(
                "\n[3] Always-checking functions: {}/{}",
                checking.len(),
                always_checks.len()
            );
            for name in &checking {
                eprintln!("\t{}", name);
            }

            let return_taint_params = analysis::compute_return_taint_params(tcx, &call_graph);

            // Non-empty entries only, sorted — deterministic output
            // despite HashMap iteration order.
            let mut flows: Vec<(String, Vec<usize>)> = return_taint_params
                .iter()
                .filter(|(_, params)| !params.is_empty())
                .map(|(def_id, params)| {
                    let mut p: Vec<usize> = params.iter().copied().collect();
                    p.sort();
                    (tcx.def_path_str(*def_id), p)
                })
                .collect();
            flows.sort();

            eprintln!(
                "\n[4] Return-tainted-by-param functions: {}/{}",
                flows.len(),
                return_taint_params.len()
            );
            for (name, params) in &flows {
                let list = params
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                eprintln!("\t{} <- param [{}]", name, list);
            }

            let entry_checked = analysis::compute_entry_checked(
                tcx,
                &call_graph,
                root,
                &fn_comparisons,
                &always_checks,
            );

            // Local functions only — non-local nodes (String::clone,
            // addr_validate, …) are vacuously entry-checked whenever all
            // their call sites happen to be gated and would drown the list.
            // Sorted for deterministic output despite HashMap iteration order.
            let mut gated: Vec<String> = entry_checked
                .iter()
                .filter_map(|(def_id, &ok)| {
                    (ok && def_id.is_local()).then(|| tcx.def_path_str(*def_id))
                })
                .collect();
            gated.sort();

            eprintln!(
                "\n[5] Entry-checked functions (local): {}/{}",
                gated.len(),
                entry_checked.len()
            );
            for name in &gated {
                eprintln!("\t{}", name);
            }


            // `call_graph.call_sites` is keyed by CALLEE; analyze_access_control
            // needs the sites *inside* each function, i.e. grouped by caller.
            let mut sites_by_caller: HashMap<DefId, Vec<&call_graph::CallSite>> = HashMap::new();
            for cs in call_graph.call_sites.values().flatten() {
                sites_by_caller.entry(cs.caller).or_default().push(cs);
            }

            let mut all_findings: Vec<(String, analysis::AccessControlFinding)> = Vec::new();
            for node in call_graph.nodes.iter() {
                if !node.is_local() || !tcx.is_mir_available(*node) {
                    continue;
                }
                let body = tcx.optimized_mir(*node);
                let comparisons = fn_comparisons.get(node).cloned().unwrap_or_default();
                let sites = sites_by_caller.get(node).map(|v| v.as_slice()).unwrap_or(&[]);
                let findings = analysis::analyze_access_control(
                    tcx,
                    body,
                    &storage_inventory,
                    &comparisons,
                    sites,
                    &always_checks,
                    &return_taint_params,
                    entry_checked.get(node).copied().unwrap_or(false),
                );
                let fname = tcx.def_path_str(*node);
                for f in findings {
                    all_findings.push((fname.clone(), f));
                }
            }

            // Sort for deterministic output: by function, then sink block.
            all_findings.sort_by(|a, b| {
                a.0.cmp(&b.0).then(
                    a.1.sink
                        .location
                        .block
                        .as_usize()
                        .cmp(&b.1.sink.location.block.as_usize()),
                )
            });

            let vuln_count = all_findings.iter().filter(|(_, f)| f.is_vulnerability()).count();

            eprintln!(
                "\n=== Access-Control findings: {} sink(s), {} vulnerability(ies)",
                all_findings.len(),
                vuln_count
            );
            for (fname, f) in &all_findings {
                // Classify: VULN = attacker-tainted write with no preceding
                // auth check; otherwise note why it is benign.
                let status = if f.is_vulnerability() {
                    "VULN "
                } else if f.gated {
                    "gated"
                } else {
                    "clean"
                };
                let taint = match &f.taint {
                    Some(src) => format!(" <- tainted by _{} : {}", src.param_local.as_usize(), src.param_ty),
                    None => String::new(),
                };
                eprintln!(
                    "\t[{}] {} writes {} @ {:?}{}",
                    status, fname, f.sink.symbolic_name, f.sink.location, taint
                );
            }
        });
    });
}

/// Interprocedural sender-taint analysis over the call graph.
///
/// Intraprocedural taint alone misses access-control checks where the sender
/// arrives indirectly: as a *function argument* (`is_admin(&info.sender)`) or
/// as a *closure upvar* (`admins.iter().any(|a| a == info.sender)`). This pass
/// computes a fixed point: taint each function with its current seeds, then
/// propagate sender-tainted actuals into the callee's formal parameters and
/// sender-tainted captures into the closure's upvars, until the seeds stop
/// growing. The final detection pass then runs with stable seeds.
fn find_auth_states(
    tcx: TyCtxt,
    call_graph: &call_graph::CallGraph,
    mut inventory: StorageInventory,
) -> (HashMap<DefId, Vec<SenderComparison>>, StorageInventory) {
    // 1. Analysis set: call-graph nodes plus every (possibly nested) closure
    //    created via an aggregate. Closures are not call edges but still need
    //    to be tainted and analysed. `closure_caps` records, per closure
    //    aggregate, the enclosing function and the captured caller locals.
    let mut nodes: HashSet<DefId> = call_graph.nodes.clone();
    let mut closure_caps: Vec<(DefId, DefId, Vec<Option<Local>>)> = Vec::new();
    let mut worklist: Vec<DefId> = call_graph.nodes.iter().copied().collect();
    let mut scanned: HashSet<DefId> = HashSet::new();
    while let Some(n) = worklist.pop() {
        if !scanned.insert(n) {
            continue;
        }
        if !n.is_local() || !tcx.is_mir_available(n) {
            continue;
        }
        let body = tcx.optimized_mir(n);
        for (closure_def, caps) in analysis::find_closure_captures(body) {
            closure_caps.push((n, closure_def, caps));
            if closure_def.is_local() && nodes.insert(closure_def) {
                worklist.push(closure_def);
            }
        }
    }

    // 2. Fixed point over seeds. Seeds grow monotonically; the iteration cap is
    //    a safety net against pathological cases.
    let mut seeds: HashMap<DefId, analysis::SenderSeeds> = HashMap::new();
    let mut taint: HashMap<DefId, HashSet<Local>> = HashMap::new();

    for _ in 0..16 {
        for &n in &nodes {
            if !n.is_local() || !tcx.is_mir_available(n) {
                continue;
            }
            let body = tcx.optimized_mir(n);
            let s = seeds.get(&n).cloned().unwrap_or_default();
            taint.insert(n, analysis::compute_sender_locals(tcx, body, &s));
        }

        let before = seeds.clone();

        // 2a. Argument wiring: a sender-tainted actual at position `i` taints
        //     the callee's formal parameter `_{i+1}` — but only if the passed
        //     value has a sender-compatible type. We gate on the *caller's*
        //     argument type, not the callee parameter: the callee may be
        //     generic (`addr: impl AsRef<str>`), whose formal type is only a
        //     type parameter, while the actual passed at the call site is
        //     concrete (`Addr`/`&Addr`/`&str`). This keeps the taint out of
        //     non-address parameters (e.g. a `Vec<Asset>`, the main FP source)
        //     while still seeding generic helpers like `is_admin`.
        for sites in call_graph.call_sites.values() {
            for cs in sites {
                let Some(caller_taint) = taint.get(&cs.caller) else {
                    continue;
                };
                let caller_body = if cs.caller.is_local() && tcx.is_mir_available(cs.caller) {
                    Some(tcx.optimized_mir(cs.caller))
                } else {
                    None
                };
                for (i, arg) in cs.arg_locals.iter().enumerate() {
                    if let Some(a) = arg {
                        if caller_taint.contains(a) {
                            let arg_sender_like = caller_body
                                .and_then(|b| b.local_decls.get(*a))
                                .map(|d| analysis::ty_is_sender_like(tcx, d.ty))
                                .unwrap_or(false);
                            if arg_sender_like {
                                seeds
                                    .entry(cs.callee)
                                    .or_default()
                                    .param_locals
                                    .insert(Local::from_usize(i + 1));
                            }
                        }
                    }
                }
            }
        }

        // 2b. Capture wiring: a sender-tainted capture `k` taints upvar index
        //     `k` of the closure — again only if the captured value has a
        //     sender-compatible type.
        for (caller, closure_def, caps) in &closure_caps {
            let Some(caller_taint) = taint.get(caller) else {
                continue;
            };
            let caller_body = if caller.is_local() && tcx.is_mir_available(*caller) {
                Some(tcx.optimized_mir(*caller))
            } else {
                None
            };
            for (k, cap) in caps.iter().enumerate() {
                if let Some(c) = cap {
                    if caller_taint.contains(c) {
                        let cap_sender_like = caller_body
                            .and_then(|b| b.local_decls.get(*c))
                            .map(|d| analysis::ty_is_sender_like(tcx, d.ty))
                            .unwrap_or(false);
                        if cap_sender_like {
                            seeds
                                .entry(*closure_def)
                                .or_default()
                                .upvar_indices
                                .insert(k);
                        }
                    }
                }
            }
        }

        if seeds == before {
            break;
        }
    }

    let mut fn_comparisons: HashMap<DefId, Vec<SenderComparison>> = HashMap::new();

    // 3. Final detection pass with stable seeds (this is what logs findings).
    for &n in &nodes {
        if !n.is_local() || !tcx.is_mir_available(n) {
            continue;
        }
        let body = tcx.optimized_mir(n);
        let s = seeds.get(&n).cloned().unwrap_or_default();
        let comparisons =
            analysis::analyze_function(tcx, body, &tcx.def_path_str(n), &s, &mut inventory);
        fn_comparisons.insert(n, comparisons);
    }

    inventory.print_inventory();
    (fn_comparisons, inventory)
}
