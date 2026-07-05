#![feature(rustc_private)]

mod storage_inventory;
mod utility;
mod call_graph;

extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;

use rustc_driver::HandledOptions;
use rustc_interface::Config;
use rustc_session::EarlyDiagCtxt;
use rustc_session::config::{self, ErrorOutputType, Input};
use std::path::PathBuf;

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
            let storage_inventory = storage_inventory::StorageInventory::build(tcx);
            storage_inventory.print_inventory();
        
            let Some(root) = utility::find_execute(tcx) else {
                eprintln!("No execute-Entry-Point, skipping analysis");
                return;
            };

            let call_graph = call_graph::CallGraph::build_from_root(tcx, root);
        });
    });
}
