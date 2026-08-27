//! Process entry point: `RUSTC_WRAPPER` handling and compiler setup.
//!
//! Airlock needs the type-checked HIR and the MIR of a contract crate, so it
//! runs inside the compiler rather than beside it. Cargo invokes the binary in
//! place of `rustc` (`RUSTC_WRAPPER`); the wrapper filters out the invocations
//! that carry no contract code, runs the analysis on the remaining ones and
//! then delegates the real compilation to the pinned toolchain so the build
//! proceeds normally. Passing a single `.rs` file on the command line runs the
//! analysis standalone.

use std::path::PathBuf;

use rustc_driver::HandledOptions;
use rustc_interface::Config;
use rustc_session::EarlyDiagCtxt;
use rustc_session::config::{self, ErrorOutputType, Input};

use crate::pipeline;
use crate::report;

/// Toolchain the wrapper delegates the actual compilation to.
///
/// Must match `rust-toolchain.toml` and the toolchain used by the corpus
/// runner: rustc 1.98.0-nightly (c397dae80 2026-07-02).
const TOOLCHAIN: &str = "nightly-2026-07-04";

pub fn run() {
    rustc_driver::install_ice_hook("", |_| ());

    let mut args: Vec<String> = std::env::args().collect();

    // Cargo passes the path of the compiler to shadow as the first argument;
    // its absence means the binary was started directly on a source file.
    let wrapper_mode = args
        .get(1)
        .map(|a| a.ends_with("rustc") || a.ends_with("rustc.exe"))
        .unwrap_or(false);

    if !wrapper_mode {
        if args.len() < 2 {
            eprintln!("Usage: airlock <file.rs>");
            std::process::exit(1);
        }
        analyze(&args);
        return;
    }

    args.remove(1);

    if !carries_contract_code(&args) {
        // Never returns: the real compiler's exit code becomes ours.
        run_real_rustc(&args[1..]);
    }

    let crate_name = argument_value(&args, "--crate-name").unwrap_or("<unknown>");
    report::crate_header(crate_name);
    analyze(&args);
    run_real_rustc(&args[1..]);
}

/// Decides whether a wrapped `rustc` invocation is worth analysing.
///
/// Cargo issues many calls that contain no contract code: capability probes
/// (`-`, `--print=…`), dependency builds (compiled with `--cap-lints allow`)
/// and build scripts. Analysing them would cost time and pollute the report,
/// so only invocations that build a crate of the workspace itself qualify.
fn carries_contract_code(args: &[String]) -> bool {
    let is_probe = args.iter().any(|a| a == "-" || a.starts_with("--print="));
    let is_dependency = args
        .windows(2)
        .any(|w| w[0] == "--cap-lints" && w[1] == "allow");
    let is_build_script = args.iter().any(|a| a.contains("build_script"));

    !is_probe && !is_dependency && !is_build_script && args.iter().any(|a| a == "--crate-type")
}

/// Returns the value that follows `flag` in the argument list.
fn argument_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].as_str())
}

/// Delegates to the pinned toolchain and exits with its status code.
fn run_real_rustc(args: &[String]) -> ! {
    let status = std::process::Command::new("rustup")
        .arg("run")
        .arg(TOOLCHAIN)
        .arg("rustc")
        .args(args)
        .status()
        .expect("Could not start rustup");
    std::process::exit(status.code().unwrap_or(1));
}

/// Runs the compiler up to the point where HIR and MIR are available and hands
/// the type context to the analysis pipeline. The compilation itself is not
/// carried further — code generation is left to [`run_real_rustc`].
fn analyze(args: &[String]) {
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
        // `--help`, `--version` and friends were already answered.
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
            pipeline::run(tcx);
        });
    });
}
