#![feature(rustc_private)]

//! Airlock — a static analysis tool for detecting exploitable access control
//! vulnerabilities in CosmWasm smart contracts.
//!
//! The tool runs as a `RUSTC_WRAPPER` (or standalone on a single source file)
//! and inspects the compiler's own intermediate representations. The analysis
//! is a pipeline of six stages; each stage lives in its own module and is wired
//! together by [`pipeline::run`]:
//!
//! 1. **Storage inventory** — [`storage_inventory`]: every `cw-storage-plus`
//!    constant (`Item`/`Map`) defined in the crate, collected from the HIR.
//! 2. **Entry point and call graph** — [`call_graph`]: the `execute` entry
//!    point and the MIR-level call graph reachable from it, including the
//!    actual-to-formal argument mapping of every call site.
//! 3. **Sender comparisons** — [`sender_taint`], [`sender_comparisons`] and
//!    [`predicates`]: every program point at which `info.sender`, or a value
//!    derived from it, is compared against another value, including
//!    comparisons inside helper functions and closures reached through an
//!    interprocedural seeding fixpoint.
//! 4. **Auth-storage marking** — [`auth_storage`]: the counterpart of each
//!    comparison is traced back to a storage load, and the storage items it
//!    reads are marked authorization-relevant.
//! 5. **Gating summaries** — [`gating`], [`auth_gate`] and [`taint`]:
//!    interprocedural fixpoints over the call graph determining which
//!    functions enforce a sender check on every path, which are called
//!    exclusively from an already checked context, and which propagate
//!    parameter taint to their return value.
//! 6. **Final evaluation** — [`evaluation`]: for every write to an
//!    authorization-relevant item, whether the written value is
//!    attacker-controlled and whether the access is gated on all paths.
//!
//! Supporting modules: [`cosmwasm`] recognises the framework's types and
//! functions, [`mir_util`] holds generic MIR helpers, [`report`] produces all
//! console output and [`debug`] the opt-in diagnostic traces.

extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_mir_dataflow;
extern crate rustc_session;
extern crate rustc_span;

mod auth_gate;
mod auth_storage;
mod call_graph;
mod cosmwasm;
mod debug;
mod driver;
mod evaluation;
mod gating;
mod mir_util;
mod pipeline;
mod predicates;
mod report;
mod sender_comparisons;
mod sender_taint;
mod storage_inventory;
mod taint;

fn main() {
    driver::run();
}
