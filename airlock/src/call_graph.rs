//! Stage 2 — entry point and call graph.
//!
//! Starting from the `execute` entry point, the reachable part of the crate is
//! explored at MIR level. Every `Call` terminator becomes a [`CallSite`] that
//! records not only the callee but also the actual-to-formal argument mapping
//! the interprocedural stages need: the actual argument at position `i`
//! corresponds to the callee's formal parameter `_{i+1}`.
//!
//! The call sites are stored grouped by the function that contains them, which
//! is the form all later stages read them in.

use std::collections::{HashMap, HashSet, VecDeque};

use rustc_middle::mir::{Body, Local, Location, TerminatorKind};
use rustc_middle::ty::{self, Instance, InstanceKind, TyCtxt, TyKind};
use rustc_span::def_id::DefId;

use crate::cosmwasm;
use crate::mir_util::operand_local;

/// A resolved call at a concrete program point.
#[derive(Debug, Clone)]
pub struct CallSite {
    /// Location of the `Call` terminator in the caller.
    pub location: Location,
    /// Function containing this call.
    pub caller: DefId,
    /// Resolved callee.
    pub callee: DefId,
    /// Actual arguments as caller locals; `None` for constant operands.
    pub arg_locals: Vec<Option<Local>>,
}

/// The call graph reachable from the entry point.
#[derive(Debug)]
pub struct CallGraph {
    /// Entry point the graph was built from.
    root: DefId,
    /// Reachable functions in discovery order, so that later stages iterate
    /// deterministically.
    nodes: Vec<DefId>,
    /// Call sites grouped by the function that contains them.
    sites_by_caller: HashMap<DefId, Vec<CallSite>>,
}

impl CallGraph {
    /// Breadth-first exploration from `root`. Only bodies the compiler can
    /// hand us are descended into; a callee from a dependency crate still
    /// becomes a node, so gating summaries can refer to it, but it has no
    /// outgoing edges.
    pub fn build_from_root(tcx: TyCtxt<'_>, root: DefId) -> Self {
        let mut nodes = Vec::new();
        let mut known: HashSet<DefId> = HashSet::new();
        let mut sites_by_caller: HashMap<DefId, Vec<CallSite>> = HashMap::new();

        let mut queue = VecDeque::new();
        // The root is a node even if none of its calls resolve; otherwise a
        // crate whose `execute` has no resolvable callee would be skipped by
        // every later stage.
        known.insert(root);
        nodes.push(root);
        queue.push_back(root);

        while let Some(caller) = queue.pop_front() {
            let Some(body) = cosmwasm::body_of(tcx, caller) else {
                continue;
            };

            for call_site in collect_call_sites(tcx, body) {
                let callee = call_site.callee;
                if known.insert(callee) {
                    nodes.push(callee);
                    queue.push_back(callee);
                }
                sites_by_caller.entry(caller).or_default().push(call_site);
            }
        }

        CallGraph {
            root,
            nodes,
            sites_by_caller,
        }
    }

    pub fn root(&self) -> DefId {
        self.root
    }

    /// All reachable functions, in discovery order.
    pub fn nodes(&self) -> &[DefId] {
        &self.nodes
    }

    /// The calls made inside `caller`.
    pub fn call_sites_in(&self, caller: DefId) -> &[CallSite] {
        self.sites_by_caller
            .get(&caller)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Every call site of the graph, regardless of caller.
    pub fn all_call_sites(&self) -> impl Iterator<Item = &CallSite> {
        self.sites_by_caller.values().flatten()
    }
}

/// Collects the resolved calls of a single body.
fn collect_call_sites<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) -> Vec<CallSite> {
    let mut call_sites = Vec::new();
    let caller = body.source.def_id();

    // Post-analysis rather than a fully monomorphized environment: the
    // exploration also descends into generic functions whose MIR still
    // contains unsubstituted parameters. In a codegen typing mode a failed
    // normalization of those is an internal compiler error.
    let typing_env = ty::TypingEnv::post_analysis(tcx, caller);

    for (block, block_data) in body.basic_blocks.iter_enumerated() {
        let terminator = block_data.terminator();
        let TerminatorKind::Call { func, args, .. } = &terminator.kind else {
            continue;
        };

        match func.ty(&body.local_decls, tcx).kind() {
            TyKind::FnDef(def_id, generic_args) => {
                // Trait methods are resolved to the implementation where
                // possible; the declaration is the fallback.
                let callee = Instance::try_resolve(tcx, typing_env, *def_id, generic_args)
                    .ok()
                    .flatten()
                    .map(|instance| match instance.def {
                        InstanceKind::Item(id) => id,
                        InstanceKind::Virtual(id, _) => id,
                        other => other.def_id(),
                    })
                    .unwrap_or(*def_id);

                call_sites.push(CallSite {
                    location: Location {
                        block,
                        statement_index: block_data.statements.len(),
                    },
                    caller,
                    callee,
                    arg_locals: args.iter().map(|a| operand_local(&a.node)).collect(),
                });
            }
            TyKind::FnPtr(..) => {
                // Calls through a function pointer have no statically known
                // target and are left out of the graph.
                crate::report::skipped_function_pointer(block, caller);
            }
            _ => {}
        }
    }

    call_sites
}

/// Maps each call-terminator location to its resolved callee.
pub fn callee_locations(call_sites: &[CallSite]) -> HashMap<Location, DefId> {
    call_sites
        .iter()
        .map(|cs| (cs.location, cs.callee))
        .collect()
}
