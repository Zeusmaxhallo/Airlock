use rustc_hir::def_id::DefId;
use rustc_middle::mir::{Body, Local, Location, Operand, TerminatorKind};
use rustc_middle::ty::{self, Instance, InstanceKind, TyCtxt, TyKind};
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Default)]
pub struct CallGraph {
    pub edges: HashMap<DefId, Vec<DefId>>,
    pub callers: HashMap<DefId, Vec<DefId>>,
    pub nodes: HashSet<DefId>,
    /// call sites per caller, in MIR order
    pub call_sites: HashMap<DefId, Vec<CallSite>>,
}

/// A resolved call at a concrete program point. Records the callee and
/// the actual-to-formal parameter wiring needed for interprocedural
/// propagation: actual argument at position `i` corresponds to the
/// callee's formal parameter `Local::from_usize(i + 1)`.
#[derive(Debug, Clone)]
pub struct CallSite {
    /// Location of the `Call` terminator in the caller.
    pub location: Location,
    /// Function that contains this call 
    pub caller: DefId,
    /// Resolved callee.
    pub callee: DefId,
    /// Actual arguments as caller locals; `None` for constant operands.
    pub arg_locals: Vec<Option<Local>>,
    /// Caller local receiving the return value.
    pub destination: Local,
}

impl CallGraph {
    pub fn new() -> Self {
        CallGraph {
            edges: HashMap::new(),
            callers: HashMap::new(),
            nodes: HashSet::new(),
            call_sites: HashMap::new(),
        }
    }

    pub fn add_edge(&mut self, caller: DefId, callee: DefId) {
        self.edges.entry(caller).or_default().push(callee);
        self.callers.entry(callee).or_default().push(caller);
        self.nodes.insert(callee);
        self.nodes.insert(caller);
    }

    pub fn callees(&self, fn_id: DefId) -> &[DefId] {
        self.edges.get(&fn_id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn call_sites(&self, fn_id: DefId) -> &[CallSite] {
        self.call_sites
            .get(&fn_id)
            .map(|cs| cs.as_slice())
            .unwrap_or(&[])
    }

    pub fn callers_of(&self, fn_id: DefId) -> &[DefId] {
        self.callers
            .get(&fn_id)
            .map(|c| c.as_slice())
            .unwrap_or(&[])
    }

    pub fn all_callers_of(&self, fn_id: DefId) -> HashSet<DefId> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(fn_id);
        visited.insert(fn_id);
        while let Some(current) = queue.pop_front() {
            for &caller in self.callers_of(current) {
                if visited.insert(caller) {
                    queue.push_back(caller);
                }
            }
        }
        visited
    }

    pub fn all_callees_of(&self, fn_id: DefId) -> HashSet<DefId> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(fn_id);
        visited.insert(fn_id);

        while let Some(current) = queue.pop_front() {
            for &callee in self.callees(current) {
                if visited.insert(callee) {
                    queue.push_back(callee);
                }
            }
        }

        visited
    }

    pub fn build_from_root(tcx: TyCtxt<'_>, root: DefId) -> Self {
        let mut graph = CallGraph::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(root);
        queue.push_back(root);
        // The root must be a node even if none of its calls resolve —
        // otherwise crates whose `execute` has no resolvable callees are
        // skipped by every analysis stage.
        graph.nodes.insert(root);

        while let Some(caller_id) = queue.pop_front() {
            if !caller_id.is_local() || !tcx.is_mir_available(caller_id) {
                continue;
            }

            let body = tcx.optimized_mir(caller_id);

            for call_site in collect_call_sites(tcx, body) {
                let callee_id = call_site.callee;
                graph.add_edge(caller_id, callee_id);
                graph
                    .call_sites
                    .entry(callee_id)
                    .or_default()
                    .push(call_site);
                if visited.insert(callee_id) {
                    queue.push_back(callee_id);
                }
            }
        }

        eprintln!(
            "[call_graph] Reachable functions from {:?}: {}",
            root,
            graph.nodes.len()
        );

        graph
    }
}

fn operand_local(operand: &Operand<'_>) -> Option<Local> {
    match operand {
        Operand::Copy(place) | Operand::Move(place) => Some(place.local),
        _ => None,
    }
}

fn collect_call_sites<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) -> Vec<CallSite> {
    let mut call_sites = Vec::new();
    let caller = body.source.def_id();
    // NICHT fully_monomorphized(): build_from_root läuft auch in generische
    // Funktionen (z. B. levana QueryablePair::Request<T>, transmuter
    // impl IntoIterator). Deren MIR enthält unsubstituierte Parameter; im
    // Codegen-TypingMode ist fehlgeschlagene Normalisierung ein ICE
    // (normalize_erasing_regions.rs "Failed to normalize Alias").
    let typing_env = ty::TypingEnv::post_analysis(tcx, body.source.def_id());

    for (block, block_data) in body.basic_blocks.iter_enumerated() {
        let terminator = block_data.terminator();

        if let TerminatorKind::Call {
            func,
            args,
            destination,
            ..
        } = &terminator.kind
        {
            let ty = func.ty(&body.local_decls, tcx);

            match ty.kind() {
                TyKind::FnDef(def_id, generic_args) => {
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
                        arg_locals: (args.iter().map(|a| operand_local(&a.node)).collect()),
                        destination: destination.local,
                    });
                }
                TyKind::FnPtr(..) => {
                    // Function pointer: static analysis cannot resolve the callee -> skip
                    eprintln!(
                        "[call_graph] Skipping function pointer call at {:?} in {:?}",
                        block,
                        body.source.def_id()
                    );
                }
                _ => continue,
            }
        }
    }

    call_sites
}
