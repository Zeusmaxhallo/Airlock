//! Stage 4 — auth-storage marking.
//!
//! A sender comparison names the principal the caller is checked against, but
//! only indirectly: as a MIR local. This stage traces that counterpart
//! backwards through copies, references and `?` glue to the call it came from.
//! Where that call is a `cw-storage-plus` load, the storage constant it reads
//! is the authorization state of the contract, and the corresponding inventory
//! item is marked authorization-relevant.
//!
//! Stage 6 reports only writes to items marked here, which is what keeps the
//! analysis focused on access control rather than on every storage write.

use std::collections::{HashMap, HashSet};

use rustc_middle::mir::{
    BasicBlock, Body, Local, Location, Operand, Rvalue, StatementKind, TerminatorKind,
};
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::DefId;

use crate::cosmwasm::{is_forwarding_glue_fn, is_storage_load_fn};
use crate::mir_util::{callee_def_id, operand_local};
use crate::sender_comparisons::SenderComparison;
use crate::storage_inventory::{ConstDefIndex, ConstTypeIndex, resolve_storage_item};

/// A storage item that a sender comparison was seen to read.
#[derive(Debug, Clone)]
pub struct AuthStateVariable {
    /// Name of the storage constant, or its type when the constant could not
    /// be resolved.
    pub symbolic_name: String,
    /// The resolved storage constant. `None` when only the type-based fallback
    /// applied, in which case no inventory item can be marked.
    pub storage_def_id: Option<DefId>,
    /// Location of the load the comparison's counterpart originates from.
    pub load_location: Location,
}

/// Upper bound on the backward trace: how many local-and-block steps are
/// followed before the search for the originating load is abandoned.
const MAX_TRACE_STEPS: usize = 24;

/// Traces the counterpart of every comparison in `comparisons` back to a
/// storage load and reports the items those loads read.
pub fn find_auth_state_variables<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    comparisons: &[SenderComparison],
    const_defs: &ConstDefIndex,
    const_types: &ConstTypeIndex<'tcx>,
) -> Vec<AuthStateVariable> {
    if comparisons.is_empty() {
        return Vec::new();
    }

    let sources = LocalSourceMap::build(tcx, body);
    let mut result = Vec::new();

    for cmp in comparisons {
        let Some(LocalSource::CallReturn {
            callee,
            arg_locals,
            location,
        }) = sources.trace_to_call(tcx, cmp.compared_local, cmp.location.block, body)
        else {
            continue;
        };

        // Only a `cw-storage-plus` load yields an authorization state.
        if !callee.is_some_and(|d| is_storage_load_fn(tcx, d)) {
            continue;
        }

        // Argument 0 of the load is the receiver, i.e. the storage constant.
        let Some(receiver) = arg_locals.first().copied().flatten() else {
            continue;
        };

        let (symbolic_name, storage_def_id) =
            resolve_storage_item(tcx, body, receiver, const_defs, const_types);

        result.push(AuthStateVariable {
            symbolic_name,
            storage_def_id,
            load_location: location,
        });
    }

    result
}

/// Where the value in a local came from, as far as the backward trace needs to
/// distinguish it.
#[derive(Debug, Clone)]
enum LocalSource {
    /// The local copies, moves, borrows or casts another local.
    CopiedFrom(Local),
    /// The local receives the return value of a call.
    CallReturn {
        callee: Option<DefId>,
        arg_locals: Vec<Option<Local>>,
        location: Location,
    },
}

/// Per-body index of the value sources of each local, kept per basic block
/// because the trace walks the control-flow graph backwards.
struct LocalSourceMap {
    by_block_local: HashMap<(BasicBlock, Local), LocalSource>,
}

impl LocalSourceMap {
    fn build<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) -> Self {
        let mut by_block_local = HashMap::new();

        for (block, block_data) in body.basic_blocks.iter_enumerated() {
            for stmt in block_data.statements.iter() {
                let StatementKind::Assign(assign) = &stmt.kind else {
                    continue;
                };
                let (lhs, rhs) = assign.as_ref();
                let source = match rhs {
                    Rvalue::Use(Operand::Copy(place) | Operand::Move(place), _)
                    | Rvalue::Ref(_, _, place)
                    | Rvalue::RawPtr(_, place) => Some(LocalSource::CopiedFrom(place.local)),
                    Rvalue::Cast(_, op, _) => operand_local(op).map(LocalSource::CopiedFrom),
                    _ => None,
                };
                if let Some(source) = source {
                    by_block_local.insert((block, lhs.local), source);
                }
            }

            if let TerminatorKind::Call {
                func,
                args,
                destination,
                ..
            } = &block_data.terminator().kind
            {
                by_block_local.insert(
                    (block, destination.local),
                    LocalSource::CallReturn {
                        callee: callee_def_id(tcx, body, func),
                        arg_locals: args.iter().map(|a| operand_local(&a.node)).collect(),
                        location: Location {
                            block,
                            statement_index: block_data.statements.len(),
                        },
                    },
                );
            }
        }

        LocalSourceMap { by_block_local }
    }

    /// Follows `start_local` backwards from `start_block` until it reaches the
    /// call that produced its value.
    ///
    /// Within a block the assignment chain is followed directly; where the
    /// local is not defined in the block, the search continues in all
    /// predecessors. Forwarding glue — the `?` desugaring, `unwrap`,
    /// `map_err`, deref coercions — is stepped through by continuing with the
    /// call's first argument, so that a load behind `STATE.load(store)?` is
    /// still found.
    fn trace_to_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        start_local: Local,
        start_block: BasicBlock,
        body: &Body<'tcx>,
    ) -> Option<LocalSource> {
        let predecessors = body.basic_blocks.predecessors();
        let mut queue: Vec<(BasicBlock, Local)> = vec![(start_block, start_local)];
        let mut visited: HashSet<(BasicBlock, Local)> = HashSet::new();

        for _ in 0..MAX_TRACE_STEPS {
            let (block, local) = queue.pop()?;

            if !visited.insert((block, local)) {
                continue;
            }

            match self.by_block_local.get(&(block, local)) {
                Some(LocalSource::CallReturn {
                    callee,
                    arg_locals,
                    location,
                }) => {
                    if callee.is_some_and(|d| is_forwarding_glue_fn(tcx, d)) {
                        let next_local = arg_locals.first().copied().flatten()?;
                        queue.push((location.block, next_local));
                        continue;
                    }
                    return Some(LocalSource::CallReturn {
                        callee: *callee,
                        arg_locals: arg_locals.clone(),
                        location: *location,
                    });
                }
                Some(LocalSource::CopiedFrom(next_local)) => {
                    queue.push((block, *next_local));
                }
                None => {
                    // Not defined in this block: continue in every predecessor.
                    for &pred in &predecessors[block] {
                        queue.push((pred, local));
                    }
                }
            }
        }

        None
    }
}
