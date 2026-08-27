//! Stage 1 — storage inventory.
//!
//! Collects every `cw-storage-plus` constant (`Item`/`Map`) the crate defines,
//! and provides the two indices with which a MIR local can later be traced
//! back to one of those constants. Stages 4 and 6 need that mapping: stage 4
//! marks the items an authorization check reads, stage 6 recognises writes to
//! exactly those items.

use std::collections::{HashMap, HashSet};

use rustc_hir::def::DefKind;
use rustc_middle::mir::{Body, Const, Local, Operand, Rvalue, StatementKind};
use rustc_middle::ty::{Ty, TyCtxt, TyKind};
use rustc_span::def_id::DefId;

use crate::cosmwasm::crate_name_is;
use crate::mir_util::{normalize_ty_str, strip_refs};

/// The storage constants defined by the crate under analysis.
#[derive(Debug, Clone)]
pub struct StorageInventory {
    pub items: Vec<StorageItem>,
}

#[derive(Debug, Clone)]
pub struct StorageItem {
    pub name: String,
    pub def_id: DefId,
    pub ty_string: String,
    pub kind: StorageItemKind,
    /// Set by stage 4 once a sender comparison was seen to read this item.
    pub is_auth: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum StorageItemKind {
    Item,
    Map,
}

impl StorageInventory {
    /// Walks the crate's free items and records every `const` whose type is a
    /// `cw_storage_plus::Item` or `cw_storage_plus::Map`.
    pub fn build(tcx: TyCtxt<'_>) -> Self {
        let mut items = Vec::new();

        for item_id in tcx.hir_free_items() {
            let item = tcx.hir_item(item_id);

            if !matches!(item.kind, rustc_hir::ItemKind::Const(..)) {
                continue;
            }

            let def_id = item_id.owner_id.def_id.to_def_id();
            let ty = tcx.type_of(def_id).skip_binder();

            let kind = match ty.kind() {
                TyKind::Adt(adt_def, _) if crate_name_is(tcx, adt_def.did(), "cw_storage_plus") => {
                    match tcx.item_name(adt_def.did()).as_str() {
                        "Item" => StorageItemKind::Item,
                        "Map" => StorageItemKind::Map,
                        _ => continue,
                    }
                }
                _ => continue,
            };

            items.push(StorageItem {
                name: tcx.item_name(def_id).to_string(),
                def_id,
                ty_string: normalize_ty_str(&ty.to_string()),
                kind,
                is_auth: false,
            });
        }

        StorageInventory { items }
    }

    /// Marks the given items as authorization-relevant.
    ///
    /// Matching is by `DefId`, not by name: several storage constants may share
    /// a name across modules (a `CONFIG` in `state` and one in `migrations::*`),
    /// and name matching would mark all of them.
    pub fn mark_auth(&mut self, auth_items: impl IntoIterator<Item = DefId>) {
        let marked: HashSet<DefId> = auth_items.into_iter().collect();
        if marked.is_empty() {
            return;
        }
        for item in &mut self.items {
            if marked.contains(&item.def_id) {
                item.is_auth = true;
            }
        }
    }

    /// Names of the items marked authorization-relevant. Stage 6 reports a
    /// write as a sink when the written item resolves to one of these.
    pub fn auth_item_names(&self) -> HashSet<&str> {
        self.items
            .iter()
            .filter(|i| i.is_auth)
            .map(|i| i.name.as_str())
            .collect()
    }
}

/// Local `const` items indexed by their region-erased type.
///
/// Serves as the fallback for storage receivers whose constant cannot be read
/// off the MIR operand. Two constants of the same type are indistinguishable
/// on that path, so the first one wins — which is why the index is built in
/// the definition order of [`TyCtxt::iter_local_def_id`] and never overwrites
/// an existing entry.
pub struct ConstTypeIndex<'tcx> {
    by_ty: HashMap<Ty<'tcx>, DefId>,
}

impl<'tcx> ConstTypeIndex<'tcx> {
    /// Builds the index once per crate. Scanning all local definitions per
    /// lookup instead would dominate the runtime of the taint fixpoints.
    pub fn build(tcx: TyCtxt<'tcx>) -> Self {
        let mut by_ty = HashMap::new();
        for local_def_id in tcx.iter_local_def_id() {
            if !matches!(tcx.def_kind(local_def_id), DefKind::Const { .. }) {
                continue;
            }
            let def_id = local_def_id.to_def_id();
            let ty = tcx.erase_and_anonymize_regions(tcx.type_of(def_id).skip_binder());
            by_ty.entry(ty).or_insert(def_id);
        }
        ConstTypeIndex { by_ty }
    }

    /// Looks up the constant whose type matches `ty` after peeling references
    /// and erasing regions. Both sides are compared as interned types rather
    /// than as formatted strings: MIR local types carry erased regions while
    /// `type_of` results carry early-bound ones, so their renderings differ
    /// even where the types are identical.
    fn lookup(&self, tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<DefId> {
        let erased = tcx.erase_and_anonymize_regions(strip_refs(ty));
        self.by_ty.get(&erased).copied()
    }
}

/// One step of the assignment chain that leads from a storage receiver local
/// back to the constant it refers to.
#[derive(Debug, Clone, Copy)]
enum ConstDef {
    /// The local is assigned a not yet evaluated local `const` item.
    Item(DefId),
    /// The local copies, moves, borrows or derefs another local.
    CopiedFrom(Local),
}

/// Per-body index of the assignments that can define a storage receiver, in
/// program order.
///
/// A receiver is built as `_a = &_b; _b = const ITEM`, and at MIR level the
/// operand still carries the constant's own `DefId` (`Const::Unevaluated`), so
/// the item can be identified exactly instead of by type identity. Indexing
/// the definitions once lets the chain be followed without rescanning the body
/// for every receiver.
pub struct ConstDefIndex {
    by_local: HashMap<Local, Vec<ConstDef>>,
}

impl ConstDefIndex {
    pub fn build<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) -> Self {
        let mut by_local: HashMap<Local, Vec<ConstDef>> = HashMap::new();

        for data in body.basic_blocks.iter() {
            for stmt in &data.statements {
                let StatementKind::Assign(assign) = &stmt.kind else {
                    continue;
                };
                let (lhs, rhs) = assign.as_ref();
                let def = match rhs {
                    Rvalue::Use(Operand::Constant(c), _) => {
                        let Const::Unevaluated(uv, _) = c.const_ else {
                            continue;
                        };
                        if !uv.def.is_local()
                            || !matches!(tcx.def_kind(uv.def), DefKind::Const { .. })
                        {
                            continue;
                        }
                        ConstDef::Item(uv.def)
                    }
                    Rvalue::Use(Operand::Copy(p) | Operand::Move(p), _)
                    | Rvalue::Ref(_, _, p)
                    | Rvalue::CopyForDeref(p) => ConstDef::CopiedFrom(p.local),
                    _ => continue,
                };
                by_local.entry(lhs.local).or_default().push(def);
            }
        }

        ConstDefIndex { by_local }
    }

    /// Follows the chain from `receiver` and returns the constant it refers to,
    /// or `None` when the chain ends in an already evaluated constant, in a
    /// non-local constant, or in an unsupported rvalue.
    fn resolve(&self, receiver: Local) -> Option<DefId> {
        let mut worklist = vec![receiver];
        let mut seen: HashSet<Local> = HashSet::new();

        while let Some(local) = worklist.pop() {
            if !seen.insert(local) {
                continue;
            }
            let Some(defs) = self.by_local.get(&local) else {
                continue;
            };
            for def in defs {
                match *def {
                    ConstDef::Item(def_id) => return Some(def_id),
                    ConstDef::CopiedFrom(next) => worklist.push(next),
                }
            }
        }

        None
    }
}

/// Resolves the storage constant behind a load or write receiver.
///
/// The exact path reads the constant's own `DefId` off the MIR operand
/// ([`ConstDefIndex`]); the fallback matches the receiver's type against the
/// crate's constants ([`ConstTypeIndex`]).
pub fn resolve_storage_def_id<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    receiver: Local,
    defs: &ConstDefIndex,
    consts: &ConstTypeIndex<'tcx>,
) -> Option<DefId> {
    defs.resolve(receiver)
        .or_else(|| consts.lookup(tcx, body.local_decls[receiver].ty))
}

/// Resolves a storage receiver to the constant's name and `DefId`. When
/// neither path resolves, the normalized type string is returned for display
/// only, without a `DefId`.
pub fn resolve_storage_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    receiver: Local,
    defs: &ConstDefIndex,
    consts: &ConstTypeIndex<'tcx>,
) -> (String, Option<DefId>) {
    match resolve_storage_def_id(tcx, body, receiver, defs, consts) {
        Some(def_id) => (tcx.item_name(def_id).to_string(), Some(def_id)),
        None => {
            let base = strip_refs(body.local_decls[receiver].ty);
            (normalize_ty_str(&format!("{:?}", base)), None)
        }
    }
}
