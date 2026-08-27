//! Recognition of the CosmWasm framework and of `cw-storage-plus`.
//!
//! Every place where the analysis has to decide "is this the framework?" is
//! collected here. All predicates match by defining crate plus item name
//! rather than by a formatted path, which keeps them allocation-free on the
//! hot paths of the taint fixpoints.

use rustc_middle::mir::Body;
use rustc_middle::ty::{Ty, TyCtxt, TyKind};
use rustc_span::{def_id::DefId, sym};

use crate::mir_util::strip_refs;

/// Returns whether `def_id` is defined in the crate called `name`.
pub fn crate_name_is(tcx: TyCtxt<'_>, def_id: DefId, name: &str) -> bool {
    tcx.crate_name(def_id.krate).as_str() == name
}

/// Returns whether the item name of `def_id` is `name`.
pub fn item_name_is(tcx: TyCtxt<'_>, def_id: DefId, name: &str) -> bool {
    tcx.item_name(def_id).as_str() == name
}

/// Identity-preserving glue that the analysis looks through: the `?`
/// desugaring (`branch`/`from_residual`), `Option`/`Result` adapters that pass
/// the success value on unchanged, and deref coercions.
///
/// `may_load` deliberately does not belong here — it is the load *sink*
/// ([`is_storage_load_fn`]); listing it as glue would make a backward trace
/// step through the load instead of stopping at it.
pub fn is_forwarding_glue_fn(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    matches!(
        tcx.item_name(def_id).as_str(),
        "branch"
            | "from_residual"
            | "unwrap"
            | "expect"
            | "ok_or"
            | "ok_or_else"
            | "map_err"
            | "into_ok"
            | "deref"
            | "deref_mut"
    )
}

/// Returns whether `def_id` is a mutating `cw-storage-plus` operation.
/// `update` counts as a write sink because it persists the closure's result.
pub fn is_storage_write_fn(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    crate_name_is(tcx, def_id, "cw_storage_plus")
        && matches!(tcx.item_name(def_id).as_str(), "save" | "insert" | "update")
}

/// Returns whether `def_id` is a reading `cw-storage-plus` operation.
pub fn is_storage_load_fn(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    crate_name_is(tcx, def_id, "cw_storage_plus")
        && matches!(tcx.item_name(def_id).as_str(), "load" | "may_load" | "get")
}

/// Returns whether `def_id` identifies the `Result` enum.
pub fn is_result_def(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    tcx.is_diagnostic_item(sym::Result, def_id)
}

/// Returns whether `ty` is `core::result::Result<_, _>`.
pub fn is_result_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    matches!(ty.kind(), TyKind::Adt(adt_def, _)
        if tcx.is_diagnostic_item(sym::Result, adt_def.did()))
}

/// Returns whether `def_id` is `cosmwasm_std::MessageInfo`.
pub fn is_message_info(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    crate_name_is(tcx, def_id, "cosmwasm_std") && item_name_is(tcx, def_id, "MessageInfo")
}

/// Returns whether `ty`, through any reference layers, is
/// `cosmwasm_std::MessageInfo`.
pub fn is_message_info_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    matches!(strip_refs(ty).kind(), TyKind::Adt(adt_def, _)
        if is_message_info(tcx, adt_def.did()))
}

/// Returns whether `ty` is a CosmWasm framework type.
///
/// The framework types (`Deps`, `DepsMut`, `OwnedDeps`, `MessageInfo`, `Env`)
/// and the `dyn Storage` trait object are supplied by the runtime, not by the
/// caller of a handler, and are therefore excluded from taint seeding.
/// Reference layers are stripped before the check.
pub fn is_framework_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    match strip_refs(ty).kind() {
        TyKind::Adt(adt_def, _) => {
            let did = adt_def.did();
            crate_name_is(tcx, did, "cosmwasm_std")
                && matches!(
                    tcx.item_name(did).as_str(),
                    "MessageInfo" | "Deps" | "DepsMut" | "OwnedDeps" | "Env"
                )
        }
        TyKind::Dynamic(preds, ..) => preds.principal_def_id().is_some_and(|d| {
            crate_name_is(tcx, d, "cosmwasm_std") && item_name_is(tcx, d, "Storage")
        }),
        _ => false,
    }
}

/// Returns whether `def_id` is one of the CosmWasm address types.
pub fn is_cosmwasm_addr(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    crate_name_is(tcx, def_id, "cosmwasm_std")
        && matches!(tcx.item_name(def_id).as_str(), "Addr" | "CanonicalAddr")
}

/// Returns whether `def_id` is the standard library's owned string type
/// (`alloc::string::String`, re-exported as `std::string::String`).
///
/// Matching by crate plus item name rather than through the `String`
/// diagnostic item is deliberate: the diagnostic-item lookup does not resolve
/// on the pinned toolchain, which would leave `String` and `Option<String>`
/// message fields unrecognised as taint sources.
pub fn is_std_string(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    item_name_is(tcx, def_id, "String")
        && (crate_name_is(tcx, def_id, "alloc") || crate_name_is(tcx, def_id, "std"))
}

/// Locates the `execute` entry point of the contract crate.
///
/// CosmWasm contracts expose their state-changing operations through a single
/// free function named `execute`, so the search is over the crate's free items
/// in the HIR.
pub fn find_execute(tcx: TyCtxt<'_>) -> Option<DefId> {
    for item_id in tcx.hir_free_items() {
        let item = tcx.hir_item(item_id);

        if !matches!(item.kind, rustc_hir::ItemKind::Fn { .. }) {
            continue;
        }

        let def_id = item_id.owner_id.def_id.to_def_id();
        if item_name_is(tcx, def_id, "execute") {
            return Some(def_id);
        }
    }

    None
}

/// Returns whether a MIR body belongs to a function the compiler can hand us
/// a body for: locally defined and with MIR available. Every stage skips the
/// remaining nodes, because a dependency's body cannot be inspected.
pub fn is_analysable(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    def_id.is_local() && tcx.is_mir_available(def_id)
}

/// Convenience wrapper returning the MIR body of an analysable function.
pub fn body_of<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> Option<&'tcx Body<'tcx>> {
    is_analysable(tcx, def_id).then(|| tcx.optimized_mir(def_id))
}
