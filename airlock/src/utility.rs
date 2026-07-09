use rustc_middle::{
    mir::{Body, Operand},
    ty::{TyCtxt, TyKind},
};
use rustc_span::def_id::DefId;

/// Check if the crate name of the given `def_id` matches the provided `name`.
pub fn crate_name_is(tcx: TyCtxt<'_>, def_id: DefId, name: &str) -> bool {
    tcx.crate_name(def_id.krate).as_str() == name
}

/// Find the Execute Entry point
pub fn find_execute(tcx: TyCtxt<'_>) -> Option<DefId> {
    for item_id in tcx.hir_free_items() {
        let item = tcx.hir_item(item_id);

        if let rustc_hir::ItemKind::Fn { .. } = item.kind {
            let def_id = item_id.owner_id.def_id.to_def_id();
            let fn_name = tcx.item_name(def_id);

            if fn_name.as_str() == "execute" {
                eprintln!("Found Execute Entry Point: {:?}", tcx.def_path_str(def_id));
                return Some(def_id);
            }
        }
    }

    None
}

pub fn normalize_ty_str(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\'' {
            // Lifetime mit Apostroph überspringen: 'erased, '_, 'a
            while let Some(&next) = chars.peek() {
                if next.is_alphanumeric() || next == '_' {
                    chars.next();
                } else {
                    break;
                }
            }
        } else if c == '{' {
            // Anonyme Lifetime / erased ohne Apostroph überspringen: {erased}
            while let Some(&next) = chars.peek() {
                chars.next();
                if next == '}' {
                    break;
                }
            }
        } else {
            result.push(c);
        }
    }
    // Whitespace normalisieren und übrige Kommas säubern: "Item<, Addr>" → "Item<Addr>"
    let result = result
        .replace(", ,", ",")
        .replace("<,", "<")
        .replace(", >", ">")
        .replace(",>", ">");
    result.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Returns the 'DefId' of the callee if the call target is a direct
/// function definition.
/// Returns `None` for indirect calls (e.g., through function pointers
/// or dynamic dispatch), where the callee cannot be resolved statically.
pub fn callee_def_id<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    func: &Operand<'tcx>,
) -> Option<DefId> {
    match func.ty(&body.local_decls, tcx).kind() {
        TyKind::FnDef(def_id, _) => Some(*def_id),
        _ => None,
    }
}

/// Identity-preserving glue the load trace looks through: `?`-desugaring
/// (`branch`/`from_residual`), `Option`/`Result` adapters that pass the
/// success value through unchanged, and deref coercions. `may_load` does NOT
/// belong here — it is the load *sink* (`is_storage_load_fn`); listing it as
/// glue would make the trace step through the load instead of returning it.
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

/// Returns `true` if the given function is a mutating `cw_storage_plus` storage
/// operation (`save`, `insert`, or `update`). `update` is treated as a write sink.
pub fn is_storage_write_fn(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    crate_name_is(tcx, def_id, "cw_storage_plus")
        && matches!(tcx.item_name(def_id).as_str(), "save" | "insert" | "update")
}

/// Returns `true` if the given function is a `cw_storage_plus` storage load
/// operation (`load`, `may_load`, or `get`).
pub fn is_storage_load_fn(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    crate_name_is(tcx, def_id, "cw_storage_plus")
        && matches!(tcx.item_name(def_id).as_str(), "load" | "may_load" | "get")
}
