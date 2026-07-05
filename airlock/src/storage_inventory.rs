use crate::utility;
use rustc_middle::{
    mir::{Local, Location},
    ty::{TyCtxt, TyKind},
};
use rustc_span::def_id::DefId;

#[derive(Debug, Clone)]
pub struct StorageInventory {
    pub items: Vec<StorageItem>,
    pub auth_state_variables: Vec<AuthStateVariable>,
}

#[derive(Debug, Clone)]
pub struct StorageItem {
    pub name: String,
    pub def_id: DefId,
    pub ty_string: String,
    pub kind: StorageItemKind,
    pub is_auth: bool,
}

#[derive(Debug, Clone)]
pub struct AuthStateVariable {
    pub compared_local: Local,
    pub storage_item_local: Local,
    pub symbolic_name: String,
    pub load_location: Location,
}

#[derive(Debug, Clone)]
pub enum StorageItemKind {
    Item,
    Map,
}

impl StorageInventory {
    pub fn new() -> Self {
        StorageInventory {
            items: Vec::new(),
            auth_state_variables: Vec::new(),
        }
    }

    pub fn build(tcx: TyCtxt<'_>) -> Self {
        let mut inventory = Self::new();

        for item_id in tcx.hir_free_items() {
            let item = tcx.hir_item(item_id);

            if !matches!(item.kind, rustc_hir::ItemKind::Const(..)) {
                continue;
            }

            let def_id = item_id.owner_id.def_id.to_def_id();
            let ty = tcx.type_of(def_id).skip_binder();
            let ty_string = utility::normalize_ty_str(&ty.to_string());
            let storage_kind = match ty.kind() {
                TyKind::Adt(adt_def, _)
                    if utility::crate_name_is(tcx, adt_def.did(), "cw_storage_plus") =>
                {
                    match tcx.item_name(adt_def.did()).as_str() {
                        "Item" => StorageItemKind::Item,
                        "Map" => StorageItemKind::Map,
                        _ => continue,
                    }
                }
                _ => continue,
            };

            let name = tcx.item_name(def_id).to_string();

            inventory.items.push(StorageItem {
                name,
                def_id,
                ty_string,
                kind: storage_kind,
                is_auth: false,
            });
        }

        inventory
    }

    pub fn print_inventory(&self) {
        eprintln!("══ Storage Inventory ══");
        for item in &self.items {
            let auth = if item.is_auth { "  [auth]" } else { "" };
            eprintln!(
                "Name: {}, DefId: {:?}, Type: {}, Kind: {:?},  {}",
                item.name, item.def_id, item.ty_string, item.kind, auth
            );
        }
    }
}
