use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::DefId;

/// Check if the crate name of the given `def_id` matches the provided `name`.
pub fn crate_name_is(tcx: TyCtxt<'_>, def_id: DefId, name: &str) -> bool {
    tcx.crate_name(def_id.krate).as_str() == name
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
