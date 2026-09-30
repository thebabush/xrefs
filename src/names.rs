//! Symbolic names attached to exact virtual addresses.
//!
//! [`NameTable`] is an immutable, VA-sorted multimap built once after loading
//! (via [`NameTableBuilder`]) and queried per xref endpoint.  Lookups are exact
//! VA matches only and return a borrowed slice — no allocation on the hot path.
//! Names are stored raw (no demangling or normalisation).

use crate::loader::Symbol;
use crate::va::Va;

/// VA-sorted table mapping an address to every name defined at exactly that address.
///
/// Stored as parallel arrays sorted by `(va, name)` with exact duplicates
/// removed, so [`names_at`](Self::names_at) is a `partition_point` range slice.
#[derive(Default, Debug, Clone)]
pub struct NameTable {
    vas: Vec<Va>,
    names: Vec<Box<str>>,
}

impl NameTable {
    /// Start an empty builder.
    pub fn builder() -> NameTableBuilder {
        NameTableBuilder::default()
    }

    /// Build a table from loader symbols.
    pub fn from_symbols(symbols: &[Symbol]) -> Self {
        let mut b = NameTableBuilder::with_capacity(symbols.len());
        for s in symbols {
            b.insert(s.va, &s.name);
        }
        b.finish()
    }

    /// Build a table from loader symbols plus additional names.
    pub fn from_symbols_and_extra(symbols: &[Symbol], extra: &[Symbol]) -> Self {
        let mut b = NameTableBuilder::with_capacity(symbols.len() + extra.len());
        for s in symbols.iter().chain(extra) {
            b.insert(s.va, &s.name);
        }
        b.finish()
    }

    /// True if the table holds no names.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Number of (va, name) entries.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// All names defined at exactly `va`, sorted; empty slice if none.
    pub fn names_at(&self, va: Va) -> &[Box<str>] {
        let lo = self.vas.partition_point(|&v| v < va);
        let hi = lo + self.vas[lo..].partition_point(|&v| v <= va);
        &self.names[lo..hi]
    }
}

/// Accumulates `(va, name)` pairs in any order; [`finish`](Self::finish) sorts
/// and de-duplicates them into a [`NameTable`].
#[derive(Default, Debug)]
pub struct NameTableBuilder {
    entries: Vec<(Va, Box<str>)>,
}

impl NameTableBuilder {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            entries: Vec::with_capacity(cap),
        }
    }

    /// Record `name` at `va`.  Empty names are ignored.
    pub fn insert(&mut self, va: Va, name: &str) {
        if !name.is_empty() {
            self.entries.push((va, name.into()));
        }
    }

    /// Sort by `(va, name)`, drop exact duplicates, and freeze into a table.
    pub fn finish(mut self) -> NameTable {
        self.entries.sort_unstable();
        self.entries.dedup();
        let (vas, names) = self.entries.into_iter().unzip();
        NameTable { vas, names }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strs(names: &[Box<str>]) -> Vec<&str> {
        names.iter().map(|n| &**n).collect()
    }

    fn sym(name: &str, va: u64) -> Symbol {
        Symbol {
            name: name.to_string(),
            va: Va::new(va),
        }
    }

    #[test]
    fn test_empty() {
        let t = NameTable::default();
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
        assert!(t.names_at(Va::new(0x1000)).is_empty());
        assert!(NameTable::from_symbols(&[]).is_empty());
    }

    #[test]
    fn test_single() {
        let t = NameTable::from_symbols(&[sym("main", 0x1000)]);
        assert!(!t.is_empty());
        assert_eq!(strs(t.names_at(Va::new(0x1000))), ["main"]);
    }

    #[test]
    fn test_multi_name_at_one_va() {
        let t = NameTable::from_symbols(&[
            sym("b_alias", 0x2000),
            sym("a_name", 0x2000),
            sym("other", 0x3000),
        ]);
        assert_eq!(strs(t.names_at(Va::new(0x2000))), ["a_name", "b_alias"]);
        assert_eq!(strs(t.names_at(Va::new(0x3000))), ["other"]);
    }

    #[test]
    fn test_dedup() {
        let t = NameTable::from_symbols(&[sym("f", 0x10), sym("f", 0x10), sym("f", 0x20)]);
        assert_eq!(t.len(), 2);
        assert_eq!(strs(t.names_at(Va::new(0x10))), ["f"]);
        assert_eq!(strs(t.names_at(Va::new(0x20))), ["f"]);
    }

    #[test]
    fn test_unsorted_input_via_builder() {
        let mut b = NameTable::builder();
        b.insert(Va::new(0x30), "c");
        b.insert(Va::new(0x10), "a");
        b.insert(Va::new(0x20), "b");
        b.insert(Va::new(0x10), "a2");
        let t = b.finish();
        assert_eq!(strs(t.names_at(Va::new(0x10))), ["a", "a2"]);
        assert_eq!(strs(t.names_at(Va::new(0x20))), ["b"]);
        assert_eq!(strs(t.names_at(Va::new(0x30))), ["c"]);
    }

    #[test]
    fn test_miss() {
        let t = NameTable::from_symbols(&[sym("f", 0x10), sym("g", 0x30)]);
        assert!(t.names_at(Va::new(0x0f)).is_empty());
        assert!(t.names_at(Va::new(0x11)).is_empty());
        assert!(t.names_at(Va::new(0x20)).is_empty());
        assert!(t.names_at(Va::new(0x31)).is_empty());
    }

    #[test]
    fn test_empty_names_ignored() {
        let t = NameTable::from_symbols(&[sym("", 0x10)]);
        assert!(t.is_empty());
    }
}
