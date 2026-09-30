//! Output types and format-specific printers for the `xr` CLI.
//!
//! The pipeline is:
//!   1. Build a `Vec<XrefRecord>` (xref + optional disasm context) once.
//!   2. Pick a `Printer` impl based on `--format`.
//!   3. Call `printer.print(&records)`.

use crate::disasm::DisasmLine;
use crate::va::Va;
use crate::xref::{Confidence, XrefKind};
use serde::Serialize;
use std::fmt::Write as FmtWrite;
use std::io::Write as IoWrite;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Truncate a string with a middle ellipsis if it exceeds `max_width`.
///
/// If `max_width` is 0, returns the original string unchanged.
/// Otherwise, keeps roughly half from the start and half from the end,
/// joined by "...".
pub fn truncate_middle(s: &str, max_width: usize) -> String {
    if max_width == 0 {
        return s.to_string();
    }

    let char_count = s.chars().count();
    if char_count <= max_width {
        return s.to_string();
    }

    // Account for the ellipsis (3 chars)
    if max_width <= 3 {
        return s.chars().take(max_width).collect();
    }

    let available = max_width - 3; // "..."
    let left_len = available.div_ceil(2); // slightly favor left side
    let right_len = available / 2;

    let left: String = s.chars().take(left_len).collect();
    // Skip to the tail without collecting-and-reversing twice.
    let right: String = s.chars().skip(char_count - right_len).collect();

    format!("{left}...{right}")
}

/// Build a space-separated lowercase hex string from raw bytes.
/// Single pre-allocated `String` — no intermediate `Vec<String>`.
fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 3);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 {
            s.push(' ');
        }
        // Writing to a String is infallible — fmt::Write for String never errors.
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Append ` <a|b>` (leading space) to `buf`; nothing when `names` is empty.
/// Control characters are escaped so a record always stays on one line.
fn write_text_names(names: &[Box<str>], buf: &mut Vec<u8>) {
    if names.is_empty() {
        return;
    }
    buf.extend_from_slice(b" <");
    for (i, name) in names.iter().enumerate() {
        if i > 0 {
            buf.push(b'|');
        }
        for c in name.chars() {
            match c {
                '\n' => buf.extend_from_slice(b"\\n"),
                '\r' => buf.extend_from_slice(b"\\r"),
                '\t' => buf.extend_from_slice(b"\\t"),
                c if c.is_control() => {
                    let _ = write!(buf, "\\x{:02x}", c as u32);
                }
                c => {
                    let mut tmp = [0u8; 4];
                    buf.extend_from_slice(c.encode_utf8(&mut tmp).as_bytes());
                }
            }
        }
    }
    buf.push(b'>');
}

/// Append `names` joined with `|` as one CSV field, quoted per RFC 4180 only
/// when it contains a comma, quote, or line break (newlines are backslash-escaped
/// so a record stays on one row, as for the `string` column).
fn write_csv_names(names: &[Box<str>], buf: &mut Vec<u8>) {
    let needs_quote = names.iter().any(|n| n.contains([',', '"', '\n', '\r']));
    if needs_quote {
        buf.push(b'"');
    }
    for (i, name) in names.iter().enumerate() {
        if i > 0 {
            buf.push(b'|');
        }
        for &b in name.as_bytes() {
            match b {
                b'"' => buf.extend_from_slice(b"\"\""),
                b'\n' => buf.extend_from_slice(b"\\n"),
                b'\r' => buf.extend_from_slice(b"\\r"),
                b => buf.push(b),
            }
        }
    }
    if needs_quote {
        buf.push(b'"');
    }
}

// ── Data types ────────────────────────────────────────────────────────────────

/// A single line of disassembly or hex context around an xref site.
#[derive(Serialize)]
pub struct ContextLine {
    pub va: Va,
    /// Raw bytes as a lowercase hex string (space-separated, e.g. `"48 89 c7"`).
    pub hex: String,
    /// Formatted instruction text, or `"(data)"` for non-code regions.
    pub text: String,
    /// True for the instruction at the xref `from` address.
    pub focus: bool,
}

impl ContextLine {
    pub fn from_disasm(line: &DisasmLine) -> Self {
        Self {
            va: Va::new(line.va),
            hex: bytes_to_hex(&line.bytes),
            text: line.text.clone(),
            focus: line.is_focus,
        }
    }

    pub fn data(va: Va, raw: &[u8]) -> Self {
        Self {
            va,
            hex: bytes_to_hex(raw),
            text: "(data)".to_string(),
            focus: true,
        }
    }
}

/// A fully resolved xref, optionally annotated with disasm context.
#[derive(Serialize)]
pub struct XrefRecord<'a> {
    pub from: Va,
    pub to: Va,
    /// Kind label (`"call"`, `"jump"`, `"data_read"`, `"data_write"`, `"data_ptr"`).
    #[serde(serialize_with = "serialize_kind")]
    pub kind: XrefKind,
    /// Confidence label (e.g. `"pair-resolved"`).
    #[serde(serialize_with = "serialize_confidence")]
    pub confidence: Confidence,
    /// Present when `-A`/`-B` (after/before context) is non-zero.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<ContextLine>>,
    /// Extracted string literal (when `--rust` and this is a data_ptr into a string blob).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "string")]
    pub rust_string: Option<String>,
    /// Symbol names at exactly `from` (empty when none or `--no-names`).
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    pub from_names: &'a [Box<str>],
    /// Symbol names at exactly `to` (empty when none or `--no-names`).
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    pub to_names: &'a [Box<str>],
}

fn serialize_kind<S: serde::Serializer>(k: &XrefKind, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(k.name())
}

fn serialize_confidence<S: serde::Serializer>(c: &Confidence, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(c.name())
}

// ── Printer trait ─────────────────────────────────────────────────────────────

/// Format and output xref records.
///
/// The hot path is:
///   1. `write_record` is called in parallel (one call per record, per thread)
///      appending into a caller-supplied `Vec<u8>`.  No per-record allocation.
///   2. Caller reduces thread-local buffers into one blob and calls `write_all`
///      once per batch, amortising syscall overhead.
///   3. `header_bytes` / `footer_bytes` are written once, serially.
pub trait Printer: Send + Sync {
    /// Bytes to write before the first batch (e.g. `[` for JSON). Empty by default.
    fn header_bytes(&self) -> Vec<u8> {
        vec![]
    }
    /// Append the formatted representation of `record` into `buf`.
    /// Called in parallel — must be pure (no interior mutability).
    fn write_record(&self, record: &XrefRecord<'_>, buf: &mut Vec<u8>);
    /// Bytes to write after the last batch (e.g. `]\n` for JSON). Empty by default.
    fn footer_bytes(&self) -> Vec<u8> {
        vec![]
    }
}

// ── Text printer ──────────────────────────────────────────────────────────────

pub struct TextPrinter;

impl Printer for TextPrinter {
    fn write_record(&self, r: &XrefRecord<'_>, buf: &mut Vec<u8>) {
        // Fast path: direct byte writes bypass fmt::write / LowerHex::fmt /
        // pad_integral entirely.  ~10× faster than the equivalent writeln!.
        r.from.write_hex_padded(buf);
        write_text_names(r.from_names, buf);
        buf.extend_from_slice(b" -> ");
        r.to.write_hex_padded(buf);
        write_text_names(r.to_names, buf);
        buf.extend_from_slice(b"  ");
        buf.extend_from_slice(r.kind.name().as_bytes());
        buf.extend_from_slice(b"  [");
        buf.extend_from_slice(r.confidence.name().as_bytes());
        buf.extend_from_slice(b"]");
        // Append Rust string if present
        if let Some(s) = &r.rust_string {
            buf.extend_from_slice(b"  \"");
            // Escape for display: replace newlines, etc.
            for c in s.chars() {
                match c {
                    '\n' => buf.extend_from_slice(b"\\n"),
                    '\r' => buf.extend_from_slice(b"\\r"),
                    '\t' => buf.extend_from_slice(b"\\t"),
                    '"' => buf.extend_from_slice(b"\\\""),
                    '\\' => buf.extend_from_slice(b"\\\\"),
                    c if c.is_control() => {
                        // \xNN for other control chars
                        let _ = write!(buf, "\\x{:02x}", c as u32);
                    }
                    c => {
                        let mut tmp = [0u8; 4];
                        buf.extend_from_slice(c.encode_utf8(&mut tmp).as_bytes());
                    }
                }
            }
            buf.extend_from_slice(b"\"");
        }
        buf.push(b'\n');
        if let Some(ctx) = &r.context {
            for line in ctx {
                if line.focus {
                    buf.extend_from_slice(b"  > ");
                } else {
                    buf.extend_from_slice(b"    ");
                }
                line.va.write_hex_padded(buf);
                buf.extend_from_slice(b"  ");
                // Pad hex column to 24 chars
                buf.extend_from_slice(line.hex.as_bytes());
                let pad = 24usize.saturating_sub(line.hex.len());
                buf.resize(buf.len() + pad, b' ');
                buf.extend_from_slice(b"  ");
                buf.extend_from_slice(line.text.as_bytes());
                buf.push(b'\n');
            }
            buf.push(b'\n');
        }
    }
}

// ── JSONL printer ─────────────────────────────────────────────────────────────

/// JSON Lines printer — one compact JSON object per line, no array wrapper.
///
/// Each record is serialised as a single line of JSON followed by `\n`.
/// No commas, no `[`/`]` delimiters. This format is trivially parallel-safe:
/// every `write_record` call is independent — no shared state needed.
pub struct JsonlPrinter;

impl Printer for JsonlPrinter {
    fn write_record(&self, r: &XrefRecord<'_>, buf: &mut Vec<u8>) {
        // XrefRecord contains only simple types (Va, XrefKind, Confidence,
        // Option<Vec<ContextLine>>), so serialisation should never fail.
        // Use `to_writer` to append directly into `buf` without an
        // intermediate String allocation.
        if serde_json::to_writer(&mut *buf, r).is_err() {
            // Defensive: write a placeholder so output isn't silently truncated.
            buf.extend_from_slice(b"{\"error\":\"serialization failed\"}");
        }
        buf.push(b'\n');
    }
}

// ── CSV printer ───────────────────────────────────────────────────────────────

pub struct CsvPrinter;

impl Printer for CsvPrinter {
    fn header_bytes(&self) -> Vec<u8> {
        b"from,to,kind,confidence,string,from_names,to_names\n".to_vec()
    }

    fn write_record(&self, r: &XrefRecord<'_>, buf: &mut Vec<u8>) {
        // Quote the string field to handle commas, quotes, newlines in content.
        let string_field = match &r.rust_string {
            Some(s) => {
                // Escape embedded newlines and carriage returns so a single xref
                // always occupies exactly one CSV row.  Double-quotes are doubled
                // per RFC 4180.
                let escaped = s
                    .replace('"', "\"\"")
                    .replace('\n', "\\n")
                    .replace('\r', "\\r");
                format!("\"{}\"", escaped)
            }
            None => String::new(),
        };
        let _ = write!(
            buf,
            "{:#x},{:#x},{},{},{},",
            r.from,
            r.to,
            r.kind.name(),
            r.confidence.name(),
            string_field,
        );
        write_csv_names(r.from_names, buf);
        buf.push(b',');
        write_csv_names(r.to_names, buf);
        buf.push(b'\n');
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn boxed(names: &[&str]) -> Vec<Box<str>> {
        names.iter().map(|&n| n.into()).collect()
    }

    fn record<'a>(from_names: &'a [Box<str>], to_names: &'a [Box<str>]) -> XrefRecord<'a> {
        XrefRecord {
            from: Va::new(0x1000),
            to: Va::new(0x2000),
            kind: XrefKind::Call,
            confidence: Confidence::PairResolved,
            context: None,
            rust_string: None,
            from_names,
            to_names,
        }
    }

    fn render(p: &dyn Printer, r: &XrefRecord<'_>) -> String {
        let mut buf = Vec::new();
        p.write_record(r, &mut buf);
        String::from_utf8(buf).unwrap()
    }

    #[test]
    fn test_text_without_names() {
        let s = render(&TextPrinter, &record(&[], &[]));
        assert!(!s.contains('<'));
        assert!(s.starts_with("0x"), "{s:?}");
        assert!(s.contains("1000 -> ") && s.contains("2000  call"), "{s:?}");
        assert!(s.ends_with("  call  [pair-resolved]\n"), "{s:?}");
    }

    #[test]
    fn test_text_with_names() {
        let (f, t) = (boxed(&["a", "b"]), boxed(&["c"]));
        let s = render(&TextPrinter, &record(&f, &t));
        assert!(s.contains("1000 <a|b> -> "), "{s:?}");
        assert!(s.contains("2000 <c>  call"), "{s:?}");
        // Only one side named: no stray brackets on the other.
        let s = render(&TextPrinter, &record(&[], &t));
        assert!(s.contains("1000 -> "), "{s:?}");
        assert!(s.contains("2000 <c>  call"), "{s:?}");
    }

    #[test]
    fn test_text_names_escape_control_chars() {
        let f = boxed(&["a\nb"]);
        let s = render(&TextPrinter, &record(&f, &[]));
        assert!(s.contains("<a\\nb>"), "{s:?}");
        assert_eq!(s.matches('\n').count(), 1);
    }

    #[test]
    fn test_jsonl_without_names_omits_fields() {
        let s = render(&JsonlPrinter, &record(&[], &[]));
        assert!(!s.contains("names"), "{s}");
    }

    #[test]
    fn test_jsonl_with_names() {
        let (f, t) = (boxed(&["a", "b"]), boxed(&["c"]));
        let s = render(&JsonlPrinter, &record(&f, &t));
        assert!(s.contains("\"from_names\":[\"a\",\"b\"]"), "{s}");
        assert!(s.contains("\"to_names\":[\"c\"]"), "{s}");
        let s = render(&JsonlPrinter, &record(&[], &t));
        assert!(!s.contains("from_names") && s.contains("to_names"), "{s}");
    }

    #[test]
    fn test_csv_header_and_no_names() {
        assert_eq!(
            CsvPrinter.header_bytes(),
            b"from,to,kind,confidence,string,from_names,to_names\n"
        );
        let s = render(&CsvPrinter, &record(&[], &[]));
        assert_eq!(s, "0x1000,0x2000,call,pair-resolved,,,\n");
    }

    #[test]
    fn test_csv_with_names() {
        let (f, t) = (boxed(&["a", "b"]), boxed(&["c"]));
        let s = render(&CsvPrinter, &record(&f, &t));
        assert_eq!(s, "0x1000,0x2000,call,pair-resolved,,a|b,c\n");
    }

    #[test]
    fn test_csv_names_quoting() {
        let f = boxed(&["a,b", "q\"x"]);
        let t = boxed(&["n\nl"]);
        let mut r = record(&f, &t);
        r.rust_string = Some("s".into());
        let s = render(&CsvPrinter, &r);
        assert_eq!(
            s,
            "0x1000,0x2000,call,pair-resolved,\"s\",\"a,b|q\"\"x\",\"n\\nl\"\n"
        );
    }

    #[test]
    fn test_truncate_middle_no_truncation_needed() {
        assert_eq!(truncate_middle("short", 80), "short");
        assert_eq!(truncate_middle("exactly80chars", 80), "exactly80chars");
    }

    #[test]
    fn test_truncate_middle_zero_means_unlimited() {
        let long = "a".repeat(200);
        assert_eq!(truncate_middle(&long, 0), long);
    }

    #[test]
    fn test_truncate_middle_basic() {
        let s = "abcdefghijklmnopqrstuvwxyz"; // 26 chars
        let result = truncate_middle(s, 10);
        assert_eq!(result.len(), 10);
        assert!(result.contains("..."));
        assert_eq!(result, "abcd...xyz");
    }

    #[test]
    fn test_truncate_middle_path() {
        let path = "/Users/babush/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anstream-0.6.21/src/adapter/strip.rs";
        let result = truncate_middle(path, 80);
        assert_eq!(result.len(), 80);
        assert!(result.starts_with("/Users/babush"));
        assert!(result.ends_with("strip.rs"));
        assert!(result.contains("..."));
    }

    #[test]
    fn test_truncate_middle_very_short_max() {
        // max_width=5 with 10-char string: 1 left + "..." + 1 right
        assert_eq!(truncate_middle("abcdefghij", 5), "a...j");
        // max_width=4: 1 left + "..." + 0 right (available=1, ceil(1/2)=1, floor=0)
        assert_eq!(truncate_middle("abcdefghij", 4), "a...");
        // max_width <= 3: just take first chars (no room for ellipsis + content)
        assert_eq!(truncate_middle("abcdefghij", 3), "abc");
        assert_eq!(truncate_middle("abcdefghij", 1), "a");
    }

    #[test]
    fn test_truncate_middle_unicode() {
        let s = "café résumé naïve"; // contains multi-byte chars
        let result = truncate_middle(s, 12);
        assert!(result.chars().count() <= 12);
        assert!(result.contains("..."));
    }
}
