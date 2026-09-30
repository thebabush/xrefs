pub mod arch;
pub mod disasm;
pub mod loader;
pub mod names;
pub mod output;
pub mod pass;
pub mod rust;
pub(crate) mod shard;
pub mod va;
pub mod xref;

pub use loader::{Arch, Arm32Segment, DecodeMode, LoadedBinary, ModeSwitch, RelocPointer,
                 Segment, SegmentArch, Symbol};
pub use names::{NameTable, NameTableBuilder};
pub use pass::{Depth, PassConfig, PassResult, XrefPass};
pub use rust::{StringBlobIndex, DEFAULT_MIN_BLOB_LEN};
pub use va::{Va, VaRange};
pub use xref::{Confidence, Xref, XrefKind};

/// Parse a virtual address from a string, returning the raw `u64` value.
///
/// Accepts `0x`/`0X`-prefixed hexadecimal or plain decimal.
/// Prefer [`Va::parse`] when a [`Va`] is needed directly.
pub fn parse_va(s: &str) -> Result<u64, String> {
    Va::parse(s).map(Va::raw)
}

#[cfg(test)]
mod tests {
    use super::parse_va;

    #[test]
    fn test_parse_va_decimal() {
        assert_eq!(parse_va("0"), Ok(0));
        assert_eq!(parse_va("1024"), Ok(1024));
        assert_eq!(parse_va("4194304"), Ok(0x400000));
    }

    #[test]
    fn test_parse_va_hex_lowercase() {
        assert_eq!(parse_va("0x0"), Ok(0));
        assert_eq!(parse_va("0x400000"), Ok(0x400000));
        assert_eq!(parse_va("0xdeadbeef"), Ok(0xdeadbeef));
        assert_eq!(parse_va("0xffffffffffffffff"), Ok(u64::MAX));
    }

    #[test]
    fn test_parse_va_hex_uppercase_prefix() {
        assert_eq!(parse_va("0X1000"), Ok(0x1000));
        assert_eq!(parse_va("0XDEADBEEF"), Ok(0xdeadbeef));
    }

    #[test]
    fn test_parse_va_hex_mixed_case_digits() {
        assert_eq!(parse_va("0xDeAdBeEf"), Ok(0xdeadbeef));
    }

    #[test]
    fn test_parse_va_invalid_decimal() {
        assert!(parse_va("abc").is_err());
        assert!(parse_va("").is_err());
        assert!(parse_va("-1").is_err());
    }

    #[test]
    fn test_parse_va_invalid_hex() {
        assert!(parse_va("0xGHIJ").is_err());
        assert!(parse_va("0x").is_err()); // empty after prefix
    }

    #[test]
    fn test_parse_va_no_0x_prefix_not_treated_as_hex() {
        // "ff" is not decimal — should fail
        assert!(parse_va("ff").is_err());
    }
}
