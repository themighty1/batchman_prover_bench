#![no_std]

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use serde_json::Value;

/// Entry point: parse JSON, resolve dot-notation path, return result string.
/// Empty string on any error.
pub fn run(json_bytes: &[u8], path_bytes: &[u8]) -> String {
    let json_str = match core::str::from_utf8(json_bytes) {
        Ok(s) => s,
        Err(_) => return String::new(),
    };
    let path_str = match core::str::from_utf8(path_bytes) {
        Ok(s) => s,
        Err(_) => return String::new(),
    };
    match serde_json::from_str::<Value>(json_str) {
        Ok(v) => match resolve_path(&v, path_str) {
            Some(found) => value_to_output(found),
            None => String::new(),
        },
        Err(_) => String::new(),
    }
}

fn resolve_path<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    let mut current = value;
    for segment in path.split('.') {
        if segment.is_empty() {
            continue;
        }
        match current {
            Value::Object(obj) => {
                current = obj.get(segment)?;
            }
            Value::Array(arr) => {
                let idx: usize = parse_usize(segment)?;
                current = arr.get(idx)?;
            }
            _ => return None,
        }
    }
    Some(current)
}

fn parse_usize(s: &str) -> Option<usize> {
    let mut result: usize = 0;
    for &b in s.as_bytes() {
        if b < b'0' || b > b'9' {
            return None;
        }
        result = result.checked_mul(10)?.checked_add((b - b'0') as usize)?;
    }
    Some(result)
}

fn value_to_output(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => String::from("null"),
        Value::Bool(true) => String::from("true"),
        Value::Bool(false) => String::from("false"),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                format_i64(i)
            } else if let Some(u) = n.as_u64() {
                format_u64(u)
            } else {
                serde_json::to_string(value).unwrap_or_else(|_| String::from("null"))
            }
        }
        _ => serde_json::to_string(value).unwrap_or_else(|_| String::from("null")),
    }
}

fn format_i64(mut n: i64) -> String {
    if n == 0 {
        return String::from("0");
    }
    let negative = n < 0;
    if negative {
        n = -n;
    }
    let mut buf = Vec::new();
    while n > 0 {
        buf.push(b'0' + (n % 10) as u8);
        n /= 10;
    }
    if negative {
        buf.push(b'-');
    }
    buf.reverse();
    unsafe { String::from_utf8_unchecked(buf) }
}

fn format_u64(mut n: u64) -> String {
    if n == 0 {
        return String::from("0");
    }
    let mut buf = Vec::new();
    while n > 0 {
        buf.push(b'0' + (n % 10) as u8);
        n /= 10;
    }
    buf.reverse();
    unsafe { String::from_utf8_unchecked(buf) }
}
