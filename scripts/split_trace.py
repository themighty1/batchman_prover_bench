#!/usr/bin/env python3
"""Split cpu_trace.bin and packed_pt.bin into N segments."""

import sys
import struct
import os
import math

def split_file(path, row_size, num_segments, output_dirs):
    """Split a binary file with a 4-byte LE u32 header + row_size-byte rows."""
    with open(path, 'rb') as f:
        count = struct.unpack('<I', f.read(4))[0]
        data = f.read()

    assert len(data) == count * row_size, f"{path}: expected {count * row_size} bytes, got {len(data)}"

    rows_per_seg = math.ceil(count / num_segments)
    for k in range(num_segments):
        start = k * rows_per_seg
        end = min((k + 1) * rows_per_seg, count)
        seg_count = end - start
        seg_data = data[start * row_size : end * row_size]

        out_path = os.path.join(output_dirs[k], os.path.basename(path))
        with open(out_path, 'wb') as f:
            f.write(struct.pack('<I', seg_count))
            f.write(seg_data)

    return count, rows_per_seg


def main():
    if len(sys.argv) < 4:
        print("Usage: split_trace.py <witness_dir> <num_segments> <output_base>")
        print("  witness_dir:  e.g. witgen/witness/json-query")
        print("  num_segments: e.g. 4")
        print("  output_base:  e.g. witgen/witness  (creates <base>/<program>__seg0/ etc)")
        sys.exit(1)

    witness_dir = sys.argv[1]
    num_segments = int(sys.argv[2])
    output_base = sys.argv[3]

    program = os.path.basename(witness_dir)
    cpu_trace = os.path.join(witness_dir, 'cpu_trace.bin')
    packed_pt = os.path.join(witness_dir, 'packed_pt.bin')

    # Create output directories
    output_dirs = []
    for k in range(num_segments):
        d = os.path.join(output_base, f"{program}__seg{k}")
        os.makedirs(d, exist_ok=True)
        output_dirs.append(d)

    # Split cpu_trace.bin (28 bytes per row)
    count, rows_per_seg = split_file(cpu_trace, 28, num_segments, output_dirs)

    # Split packed_pt.bin (16 bytes per row) if it exists
    if os.path.exists(packed_pt):
        split_file(packed_pt, 16, num_segments, output_dirs)

    # Print summary
    for k in range(num_segments):
        start = k * rows_per_seg
        end = min((k + 1) * rows_per_seg, count)
        print(f"  segment {k}: rows {start}-{end-1} ({end - start} steps) -> {output_dirs[k]}")
    print(f"  total: {count} steps in {num_segments} segments")


if __name__ == '__main__':
    main()
