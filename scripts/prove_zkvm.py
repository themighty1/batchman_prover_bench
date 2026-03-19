#!/usr/bin/env python3
"""
zkVM proof orchestrator.

Launches Batchman with segmented proving and monitors segment completion.
As each segment finishes, reports it immediately.
"""

import os
import sys
import struct
import subprocess
import time
import tempfile
import shutil
import signal

SEGMENT_SIZE = 10_000
CONCURRENCY = 2

def find_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        root = find_root()
        guest_dir = os.path.join(root, 'minimal_isa', 'guest-programs')
        print("zkVM proof orchestrator.")
        print()
        print("Usage: prove_zkvm.py <program>")
        print()
        print("Available programs:")
        for name in sorted(os.listdir(guest_dir)):
            if os.path.isdir(os.path.join(guest_dir, name)):
                print(f"  prove_zkvm.py {name}")
        sys.exit(0)

    program = sys.argv[1]
    root = find_root()

    witness_dir = os.path.join(root, 'witgen', 'witness', program)
    cpu_trace = os.path.join(witness_dir, 'cpu_trace.bin')
    binary = os.path.join(root, 'build', 'bin', 'test_bench_batchman_bool')
    circuit_dir = os.path.join(root, 'circuits', 'generated')

    # Check witness exists
    if not os.path.exists(cpu_trace):
        print(f"Witness not found, generating...")
        subprocess.run([os.path.join(root, 'scripts', 'witgen.sh'), program], check=True)

    # Read step count
    with open(cpu_trace, 'rb') as f:
        total_steps = struct.unpack('<I', f.read(4))[0]

    num_segments = (total_steps + SEGMENT_SIZE - 1) // SEGMENT_SIZE
    print(f"=== prove_zkvm: {program} ===")
    print(f"  {total_steps} steps, {num_segments} segments of {SEGMENT_SIZE}, concurrency {CONCURRENCY}")
    print()

    # Create tmp dir for segment outputs
    seg_out = tempfile.mkdtemp(prefix='zkvm_segments_')

    # Build if needed
    if not os.path.exists(binary):
        print("Building Batchman binary...")
        build_dir = os.path.join(root, 'build')
        os.makedirs(build_dir, exist_ok=True)
        subprocess.run(['cmake', '..'], cwd=build_dir, capture_output=True)
        subprocess.run(['make', '-j4', 'test_bench_batchman_bool'], cwd=build_dir, capture_output=True)

    if not os.path.exists(os.path.join(circuit_dir, 'add.bin')):
        print("Generating circuit files...")
        subprocess.run(['cargo', 'run', '--release'], cwd=os.path.join(root, 'circuits'), capture_output=True)

    # Build binius binaries
    binius_dir = os.path.join(root, 'binius')
    binius_bin = os.path.join(binius_dir, 'target', 'release')
    print("  Building binius binaries...")
    subprocess.run(['cargo', 'build', '--release'], cwd=binius_dir, capture_output=True)

    # Launch Batchman prover + verifier
    env = os.environ.copy()
    env['SEGMENT_SIZE'] = str(SEGMENT_SIZE)
    env['CONCURRENCY'] = str(CONCURRENCY)
    env['SEGMENT_OUT_DIR'] = seg_out

    port = 20100
    steps = str(total_steps)

    procs = []

    # Verifier
    v_log = os.path.join(seg_out, 'verifier.log')
    with open(v_log, 'w') as vf:
        v = subprocess.Popen(
            [binary, '2', str(port), '127.0.0.1', steps, circuit_dir, cpu_trace],
            stdout=vf, stderr=subprocess.STDOUT, env=env)
        procs.append(v)

    time.sleep(0.3)

    # Prover
    p_log = os.path.join(seg_out, 'prover.log')
    with open(p_log, 'w') as pf:
        p = subprocess.Popen(
            [binary, '1', str(port), '127.0.0.1', steps, circuit_dir, cpu_trace],
            stdout=pf, stderr=subprocess.STDOUT, env=env)
        procs.append(p)

    # Memory check (runs in parallel with Batchman, independent of segments)
    mem_prover_log = os.path.join(seg_out, 'memory_check_prover.log')
    mem_verifier_log = os.path.join(seg_out, 'memory_check_verifier.log')
    mem_env = {'RUST_MIN_STACK': '67108864', **os.environ}

    def run_memory_check():
        with open(mem_prover_log, 'w') as f:
            mp = subprocess.Popen(
                [os.path.join(binius_bin, 'memory_check_prover'), program],
                stdout=f, stderr=subprocess.STDOUT, env=mem_env, cwd=binius_dir)
            procs.append(mp)
            mp.wait()
        with open(mem_verifier_log, 'w') as f:
            mv = subprocess.Popen(
                [os.path.join(binius_bin, 'memory_check_verifier'), program],
                stdout=f, stderr=subprocess.STDOUT, env=mem_env, cwd=binius_dir)
            procs.append(mv)
            mv.wait()
            return mv.returncode == 0

    import threading
    mem_result = [None]
    def mem_thread():
        mem_result[0] = run_memory_check()
    mt = threading.Thread(target=mem_thread)
    mt.start()

    # Cleanup on exit
    def cleanup(*_):
        for proc in procs:
            try: proc.kill()
            except: pass

    signal.signal(signal.SIGINT, lambda *_: (cleanup(), sys.exit(1)))
    signal.signal(signal.SIGTERM, lambda *_: (cleanup(), sys.exit(1)))

    # Monitor segment completion
    start = time.time()
    seen = set()
    cpu_done = False

    while not cpu_done:
        # Check for new segment.done markers
        for seg in range(num_segments):
            if seg in seen:
                continue
            marker = os.path.join(seg_out, f'seg_{seg}', 'segment.done')
            if os.path.exists(marker):
                seen.add(seg)
                elapsed = time.time() - start
                seg_start = seg * SEGMENT_SIZE
                seg_end = min(seg_start + SEGMENT_SIZE, total_steps)
                print(f"  segment {seg:>2}: {seg_end-seg_start:>5} steps  done at {elapsed:.1f}s")

        # Check if batchman finished
        if p.poll() is not None and v.poll() is not None:
            cpu_done = True
        else:
            time.sleep(0.1)

    cpu_wall = time.time() - start
    p_ok = p.returncode == 0
    v_ok = v.returncode == 0

    if p_ok and v_ok:
        print(f"  CPU proof: {num_segments} segments in {cpu_wall:.1f}s")
    else:
        print(f"  CPU proof FAILED (prover={p.returncode}, verifier={v.returncode})")

    # Wait for memory check
    mt.join()
    mem_wall = time.time() - start

    if mem_result[0]:
        print(f"  Memory check: PASSED ({mem_wall:.1f}s)")
    else:
        print(f"  Memory check: FAILED (see {mem_prover_log})")

    wall = time.time() - start
    print()
    print(f"  Wall time: {wall:.1f}s")

    if not (p_ok and v_ok and mem_result[0]):
        sys.exit(1)


if __name__ == '__main__':
    main()
