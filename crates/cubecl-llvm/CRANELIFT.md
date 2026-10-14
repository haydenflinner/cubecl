# Cranelift CPU JIT (`cranelift` feature) vs LLVM LLJIT

Cranelift is the default when the feature is on; `CUBECL_CPU_JIT=llvm` selects LLJIT.
`CUBECL_JIT_TIME=1` prints `cubecl-jit-time <jit> <kernel> <µs>` per compile.
Lowering: `pliron-clif-lower` (from rustc_codegen_pliron). Vectors lower as 128-bit SIMD parts
(`WIDE_SIMD`); `CUBECL_CLIF_SIMD=0` scalarizes per lane instead (the "scalar" column).

Host: Xeon Platinum 8559C (AVX-512, AVX512-FP16), release build.

## Correctness

`cargo test --release -p cubecl-cpu --features cranelift`: 1039 passed, 0 failed, 20 ignored on both JITs.

## Compile latency (same test suite, `--test-threads=1 --nocapture`)

| | Cranelift | LLVM |
|---|---|---|
| kernels compiled | 1418 | 1418 |
| total compile time | 7.05 s | 15.87 s |
| median per kernel | 2.99 ms | 8.66 ms |
| median ratio clif/llvm (p10 / p90) | 0.36 (0.17 / 0.97) | |
| kernels slower on Cranelift | 137 / 1416 | |
| worst: `transcendentals` n=8 | 137 ms | 15 ms |
| throughput `compute_direct` bf16 n=128 | 2685 ms | 73 ms |
| throughput `compute_direct` f16 n=128 | 285 ms | 44 ms |

The slow cases are wide vectors: per-lane scalarization (plus bf16 promotion per lane) grows the function linearly in lanes.

## Kernel runtime (`examples/throughput` `all`, best of 3, persistence cache cleared per run)

| probe | Cranelift | LLVM | clif/llvm |
|---|---|---|---|
| compute-direct f32 | 31.1 GOPS | 1079 GOPS | 0.03 |
| compute-direct f16 | 31.8 GOPS | 2056 GOPS | 0.02 |
| compute-direct bf16 | 32.6 GOPS | 1074 GOPS | 0.03 |
| memory copy 1 GiB | 43.8 GB/s | 58.2 GB/s | 0.75 |
| memory read 512 MiB | 25.1 GB/s | 39.8 GB/s | 0.63 |
| memory write 512 MiB | 49.0 GB/s | 52.8 GB/s | 0.93 |
| launch (noisy: 4–35 µs both) | 10.6 µs | 4.5 µs | |

Compute is ~30x behind because every vector op becomes scalar lane ops; LLVM uses
512-bit FMA. Cranelift's x86 backend tops out at 128-bit SIMD, so keeping `f32x4`
chunks native is the next step; full parity needs wider vectors than Cranelift has.

## With native SIMD (`WIDE_SIMD`, 128-bit parts; 244f680 + backend 33114dd)

Same harness, best of 3. f16/bf16 lines are still promoted per lane.

| | Cranelift scalar | Cranelift SIMD | LLVM |
|---|---|---|---|
| compute-direct f32 | 31.1 GOPS | 295 GOPS | 1079 GOPS |
| compute-direct f16 | 31.8 GOPS | 266 GOPS | 2056 GOPS |
| compute-direct bf16 | 32.6 GOPS | 271 GOPS | 1074 GOPS |
| memory copy | 43.8 GB/s | 58.2 GB/s | 58.2 GB/s |
| memory read | 25.1 GB/s | 38.7 GB/s | 39.8 GB/s |
| memory write | 49.0 GB/s | 47.2 GB/s | 52.8 GB/s |
| test-suite compile total | 7.05 s | 6.20 s | 15.87 s |
| median per kernel (ratio clif/llvm, p90) | 2.99 ms (0.36, 0.97) | 2.79 ms (0.35, 0.73) | 8.66 ms |
| `compute_direct` bf16 n=128 compile | 2685 ms | 2847 ms | 73 ms |

64-bit vectors (`f32x2`) stay scalarized: native x64 lowering of them failed `test_fma_from_sub`/`test_trunc`.

## deadrec fix (backend: `self_feeding` used a whole-function scan per value)

`PLIRON_TIMING=1` showed `deadrec` at 2124 ms of the 2425 ms pipeline for bf16 n=128.
With one `Uses` index per pass: deadrec 4.6 ms, `compute_direct` bf16 n=128 compile
2847 ms -> 570 ms, f16 n=128 289 ms -> 252 ms (single run). Suite: 1039 passed.

## Rejected: whole-vector bf16 promotion

Promoting `<n x bf16>` arith to `<n x f32>` before scalarizing (so FMA runs as
`fma.f32x4`; verified in the CLIF dump) passed the suite but did not pay off.
`compute_direct`, best of 3:

| | bf16 GOPS | bf16 n=128 compile |
|---|---|---|
| per-lane promotion (kept) | 293 | 537 ms |
| whole-vector promotion | 298 | 778 ms |

The per-lane bf16<->f32 casts (extract/insert + bit ops) dominate. A win needs bf16
vectors carried as `i16x8` leaves so the casts are SIMD too (`uwiden` + `ishl 16`).

## bf16 vectors as `i16x8` leaves (SIMD casts)

128-bit bf16 vectors are `i16x8` bit patterns; casts to/from `<n x f32>` are
`uwiden`+`ishl` / RNE shift+`unarrow`, and the whole-vector promotion above now
pays off. `compute_direct`, 3 runs each:

| | bf16 GOPS (runs) | bf16 n=128 compile (runs) |
|---|---|---|
| per-lane casts | 248 / 172 / 166 | 573 / 600 / 570 ms |
| `i16x8` + SIMD casts | 262 / 168 / 189 | 157 / 180 / 157 ms |

Runtime is within noise on this box (the untouched f32 probe swung 169-266 GOPS
in the same runs); the compile-time win is 3.6x. Suite: 1039 passed.
