# Cranelift CPU JIT

cubecl-cpu compiles kernels with a pure-Rust JIT by default. CubeCL's pliron-llvm
IR is lowered straight to Cranelift by `pliron-clif-lower` (from
rustc_codegen_pliron), running our CLIF optimizer and a vendored Cranelift whose
x64 backend has these extensions:

- 256/512-bit vectors (VEX.256 / EVEX.512) and all 32 vector registers (zmm16-31);
- native AVX512-FP16 arithmetic (`vaddph`..`vfmadd213ph`, xmm/ymm/zmm);
- packed F16C conversion (`vcvtph2ps` / `vcvtps2ph`), scalar and vector;
- AVX512-BF16 rounding (`vcvtneps2bf16`).

Nothing has to be enabled. Every feature is picked from CPUID at startup and falls
back to 128-bit SIMD or scalar code on CPUs without it. LLVM LLJIT is still built:
`CUBECL_CPU_JIT=llvm` selects it, and so does building without the `cranelift`
feature.

## Performance vs LLVM LLJIT

Xeon Platinum 8559C, release build. Throughput is `examples/throughput all`, 3
alternating runs with `~/.cache/cubecl` and `target/environment` cleared before each.
Each cell is best (worst).

| | Cranelift | LLVM |
|---|---|---|
| compute-direct f32 | 2.03 (1.41) TOPS | 2.03 (1.42) TOPS |
| compute-direct f16 | 4.00 (3.38) TOPS | 4.06 (4.03) TOPS |
| compute-direct bf16 | 2.02 (2.00) TOPS | 2.02 (1.75) TOPS |
| bf16-operand kernels ¹ | 95-105 GOPS | 44-45 GOPS |
| memory copy 1 GiB | 98.2 (91.4) GB/s | 100.8 (99.1) GB/s |
| memory read 512 MiB | 77.4 (75.4) GB/s | 79.1 (77.1) GB/s |
| memory write 512 MiB | 92.3 (87.6) GB/s | 91.3 (89.8) GB/s |
| launch | 3.68 (4.04) µs | 3.57 (3.77) µs |
| compile, whole test suite (1432 kernels, 1 thread) | 5.58 s | 11.16 s |
| compile, median / p90 / max per kernel | 2.7 / 6.3 / 110 ms | 7.1 / 9.5 / 356 ms |
| `cargo test --release -p cubecl-cpu` | 1039 pass, 0 fail | 1039 pass, 0 fail |

Compute and memory are within run-to-run noise of LLVM: the slow outliers show up
in both columns. Compiles are about 2x faster.

¹ The headline bf16 row runs on an f32-accumulator kernel, so it measures f32. The
kernels whose operands really are bf16 only show up in the probe's per-width output
(best of 8 widths). Those numbers come from a separate, slower session, so compare
the two columns, not other rows.

## Semantics differences from LLVM

- bf16 FMA is one fused f32 FMA with a single rounding. LLVM multiplies, rounds,
  adds and rounds again. Ours is more precise, but the bits differ.
- On AVX512-BF16 CPUs, `vcvtneps2bf16` flushes subnormal f32 inputs to zero
  (`1e-40` -> `0000`). LLVM uses the same instruction. The fallback rounding keeps
  them (`0001`). Normals, ties, inf and NaN are bit-identical on both paths.

## Opt-outs (for A/B and bisecting)

| env | effect |
|---|---|
| `CUBECL_CPU_JIT=llvm` | LLVM LLJIT |
| `CUBECL_CLIF_WIDE=0\|256\|512` | widest f32 vector (default: widest the CPU has) |
| `CUBECL_CLIF_SIMD=0` | split vectors per lane |
| `CUBECL_CLIF_FP16=0` | f16 math via f32 |
| `CUBECL_CLIF_F16C=0` | f16 casts via compiler-builtins libcalls |
| `CUBECL_CLIF_BF16=0` | bf16 rounding via integer ops, 128-bit |
| `PLIRON_NO_HIGH_XMM=1` | only xmm0-15 |
| `PLIRON_POSTINC_NE=0` | no latch rewrite (compare after increment) |

## Opt-in

`CUBECL_CLIF_IROPT=1` runs rustc_codegen_pliron's pliron-IR passes (instcombine,
memcpyopt, sroa, phisimp, constload) before lowering. They are off because they
show no runtime gain (f32/f16/bf16 2.02 / 4.03 / 2.03 TOPS, best of 3, in the runs
above) and they cost about 35% more compile time. CubeCL's own optimizer has already
simplified the IR by then.

## Debugging

`CUBECL_JIT_TIME=1` prints per-kernel compile µs. `CUBECL_CLIF_DUMP=1` dumps the
pliron IR. `PLIRON_CLIF=<substr>` / `PLIRON_VCODE=<substr>` dump CLIF and machine
code. `PLIRON_TIMING=1` prints pass times. `CUBECL_DEBUG_PLIRON=<dir>` writes the
LLVM IR before and after O3.

## Windows x64 notes

- `msvc_stl_compat.cpp` (compiled by build.rs on `target_env = "msvc"`)
  supplies scalar `__std_*` vectorized-algorithm exports the bundled tracel
  LLVM needs but VS2019's msvcprt lacks (plus `_Thrd_sleep_for`,
  `__std_init_once_link_alternate_names_and_abort`).
- `OrcJit` registers `__truncsfbf2`/`__extendhfsf2`/`__truncdfsf2` as
  absolute symbols — the UCRT doesn't export the bf16 builtins.
- Call relocs beyond ±2GB (ucrtbase.dll, exe functions vs. VirtualAlloc'd
  JIT memory) rely on the x64 call veneers in vendored `cranelift-jit`
  (`jmp [rip]; dq addr`), patch at rustc_codegen_pliron/vendor/cranelift-jit.
- The Cranelift JIT binds libm names (`sinf`, `powf`, `fmax`, `hypotf`,
  `expm1`, `roundeven`, `fmodf`, ...) to Rust `std` shims so nothing
  depends on host CRT exports.
- Known failure: `test_debug_print` — kernels calling variadic `printf`
  need Win64 varargs float args duplicated into GPR slots, which Cranelift
  cannot express; the call is redirected to `__pliron_va8.printf` (a
  Linux-only thunk) and the JIT rejects the pending asm. A
  cranelift Win64 variadic-duplication feature or a fixed-arity printf
  shim would fix it.
- `tests::*::test_powi_constants` fails identically under
  `CUBECL_CPU_JIT=llvm` — pre-existing, not a Cranelift bug.
- i7-12700K (8P+4E): workers are ordered by `EfficiencyClass` so P-core
  primaries come first (see cubecl-cpu affinity). Unordered dispatch on
  the hybrid part produced 0.17-1.13 TOPS swings; ordered, f32 stays
  ~0.75-0.95 TOPS across runs.
