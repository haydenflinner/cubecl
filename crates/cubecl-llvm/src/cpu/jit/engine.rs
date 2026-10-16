use cubecl_runtime::kernel::BufferIOAttr;
use std::ffi::c_void;
use std::fmt::Display;
use std::sync::{Arc, Once};

use pliron::builtin::ops::ModuleOp;
use pliron::context::Context;
use pliron_llvm::llvm_sys::core::{LLVMContext, LLVMMemoryBuffer, LLVMModule};
use pliron_llvm::llvm_sys::lljit::LLVMLLJIT;
use pliron_llvm::llvm_sys::target::initialize_native;
use pliron_llvm::to_llvm_ir;

use super::data::PlironData;
use crate::cpu::shared_memory::SharedMemories;

/// Host ABI of a JIT'd kernel: `(buffer_ptrs, cube_count_x/y/z, unit_pos_x/y/z, sync_cube_state,
/// metadata)`. The variable-count pointers — the buffers and then the shared memories — are
/// hidden behind `buffer_ptrs`, while the builtins and both other pointers are passed directly.
type KernelFn = extern "C" fn(*mut *mut c_void, u32, u32, u32, u32, u32, u32, *mut u32, *mut u64);

/// What the host has to provide to launch a kernel, beyond its arguments.
#[derive(Clone, Debug, Default)]
pub struct KernelRequirements {
    /// Whether the kernel synchronizes its cube, in which case each of its units needs a thread
    /// of its own to run on.
    pub needs_parallelism: bool,
    /// The shared memory to reserve for a launch, and where its pointers go.
    pub shared_memories: SharedMemories,
}

/// A JIT-compiled kernel.
///
/// The `LLVMContext` the module was built in is not held here: `LLVMLLJIT::add_module`
/// takes it by value and transfers it to the JIT's thread-safe context, so the JIT is
/// what keeps it alive for as long as the compiled code exists.
#[repr(C)]
struct JitKernel {
    func: KernelFn,
    requirements: KernelRequirements,
    /// What the kernel does with each buffer binding, by buffer position --
    /// read off the IR before the entry ABI lowering erased the arguments,
    /// for the launch path's taint bookkeeping.
    io: Vec<BufferIOAttr>,
    _jit: Jit,
}

/// Owner of the compiled code; dropping it would unmap the kernel.
enum Jit {
    Llvm(#[allow(dead_code)] LLVMLLJIT),
    #[cfg(feature = "cranelift")]
    Cranelift(#[allow(dead_code)] cranelift_jit::JITModule),
}

/// Safety: The kernel is immutable machine code plus the JIT/context that keep it alive.
unsafe impl Send for JitKernel {}
unsafe impl Sync for JitKernel {}

/// A compiled kernel, cloneable across worker threads.
#[derive(Clone)]
pub struct PlironEngine(Arc<JitKernel>);

static INIT_NATIVE: Once = Once::new();

impl PlironEngine {

    /// `CUBECL_JIT_TIME=1` prints `cubecl-jit-time <jit> <kernel> <µs>` per
    /// compile (pliron-llvm module in, callable kernel out).
    pub fn compile(
        ctx: &mut Context,
        module: ModuleOp,
        kernel_name: &str,
        requirements: KernelRequirements,
        io: Vec<BufferIOAttr>,
    ) -> pliron::result::Result<Self> {
        let t = std::time::Instant::now();
        let r = Self::compile_inner(ctx, module, kernel_name, requirements, io);
        static TIME: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *TIME.get_or_init(|| std::env::var_os("CUBECL_JIT_TIME").is_some()) {
            #[cfg(feature = "cranelift")]
            let jit = if use_cranelift() { "cranelift" } else { "llvm" };
            #[cfg(not(feature = "cranelift"))]
            let jit = "llvm";
            eprintln!("cubecl-jit-time {jit} {kernel_name} {}", t.elapsed().as_micros());
        }
        r
    }

    fn compile_inner(
        ctx: &mut Context,
        module: ModuleOp,
        kernel_name: &str,
        requirements: KernelRequirements,
        io: Vec<BufferIOAttr>,
    ) -> pliron::result::Result<Self> {
        #[cfg(feature = "cranelift")]
        if use_cranelift() {
            let (jit, addr) = cranelift::compile(ctx, module, kernel_name);
            // SAFETY: The generated entry point matches `KernelFn`.
            let func: KernelFn = unsafe { std::mem::transmute::<*const u8, KernelFn>(addr) };
            return Ok(PlironEngine(Arc::new(JitKernel {
                func,
                requirements,
                io,
                _jit: Jit::Cranelift(jit),
            })));
        }
        INIT_NATIVE.call_once(|| {
            initialize_native().expect("failed to initialize native target");
        });

        let llvm_ctx = LLVMContext::default();
        let llvm_module = to_llvm_ir::convert_module(ctx, &llvm_ctx, module)?;
        #[cfg(feature = "pliron-dump")]
        if let Some(dir) = ir_dump_path(kernel_name) {
            let _ = std::fs::write(dir.join("llvm.ll"), llvm_module.to_string());
        }

        let llvm_module = optimize(llvm_module, &llvm_ctx, kernel_name)
            .unwrap_or_else(|err| panic!("LLVM optimization failed for '{kernel_name}': {err}"));
        #[cfg(feature = "pliron-dump")]
        if let Some(dir) = ir_dump_path(kernel_name) {
            let _ = std::fs::write(dir.join("llvm.opt.ll"), llvm_module.to_string());
        }

        let lljit = LLVMLLJIT::new_with_default_builder().expect("failed to create LLJIT");
        // Consumes the context: `optimize` re-parsed the module into `llvm_ctx`, which is
        // what `add_module` asserts, and ownership passes to the JIT from here.
        lljit
            .add_module(llvm_ctx, llvm_module)
            .expect("failed to add module to JIT");
        let addr = lljit
            .lookup_symbol(kernel_name)
            .unwrap_or_else(|err| panic!("kernel symbol '{kernel_name}' not found: {err}"));
        // Safety: the generated function is always of this form
        let func: KernelFn = unsafe { std::mem::transmute::<u64, KernelFn>(addr) };

        Ok(PlironEngine(Arc::new(JitKernel {
            func,
            requirements,
            io,
            _jit: Jit::Llvm(lljit),
        })))
    }

    /// What the host has to provide to launch this kernel, see [`KernelRequirements`].
    pub fn requirements(&self) -> &KernelRequirements {
        &self.0.requirements
    }

    /// What the kernel does with each buffer binding, by buffer position.
    pub fn buffer_io(&self) -> &[BufferIOAttr] {
        &self.0.io
    }

    pub fn run_kernel(&self, data: &mut PlironData) {
        let b = data.builtins;
        let buffer_ptrs = data.shared.buffer_ptrs.as_ptr() as *mut *mut c_void;
        let metadata = data.shared.metadata.as_ptr() as *mut u64;
        let sync_cube_state = data.shared.sync_cube_state.as_ptr() as *mut u32;
        (self.0.func)(
            buffer_ptrs,
            b[0],
            b[1],
            b[2],
            b[3],
            b[4],
            b[5],
            sync_cube_state,
            metadata,
        );
    }
}

impl Display for PlironEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Pliron JIT engine")
    }
}

#[cfg(feature = "pliron-dump")]
/// The kernel's dump directory when `CUBECL_DEBUG_PLIRON` is set: the LLVM IR
/// stages land beside the pliron pass dumps.
pub(crate) fn ir_dump_path(kernel_name: &str) -> Option<std::path::PathBuf> {
    let dir = std::env::var("CUBECL_DEBUG_PLIRON").ok()?;
    let path = std::path::Path::new(&dir).join(kernel_name);
    std::fs::create_dir_all(&path).ok()?;
    Some(path)
}

/// The pipeline run before the JIT: LLJIT runs no IR-level passes of its own,
/// and the dialect lowering emits O0-shaped IR that runs ~50× under the
/// machine's streaming rate. Paid once per kernel and cached.
const PASS_PIPELINE: &str = "default<O3>";

/// Optimizes a module by round-tripping it through textual IR: [`LLVMModule`]
/// seals its `LLVMModuleRef`, so the IR is re-parsed into a context this
/// function owns, optimized there, and parsed back. Fold into a direct pass
/// run when pliron-llvm exposes one.
fn optimize(
    module: LLVMModule,
    llvm_ctx: &LLVMContext,
    kernel_name: &str,
) -> Result<LLVMModule, String> {
    let optimized = run_pipeline(&module.to_string())?;
    drop(module);
    LLVMModule::from_ir_in_memory_buffer(
        llvm_ctx,
        LLVMMemoryBuffer::from_str(&optimized, kernel_name),
    )
}

/// Parses `ir` into a private LLVM context, runs [`PASS_PIPELINE`] over it,
/// and prints the optimized module back out.
fn run_pipeline(ir: &str) -> Result<String, String> {
    use llvm_sys::core::{
        LLVMContextCreate, LLVMContextDispose, LLVMCreateMemoryBufferWithMemoryRangeCopy,
        LLVMDisposeMessage, LLVMDisposeModule, LLVMPrintModuleToString,
    };
    use llvm_sys::error::{LLVMDisposeErrorMessage, LLVMGetErrorMessage};
    use llvm_sys::ir_reader::LLVMParseIRInContext2;
    use llvm_sys::transforms::pass_builder::{
        LLVMCreatePassBuilderOptions, LLVMDisposePassBuilderOptions, LLVMRunPasses,
    };

    unsafe {
        let ctx = LLVMContextCreate();
        let buffer = LLVMCreateMemoryBufferWithMemoryRangeCopy(
            ir.as_ptr() as *const _,
            ir.len(),
            c"kernel".as_ptr(),
        );
        let mut module = std::ptr::null_mut();
        let mut parse_err = std::ptr::null_mut();
        // `LLVMParseIRInContext2` consumes the buffer, on failure included.
        if LLVMParseIRInContext2(ctx, buffer, &mut module, &mut parse_err) != 0 {
            let msg = std::ffi::CStr::from_ptr(parse_err)
                .to_string_lossy()
                .into_owned();
            LLVMDisposeMessage(parse_err);
            LLVMContextDispose(ctx);
            return Err(msg);
        }

        let passes = std::ffi::CString::new(PASS_PIPELINE).expect("static pass string");
        let options = LLVMCreatePassBuilderOptions();
        let err = LLVMRunPasses(module, passes.as_ptr(), std::ptr::null_mut(), options);
        LLVMDisposePassBuilderOptions(options);

        let result = if err.is_null() {
            let c_ir = LLVMPrintModuleToString(module);
            let optimized = std::ffi::CStr::from_ptr(c_ir)
                .to_string_lossy()
                .into_owned();
            LLVMDisposeMessage(c_ir);
            Ok(optimized)
        } else {
            let c_msg = LLVMGetErrorMessage(err);
            let msg = std::ffi::CStr::from_ptr(c_msg)
                .to_string_lossy()
                .into_owned();
            LLVMDisposeErrorMessage(c_msg);
            Err(msg)
        };
        LLVMDisposeModule(module);
        LLVMContextDispose(ctx);
        result
    }
}

/// Vector registers the active JIT allocates, when fewer than the ISA has:
/// Cranelift x64 allocates zmm16-31 only in 512-bit wide mode.
pub fn vector_register_cap() -> Option<u32> {
    #[cfg(feature = "cranelift")]
    if use_cranelift() && (wide_bits() < 512 || std::env::var_os("PLIRON_NO_HIGH_XMM").is_some()) {
        return Some(16);
    }
    None
}

/// `CUBECL_CPU_JIT=llvm` switches a `cranelift` build back to LLJIT.
#[cfg(feature = "cranelift")]
fn use_cranelift() -> bool {
    static USE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *USE.get_or_init(|| std::env::var("CUBECL_CPU_JIT").map_or(true, |v| v != "llvm"))
}

#[cfg(feature = "cranelift")]
mod cranelift {
    use super::{Context, ModuleOp};
    use cranelift_codegen::settings::{self, Configurable};
    use pliron::op::Op;
    use pliron_clif_lower::{lower, scan};

    // SysV x86-64 variadic calls need %al = an upper bound on the xmm args,
    // which Cranelift can't set, so the lowerer routes foreign variadic calls
    // through `__pliron_va8.<sym>` stubs (`mov $8, %eax; jmp <sym>`). An
    // object build assembles them from `pending_asm`; the JIT has no
    // assembler, so the variadic callees CubeCL emits get prebuilt stubs.
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    mod va8 {
        unsafe extern "C" {
            fn printf(fmt: *const core::ffi::c_char, ...) -> core::ffi::c_int;
            fn __cubecl_va8_printf();
        }
        core::arch::global_asm!(
            ".pushsection .text.__cubecl_va8_printf,\"ax\",@progbits",
            ".hidden __cubecl_va8_printf",
            "__cubecl_va8_printf:",
            "mov eax, 8",
            "jmp {printf}",
            ".popsection",
            printf = sym printf,
        );
        pub const STUBS: &[(&str, &str)] = &[("printf", "__pliron_va8.printf")];
        pub fn addr(callee: &str) -> *const u8 {
            match callee {
                "printf" => __cubecl_va8_printf as *const u8,
                _ => unreachable!(),
            }
        }
    }

    /// `pending_asm` is only `%al` trampolines we prebuilt (see `va8`).
    fn only_known_va8(asm: &str) -> bool {
        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
        let known = |l: &str| {
            va8::STUBS.iter().any(|(c, s)| {
                l == format!("{s}:") || l == format!("jmp {c}") || l == "mov $8, %eax"
            })
        };
        #[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
        let known = |_: &str| false;
        asm.lines()
            .map(str::trim)
            .all(|l| l.is_empty() || l.starts_with('.') || known(l))
    }

    /// pliron-llvm -> Cranelift -> executable memory, entirely in Rust.
    pub fn compile(
        ctx: &mut Context,
        module: ModuleOp,
        kernel_name: &str,
    ) -> (cranelift_jit::JITModule, *const u8) {
        let mut fb = settings::builder();
        fb.set("opt_level", "speed").unwrap();
        fb.set("is_pic", "false").unwrap();
        let isa = cranelift_native::builder()
            .expect("host ISA unsupported by Cranelift")
            .finish(settings::Flags::new(fb))
            .unwrap();
        if std::env::var_os("CUBECL_CLIF_DUMP").is_some() {
            use pliron::printable::Printable;
            eprintln!("==== {kernel_name} (pliron) ====\n{}", module.get_operation().disp(ctx));
        }
        // Lines are up to 128 lanes: keep them as 128-bit SIMD parts rather
        // than per-lane scalars. `CUBECL_CLIF_SIMD=0` scalarizes (ablation).
        pliron_clif_lower::types::WIDE_SIMD.store(
            std::env::var("CUBECL_CLIF_SIMD").map_or(true, |v| v != "0"),
            std::sync::atomic::Ordering::Relaxed,
        );
        #[allow(unused_mut)]
        let wide = super::wide_bits();
    pliron_clif_lower::types::WIDE_F32_BITS.store(wide, std::sync::atomic::Ordering::Relaxed);
    #[cfg(target_arch = "x86_64")]
    cranelift_codegen::isa::x64::WIDE_VECTOR_BITS
        .store(wide.max(128), std::sync::atomic::Ordering::Relaxed);
    #[cfg(target_arch = "x86_64")]
    {
        // f16 vector arithmetic in f16 on ymm (AVX512-FP16 + VL); `CUBECL_CLIF_FP16=0` = via f32.
        let fp16 = wide >= 512
            && std::arch::is_x86_feature_detected!("avx512fp16")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && !std::env::var("CUBECL_CLIF_FP16").is_ok_and(|v| v == "0");
        pliron_clif_lower::types::NATIVE_F16.store(fp16, std::sync::atomic::Ordering::Relaxed);
        cranelift_codegen::isa::x64::NATIVE_FP16.store(fp16, std::sync::atomic::Ordering::Relaxed);
        // Scalar f16 casts in-register; `CUBECL_CLIF_F16C=0` = compiler-builtins libcalls.
        let f16c = std::arch::is_x86_feature_detected!("f16c")
            && !std::env::var("CUBECL_CLIF_F16C").is_ok_and(|v| v == "0");
        pliron_clif_lower::types::F16C.store(f16c, std::sync::atomic::Ordering::Relaxed);
        // Wide bf16 vectors packed, rounded by `vcvtneps2bf16`; `CUBECL_CLIF_BF16=0` = 128-bit emulation.
        let bf16 = wide >= 256
            && std::arch::is_x86_feature_detected!("avx512bf16")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && !std::env::var("CUBECL_CLIF_BF16").is_ok_and(|v| v == "0");
        pliron_clif_lower::types::NATIVE_BF16.store(bf16, std::sync::atomic::Ordering::Relaxed);
        cranelift_codegen::isa::x64::NATIVE_BF16.store(bf16, std::sync::atomic::Ordering::Relaxed);
    }
    let mut st = scan::scan(ctx, module.get_operation());
        // rustc_codegen_pliron's pliron-IR passes (A/B: `CUBECL_CLIF_IROPT=1`).
        if std::env::var("CUBECL_CLIF_IROPT").is_ok_and(|v| v == "1") {
            use pliron_clif_lower::iropt::*;
            instcombine::run(ctx, &mut st);
            memcpyopt::run(ctx, &mut st);
            sroa::forward(ctx, &mut st);
            phisimp::run(ctx, &st);
            sroa::run(ctx, &mut st);
            constload::run(ctx, &mut st);
        }
        #[allow(unused_mut)]
        let mut jb = cranelift_jit::JITBuilder::with_isa(
            isa.clone(),
            cranelift_module::default_libcall_names(),
        );
        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
        for (callee, stub) in va8::STUBS {
            jb.symbol(*stub, va8::addr(callee));
        }
        let mut jit = cranelift_jit::JITModule::new(jb);
        let lw = lower::lower_module(
            &mut jit,
            false,
            false,
            ctx,
            &st,
            &isa,
            kernel_name,
            &lower::NoHost,
        );
        let asm = lw.pending_asm(&st, false);
        assert!(
            only_known_va8(&asm),
            "cranelift JIT can't link module asm for '{kernel_name}':\n{asm}"
        );
        jit.finalize_definitions()
            .unwrap_or_else(|e| panic!("cranelift JIT link failed for '{kernel_name}': {e}"));
        let id = lw
            .func_id(kernel_name)
            .unwrap_or_else(|| panic!("kernel symbol '{kernel_name}' not found"));
        let addr = jit.get_finalized_function(id);
        (jit, addr)
    }
}

/// Widest f32 vector leaf (ymm = 256, zmm = 512) for this CPU; 0 = 128-bit
/// SIMD only. `CUBECL_CLIF_WIDE=0|256|512` overrides (capped by the CPU).
#[cfg(feature = "cranelift")]
fn wide_bits() -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        // F16C converts packed f16 twins of the wide f32 vectors.
        let f16c = std::arch::is_x86_feature_detected!("f16c");
        let cpu = if !f16c {
            0
        } else if std::arch::is_x86_feature_detected!("avx512f") {
            512
        } else if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            256
        } else {
            0
        };
        let want = std::env::var("CUBECL_CLIF_WIDE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512);
        if std::env::var("CUBECL_CLIF_SIMD").is_ok_and(|v| v == "0") {
            return 0;
        }
        cpu.min(want)
    }
    #[cfg(not(target_arch = "x86_64"))]
    0
}
