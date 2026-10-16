use super::data::PlironData;
use crate::{
    cpu::shared_memory::SharedMemories,
    prelude::{Context, ModuleOp},
    shared::llvm_module::{LlvmModule, TargetMachine},
};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::bytes::Bytes;
use cubecl_runtime::compiler::CompilationError;
use cubecl_runtime::kernel::BufferIOAttr;
use pliron_llvm::{llvm_sys::core::LLVMContext, to_llvm_ir};
use std::{
    ffi::{CStr, CString, c_void},
    fmt::Display,
    sync::{Arc, Mutex, Once},
};

/// Kernel ABI: buffer pointers, cube count x/y/z, unit position x/y/z,
/// barrier state, metadata.
type KernelFn = extern "C" fn(*mut *mut c_void, u32, u32, u32, u32, u32, u32, *mut u32, *mut u64);

// compiler-rt builtins missing from the MSVC runtime; defined in-process so
// the LLJIT's symbol search (or our explicit absolute mapping) can find them.
// compiler-rt declares these with `uint16_t` for the 16-bit float —
// integer-register ABI, not xmm.
extern "C" fn __truncsfbf2(v: f32) -> u16 {
    half::bf16::from_f32(v).to_bits()
}

extern "C" fn __extendhfsf2(v: u16) -> f32 {
    half::f16::from_bits(v).to_f32()
}

extern "C" fn __truncdfsf2(v: f64) -> f32 {
    v as f32
}

/// Resources and scheduling required for a launch.
#[derive(Clone, Debug, Default)]
pub struct KernelRequirements {
    /// Cube barriers require a separate thread for each unit.
    pub needs_parallelism: bool,
    /// Shared memory required for a launch.
    pub shared_memories: SharedMemories,
}

/// Self-contained wrapper around LLVM `OrcV2` LLJIT handle.
struct OrcJit(llvm_sys::orc2::lljit::LLVMOrcLLJITRef);

impl OrcJit {
    fn new() -> Result<Self, String> {
        unsafe {
            let mut jit = std::mem::MaybeUninit::uninit();
            let err =
                llvm_sys::orc2::lljit::LLVMOrcCreateLLJIT(jit.as_mut_ptr(), std::ptr::null_mut());
            if !err.is_null() {
                let msg_ptr = llvm_sys::error::LLVMGetErrorMessage(err);
                let text = CStr::from_ptr(msg_ptr).to_string_lossy().into_owned();
                llvm_sys::error::LLVMDisposeErrorMessage(msg_ptr);
                return Err(text);
            }
            let jit = jit.assume_init();

            let prefix = llvm_sys::orc2::lljit::LLVMOrcLLJITGetGlobalPrefix(jit);
            let mut generator = std::ptr::null_mut();
            let err = llvm_sys::orc2::LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess(
                &mut generator,
                prefix,
                None,
                std::ptr::null_mut(),
            );
            if !err.is_null() {
                let msg_ptr = llvm_sys::error::LLVMGetErrorMessage(err);
                let text = CStr::from_ptr(msg_ptr).to_string_lossy().into_owned();
                llvm_sys::error::LLVMDisposeErrorMessage(msg_ptr);
                llvm_sys::orc2::lljit::LLVMOrcDisposeLLJIT(jit);
                return Err(text);
            }
            let main_jd = llvm_sys::orc2::lljit::LLVMOrcLLJITGetMainJITDylib(jit);
            llvm_sys::orc2::LLVMOrcJITDylibAddGenerator(main_jd, generator);

            // compiler-rt builtins the UCRT doesn't provide; needed on Windows
            // where the process search generator can't see them otherwise.
            for (name, addr) in [
                ("__truncsfbf2", __truncsfbf2 as *const () as u64),
                ("__extendhfsf2", __extendhfsf2 as *const () as u64),
                ("__truncdfsf2", __truncdfsf2 as *const () as u64),
            ] {
                let name_interned = llvm_sys::orc2::lljit::LLVMOrcLLJITMangleAndIntern(
                    jit,
                    CString::new(name).unwrap().as_ptr(),
                );
                let mut pair = llvm_sys::orc2::LLVMOrcCSymbolMapPair {
                    Name: name_interned,
                    Sym: llvm_sys::orc2::LLVMJITEvaluatedSymbol {
                        Address: addr,
                        Flags: llvm_sys::orc2::LLVMJITSymbolFlags {
                            GenericFlags: llvm_sys::orc2::LLVMJITSymbolGenericFlags::LLVMJITSymbolGenericFlagsExported
                                as u8,
                            TargetFlags: 0,
                        },
                    },
                };
                let mu = llvm_sys::orc2::LLVMOrcAbsoluteSymbols(&mut pair as *mut _, 1);
                let err = llvm_sys::orc2::LLVMOrcJITDylibDefine(main_jd, mu);
                if !err.is_null() {
                    let msg_ptr = llvm_sys::error::LLVMGetErrorMessage(err);
                    let text = CStr::from_ptr(msg_ptr).to_string_lossy().into_owned();
                    llvm_sys::error::LLVMDisposeErrorMessage(msg_ptr);
                    llvm_sys::orc2::lljit::LLVMOrcDisposeLLJIT(jit);
                    return Err(text);
                }
            }

            Ok(Self(jit))
        }
    }

    fn add_object_file(&self, bytes: &[u8], name: &str) -> Result<(), String> {
        let name_c = CString::new(name).map_err(|e| e.to_string())?;
        unsafe {
            let main_jd = llvm_sys::orc2::lljit::LLVMOrcLLJITGetMainJITDylib(self.0);
            let mem_buf = llvm_sys::core::LLVMCreateMemoryBufferWithMemoryRangeCopy(
                bytes.as_ptr() as *const _,
                bytes.len(),
                name_c.as_ptr(),
            );
            let err = llvm_sys::orc2::lljit::LLVMOrcLLJITAddObjectFile(self.0, main_jd, mem_buf);
            if !err.is_null() {
                let msg_ptr = llvm_sys::error::LLVMGetErrorMessage(err);
                let text = CStr::from_ptr(msg_ptr).to_string_lossy().into_owned();
                llvm_sys::error::LLVMDisposeErrorMessage(msg_ptr);
                return Err(text);
            }
            Ok(())
        }
    }

    fn lookup_symbol(&self, name: &str) -> Result<u64, String> {
        let name_c = CString::new(name).map_err(|e| e.to_string())?;
        unsafe {
            let mut addr = 0u64;
            let err = llvm_sys::orc2::lljit::LLVMOrcLLJITLookup(self.0, &mut addr, name_c.as_ptr());
            if !err.is_null() {
                let msg_ptr = llvm_sys::error::LLVMGetErrorMessage(err);
                let text = CStr::from_ptr(msg_ptr).to_string_lossy().into_owned();
                llvm_sys::error::LLVMDisposeErrorMessage(msg_ptr);
                return Err(text);
            }
            Ok(addr)
        }
    }
}

impl Drop for OrcJit {
    fn drop(&mut self) {
        unsafe {
            let err = llvm_sys::orc2::lljit::LLVMOrcDisposeLLJIT(self.0);
            if !err.is_null() {
                let msg_ptr = llvm_sys::error::LLVMGetErrorMessage(err);
                llvm_sys::error::LLVMDisposeErrorMessage(msg_ptr);
            }
        }
    }
}

/// A compiled kernel and its owning JIT.
struct JitKernel {
    func: KernelFn,
    requirements: KernelRequirements,
    /// Buffer access modes in binding order.
    io: Vec<BufferIOAttr>,
    object_bytes: Mutex<Option<Bytes>>,
    _jit: Jit,
}

/// Owner of the compiled code; dropping it would unmap the kernel.
enum Jit {
    Llvm(#[allow(dead_code)] OrcJit),
    #[cfg(feature = "cranelift")]
    Cranelift(#[allow(dead_code)] cranelift_jit::JITModule),
}
unsafe impl Send for JitKernel {}
unsafe impl Sync for JitKernel {}

#[derive(Clone)]
pub struct PlironEngine(Arc<JitKernel>);

static INIT_NATIVE: Once = Once::new();

pub(crate) fn init_native() {
    INIT_NATIVE.call_once(|| {
        pliron_llvm::llvm_sys::target::initialize_native()
            .expect("failed to initialize native target");
    });
}

impl PlironEngine {
    /// `CUBECL_JIT_TIME=1` prints `cubecl-jit-time <jit> <kernel> <µs>` per
    /// compile (pliron-llvm module in, callable kernel out).
    pub fn compile(
        ctx: &mut Context,
        module: ModuleOp,
        kernel_name: &str,
        requirements: KernelRequirements,
        io: Vec<BufferIOAttr>,
        keep_object: bool,
    ) -> pliron::result::Result<Self> {
        let t = std::time::Instant::now();
        let r = Self::compile_inner(ctx, module, kernel_name, requirements, io, keep_object);
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
        keep_object: bool,
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
                object_bytes: Mutex::new(None),
                _jit: Jit::Cranelift(jit),
            })));
        }
        init_native();

        // CUBECL_PLIRON_CLIF=1: pliron LLVM dialect -> Cranelift -> object,
        // skipping LLVM entirely. Same OrcJit load path either way.
        if std::env::var_os("CUBECL_PLIRON_CLIF").is_some() {
            return Self::compile_clif(ctx, module, kernel_name, requirements, io, keep_object);
        }

        let llvm_ctx = LLVMContext::default();
        let llvm_module = to_llvm_ir::convert_module(ctx, &llvm_ctx, module)?;
        #[cfg(feature = "pliron-dump")]
        if let Some(dir) = ir_dump_path(kernel_name) {
            let _ = std::fs::write(dir.join("llvm.ll"), llvm_module.to_string());
        }

        let module = LlvmModule::new(&llvm_module.to_string())
            .unwrap_or_else(|err| panic!("LLVM parsing failed for '{kernel_name}': {err}"));

        let target_machine = TargetMachine::native(&module)
            .unwrap_or_else(|err| panic!("failed to create native target machine: {err}"));

        module
            .run_passes(PASS_PIPELINE, &target_machine)
            .unwrap_or_else(|err| panic!("LLVM optimization failed for '{kernel_name}': {err}"));

        #[cfg(feature = "pliron-dump")]
        if let Some(dir) = ir_dump_path(kernel_name) {
            let _ = std::fs::write(dir.join("llvm.opt.ll"), module.print());
        }

        let object_bytes = target_machine
            .emit(
                module,
                llvm_sys::target_machine::LLVMCodeGenFileType::LLVMObjectFile,
            )
            .unwrap_or_else(|err| panic!("LLVM code generation failed for '{kernel_name}': {err}"));

        #[cfg(feature = "pliron-dump")]
        if let Some(dir) = ir_dump_path(kernel_name) {
            let _ = std::fs::write(dir.join("kernel.o"), &object_bytes);
        }

        let engine = Self::from_object(&object_bytes, kernel_name, requirements, io)
            .unwrap_or_else(|err| {
                panic!("failed to load compiled object for '{kernel_name}': {err}")
            });

        if keep_object {
            *engine.0.object_bytes.lock().unwrap() = Some(Bytes::from_bytes_vec(object_bytes));
        }

        Ok(engine)
    }

    /// pliron LLVM dialect -> Cranelift -> object -> OrcJit, bypassing LLVM.
    fn compile_clif(
        ctx: &mut Context,
        module: ModuleOp,
        kernel_name: &str,
        requirements: KernelRequirements,
        io: Vec<BufferIOAttr>,
        keep_object: bool,
    ) -> pliron::result::Result<Self> {
        use pliron_clif_lower::{iropt, lower, scan};

        use pliron::{op::Op as _, printable::Printable as _};
        #[cfg(feature = "pliron-dump")]
        if let Some(dir) = ir_dump_path(kernel_name) {
            let text = module
                .get_operation()
                .deref(ctx)
                .print(ctx, &pliron::printable::State::default())
                .to_string();
            let _ = std::fs::write(dir.join("kernel.pliron"), text);
        }
        let isa = clif_isa();

        // Host SIMD knobs for pliron-ir-opt's leaf planning. CubeCL lines
        // reach 128 lanes, so vectors wider than 128 bits must split into
        // native parts rather than per-lane scalars; packed f16 needs both
        // WIDE_SIMD and NATIVE_F16/F16C to stay SIMD. Read the flags off the
        // ISA we just built so they can never disagree with the backend.
        {
            use pliron_clif_lower::types;
            use std::sync::atomic::Ordering::Relaxed;
            let flag = |name: &str| {
                isa.isa_flags()
                    .iter()
                    .find(|v| v.name == name)
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
            };
            types::WIDE_SIMD.store(true, Relaxed);
            types::NATIVE_F16
                .store(flag("has_fp16") || flag("has_avx512fp16"), Relaxed);
            types::F16C.store(flag("has_fp16") || flag("has_f16c"), Relaxed);
            types::NATIVE_BF16.store(flag("has_avx512bf16"), Relaxed);
            let wide = if flag("has_avx512f") {
                512
            } else if flag("has_avx2") {
                256
            } else {
                0
            };
            types::WIDE_F32_BITS.store(wide, Relaxed);
        }

        let mut st = scan::scan(ctx, module.get_operation());

        // rustc backend's pass order, minus passes that need rustc metadata
        // (spec/devirt use callgraph tables rustc populates itself).
        let mut adj = None;
        iropt::inline::run(
            ctx,
            &mut st,
            false,
            iropt::inline::SMALL_LIMIT,
            None,
            false,
            None,
            &mut adj,
        );
        iropt::instcombine::run(ctx, &mut st);
        iropt::memcpyopt::run(ctx, &mut st);
        iropt::phisimp::run(ctx, &mut st);
        iropt::sroa::run(ctx, &mut st);
        iropt::constload::run(ctx, &mut st);

        let object_bytes = lower::lower_to_object(
            false,
            false,
            ctx,
            &st,
            isa,
            kernel_name,
            &lower::NoHost,
        );

        let engine = Self::from_object(&object_bytes, kernel_name, requirements, io)
            .map_err(|err| {
                pliron::arg_error_noloc!(
                    "failed to load pliron-clif object for '{kernel_name}': {err}"
                )
            })?;

        if keep_object {
            *engine.0.object_bytes.lock().unwrap() = Some(Bytes::from_bytes_vec(object_bytes));
        }

        Ok(engine)
    }

    /// Load precompiled object bytes directly into the JIT linking layer.
    pub fn from_object(
        bytes: &[u8],
        entrypoint: &str,
        requirements: KernelRequirements,
        io: Vec<BufferIOAttr>,
    ) -> Result<Self, CompilationError> {
        init_native();

        let jit = OrcJit::new().map_err(|err| CompilationError::Generic {
            reason: format!("failed to create OrcJit: {err}"),
            backtrace: BackTrace::capture(),
        })?;

        jit.add_object_file(bytes, entrypoint)
            .map_err(|err| CompilationError::Generic {
                reason: format!("failed to add object file to JIT: {err}"),
                backtrace: BackTrace::capture(),
            })?;

        let addr = jit
            .lookup_symbol(entrypoint)
            .map_err(|err| CompilationError::Generic {
                reason: format!("kernel symbol '{entrypoint}' not found: {err}"),
                backtrace: BackTrace::capture(),
            })?;

        let func: KernelFn = unsafe { std::mem::transmute::<u64, KernelFn>(addr) };

        Ok(PlironEngine(Arc::new(JitKernel {
            func,
            requirements,
            io,
            object_bytes: Mutex::new(None),
            _jit: Jit::Llvm(jit),
        })))
    }

    /// Takes the precompiled object bytes by value, leaving None in the engine.
    pub fn take_object_bytes(&self) -> Option<Bytes> {
        self.0.object_bytes.lock().unwrap().take()
    }

    pub fn requirements(&self) -> &KernelRequirements {
        &self.0.requirements
    }

    /// Buffer access modes in binding order.
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

/// Host ISA for the Cranelift path: native feature detection, max speed.
fn clif_isa() -> Arc<dyn cranelift_codegen::isa::TargetIsa> {
    use cranelift_codegen::settings::{self, Configurable};
    let mut fb = settings::builder();
    fb.set("is_pic", "true").unwrap();
    fb.set("enable_verifier", "false").unwrap();
    fb.set("preserve_frame_pointers", "false").unwrap();
    fb.set("opt_level", "speed_and_size").unwrap();
    fb.set("enable_llvm_abi_extensions", "true").unwrap();
    fb.enable("enable_multi_ret_implicit_sret").unwrap();
    let flags = settings::Flags::new(fb);
    cranelift_native::builder()
        .expect("native ISA builder")
        .finish(flags)
        .expect("native ISA finish")
}

#[cfg(feature = "pliron-dump")]
/// IR dump directory, enabled by `CUBECL_DEBUG_PLIRON`.
pub(crate) fn ir_dump_path(kernel_name: &str) -> Option<std::path::PathBuf> {
    let dir = std::env::var("CUBECL_DEBUG_PLIRON").ok()?;
    let path = std::path::Path::new(&dir).join(kernel_name);
    std::fs::create_dir_all(&path).ok()?;
    Some(path)
}

/// Optimization pipeline for JIT compilation.
const PASS_PIPELINE: &CStr = c"default<O3>";
/// Vector registers the active JIT allocates, when fewer than the ISA has:
/// Cranelift x64 allocates zmm16-31 only in 512-bit wide mode.
pub fn vector_register_cap() -> Option<u32> {
    #[cfg(feature = "cranelift")]
    if use_cranelift()
        && (wide_bits() < 512 || std::env::var_os("PLIRON_NO_HIGH_XMM").is_some())
    {
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
        // `PLIRON_TRACE=pliron::bcheck=debug` (any subscriber-worthy filter)
        // installs a stderr tracing subscriber so the optimizer's
        // `target: "pliron::*"` debug logs are visible — the JIT installs
        // no tracing layer of its own.
        if let Ok(filter) = std::env::var("PLIRON_TRACE") {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                use tracing_subscriber::prelude::*;
                let filter = tracing_subscriber::EnvFilter::builder().parse_lossy(filter);
                let _ = tracing_subscriber::registry()
                    .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
                    .with(filter)
                    .try_init();
            });
        }
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
        // libm builtins: explicit bindings beat `lookup_with_dlsym`, whose
        // Windows fallback only checks the exe + ucrtbase.
        for (name, addr) in libm_symbols() {
            jb.symbol(name, addr);
        }
        // `printf` has no Windows DLL export; forward to our C++ shim.
        #[cfg(all(target_arch = "x86_64", target_os = "windows"))]
        {
            unsafe extern "C" {
                fn __cubecl_jit_printf(fmt: *const core::ffi::c_char, ...) -> core::ffi::c_int;
            }
            jb.symbol("printf", __cubecl_jit_printf as *const u8);
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

    /// Symbol bindings for the libm calls the lowerer emits, resolved to Rust
    /// `std` fns so nothing depends on what the host CRT happens to export.
    /// `__truncsfbf2` is missing from the MSVC CRT entirely.
    fn libm_symbols() -> Vec<(&'static str, *const u8)> {
        macro_rules! unary {
            ($($f:ident: |$v:ident| $e:expr),* $(,)?) => {$(
                extern "C" fn $f($v: f64) -> f64 { $e }
            )*};
        }
        macro_rules! unaryf {
            ($($f:ident: |$v:ident| $e:expr),* $(,)?) => {$(
                extern "C" fn $f($v: f32) -> f32 { $e }
            )*};
        }
        unary! {
            _sin: |v| v.sin(),
            _cos: |v| v.cos(),
            _tan: |v| v.tan(),
            _asin: |v| v.asin(),
            _acos: |v| v.acos(),
            _atan: |v| v.atan(),
            _exp: |v| v.exp(),
            _exp2: |v| v.exp2(),
            _ln: |v| v.ln(),
            _log2: |v| v.log2(),
            _log10: |v| v.log10(),
            _sinh: |v| v.sinh(),
            _cosh: |v| v.cosh(),
            _tanh: |v| v.tanh(),
            _roundeven: |v| v.round_ties_even(),
            _expm1: |v| v.exp_m1(),
        }
        unaryf! {
            _sinf: |v| v.sin(),
            _cosf: |v| v.cos(),
            _tanf: |v| v.tan(),
            _asinf: |v| v.asin(),
            _acosf: |v| v.acos(),
            _atanf: |v| v.atan(),
            _expf: |v| v.exp(),
            _exp2f: |v| v.exp2(),
            _lnf: |v| v.ln(),
            _log2f: |v| v.log2(),
            _log10f: |v| v.log10(),
            _sinhf: |v| v.sinh(),
            _coshf: |v| v.cosh(),
            _tanhf: |v| v.tanh(),
            _roundevenf: |v| v.round_ties_even(),
            _expm1f: |v| v.exp_m1(),
        }
        extern "C" fn _pow(a: f64, b: f64) -> f64 { a.powf(b) }
        extern "C" fn _powf(a: f32, b: f32) -> f32 { a.powf(b) }
        extern "C" fn _atan2(a: f64, b: f64) -> f64 { a.atan2(b) }
        extern "C" fn _atan2f(a: f32, b: f32) -> f32 { a.atan2(b) }
        extern "C" fn _hypot(a: f64, b: f64) -> f64 { a.hypot(b) }
        extern "C" fn _hypotf(a: f32, b: f32) -> f32 { a.hypot(b) }
        extern "C" fn _fmax(a: f64, b: f64) -> f64 { a.max(b) }
        extern "C" fn _fmaxf(a: f32, b: f32) -> f32 { a.max(b) }
        extern "C" fn _fmin(a: f64, b: f64) -> f64 { a.min(b) }
        extern "C" fn _fminf(a: f32, b: f32) -> f32 { a.min(b) }
        extern "C" fn _fmod(a: f64, b: f64) -> f64 { a % b }
        extern "C" fn _fmodf(a: f32, b: f32) -> f32 { a % b }
        extern "C" fn __truncsfbf2(v: f32) -> u16 { half::bf16::from_f32(v).to_bits() }
        extern "C" fn __extendhfsf2(v: u16) -> f32 { half::f16::from_bits(v).to_f32() }

        vec![
            ("sin", _sin as *const u8), ("sinf", _sinf as *const u8),
            ("cos", _cos as *const u8), ("cosf", _cosf as *const u8),
            ("tan", _tan as *const u8), ("tanf", _tanf as *const u8),
            ("asin", _asin as *const u8), ("asinf", _asinf as *const u8),
            ("acos", _acos as *const u8), ("acosf", _acosf as *const u8),
            ("atan", _atan as *const u8), ("atanf", _atanf as *const u8),
            ("exp", _exp as *const u8), ("expf", _expf as *const u8),
            ("exp2", _exp2 as *const u8), ("exp2f", _exp2f as *const u8),
            ("log", _ln as *const u8), ("logf", _lnf as *const u8),
            ("log2", _log2 as *const u8), ("log2f", _log2f as *const u8),
            ("log10", _log10 as *const u8), ("log10f", _log10f as *const u8),
            ("sinh", _sinh as *const u8), ("sinhf", _sinhf as *const u8),
            ("cosh", _cosh as *const u8), ("coshf", _coshf as *const u8),
            ("tanh", _tanh as *const u8), ("tanhf", _tanhf as *const u8),
            ("roundeven", _roundeven as *const u8),
            ("roundevenf", _roundevenf as *const u8),
            ("expm1", _expm1 as *const u8), ("expm1f", _expm1f as *const u8),
            ("pow", _pow as *const u8), ("powf", _powf as *const u8),
            ("atan2", _atan2 as *const u8), ("atan2f", _atan2f as *const u8),
            ("hypot", _hypot as *const u8), ("hypotf", _hypotf as *const u8),
            ("fmax", _fmax as *const u8), ("fmaxf", _fmaxf as *const u8),
            ("fmin", _fmin as *const u8), ("fminf", _fminf as *const u8),
            ("fmod", _fmod as *const u8), ("fmodf", _fmodf as *const u8),
            ("__truncsfbf2", __truncsfbf2 as *const u8),
            ("__extendhfsf2", __extendhfsf2 as *const u8),
        ]
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
