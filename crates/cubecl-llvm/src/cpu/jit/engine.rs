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
    _jit: OrcJit,
}

/// SAFETY: Compiled code is immutable and its JIT owns the context.
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
    pub fn compile(
        ctx: &mut Context,
        module: ModuleOp,
        kernel_name: &str,
        requirements: KernelRequirements,
        io: Vec<BufferIOAttr>,
        keep_object: bool,
    ) -> pliron::result::Result<Self> {
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
            _jit: jit,
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
