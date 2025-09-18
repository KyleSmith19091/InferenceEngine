# How to Add and Use Metal Kernels (Multi‑Kernel + Batched 3D)

This project runs Metal compute kernels from Go via cgo. It now supports:
- Compiling a library once and running multiple kernels by name.
- A generic runner for kernels with parameters and up to 3 buffers.
- A built‑in batched 3D matmul: [B,M,K] × [B,K,N] → [B,M,N].

## Modern LLM Roadmap
- The active focus is expanding these building blocks into a full GPU-first transformer inference stack capable of serving models like Qwen3 and Gemma3.
- Track progress and open work items in `TODO.md`; contributors should align new work with that list before implementing features.
- CPU fallbacks are intentionally out of scope for now—prioritize Metal kernels and GPU execution paths.

## Prerequisites
- macOS with Xcode Command Line Tools (Metal frameworks).
- Go 1.24+ with `CGO_ENABLED=1` (default on macOS).

## 1) Add your kernel to the embedded source
All kernels are compiled from `internal/metal/kernels/mm.metal` (embedded by `internal/metal/embed.go`).

Keep the parameter layout identical to the host side and follow the binding indices:
- Resource bindings: 0 = params (setBytes), 1 = A, 2 = B, 3 = C
- 2D grid mapping for matmul: threadsPerGrid = (b_cols, a_rows, 1)
- 3D batched mapping: threadsPerGrid = (n, m, batch)

2D example (parameters mirror the host’s C/Go struct):
```metal
typedef struct MatrixParams { int a_rows, a_cols; int b_rows, b_cols; } MatrixParams;

kernel void matrix_multiply_mykernel(
  device const MatrixParams *params,
  device const float *A,
  device const float *B,
  device float *C,
  uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params->b_cols || gid.y >= params->a_rows) return; // (n, m)
  float sum = 0.0;
  for (int k = 0; k < params->a_cols; ++k) {
    sum += A[gid.y * params->a_cols + k] * B[k * params->b_cols + gid.x];
  }
  C[gid.y * params->b_cols + gid.x] = sum;
}
```

Batched 3D example (already included as `matrix_multiply_batched_naive`):
```metal
typedef struct MatMul3DParams { int batch, m, k, n; } MatMul3DParams;

kernel void matrix_multiply_batched_naive(
  device const MatMul3DParams *params,
  device const float *A,
  device const float *B,
  device float *C,
  uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= params->n || gid.y >= params->m || gid.z >= params->batch) return; // (n,m,B)
  int b=gid.z, row=gid.y, col=gid.x, M=params->m, K=params->k, N=params->n;
  int aBase = b * M * K, bBase = b * K * N, cBase = b * M * N;
  float sum = 0.0f;
  for (int kk = 0; kk < K; ++kk) {
    sum += A[aBase + row * K + kk] * B[bBase + kk * N + col];
  }
  C[cBase + row * N + col] = sum;
}
```

## 2) Initialize the library and run kernels (generic path)
With the generic runtime you don’t need to add Objective‑C wrappers per kernel.

- Initialize once:
```go
import "kylesmith19091/fastgo/internal/metal"

metal.CompileLibrary() // compiles internal/metal/kernels/mm.metal and creates a command queue
```

- Ensure a kernel is compiled (optional; on‑demand compile also works):
```go
metal.EnsureKernel("matrix_multiply_batched_naive")
```

- Run a kernel using the generic runner:
```go
// Example for 3D batched matmul using a typed helper
err := metal.MatMulBatchedBuffers(bufA, bufB, bufC, batch, m, k, n)

// Or directly via the generic runner
params := metal.MatMul3DParams{Batch:int32(batch), M:int32(m), K:int32(k), N:int32(n)}
err := metal.RunKernel3(
    "matrix_multiply_batched_naive",
    unsafe.Pointer(&params), int(unsafe.Sizeof(params)),
    n, m, batch, // grid (n,m,B)
    bufA, bufB, bufC,
)
```

Convenience helpers available in Go:
- `CompileLibrary()` / `CompileLibraryFrom(src string)`
- `EnsureKernel(name string)`
- `RunKernel3(name string, paramsPtr unsafe.Pointer, paramsLen int, gridX, gridY, gridZ int, b0, b1, b2 *Buffer) error`
- `MatMulBatchedBuffers(a,b,c *Buffer, batch, m, k, n int) error`
- Legacy: `CompileDefault(kernelName string)` compiles and selects a single kernel (still supported).

## 3) Legacy wrapper approach (optional)
If you prefer a named wrapper per kernel:
- You can still add a C function in `internal/metal/metal.h` and implement it in `internal/metal/metal.m` that calls a shared encoder routine.
- Then expose a typed Go helper, similar to how `MultiplyNaiveBuffers` works.
- This is no longer required for new kernels thanks to the generic runner.

## 4) Examples
- 2D matmul example: `go run ./examples/mm`
- Batched 3D matmul example: `go run ./examples/batched_mm`
- Embedding layer toy example: `go run ./examples/embedding`

## Build & Run
- Build: `go build ./...`
- Run example: `go run ./examples/batched_mm`
- Cross‑platform (no Metal): `CGO_ENABLED=0 go build` (stubs; no GPU execution).

## Troubleshooting
- Kernel not found: ensure the function name in `mm.metal` matches what you pass to `EnsureKernel`/`RunKernel3`.
- Wrong results: confirm resource bindings (0=params,1=A,2=B,3=C) and grid mapping (2D: `(b_cols, a_rows, 1)`, 3D: `(n, m, batch)`).
- Params mismatch: C/Go param struct must match the Metal typedef (field order and 32‑bit ints).
- Ensure `CompileLibrary()` (or `CompileDefault("name")`) is called before running kernels.