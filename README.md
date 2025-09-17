# How to Add and Use a New Metal Kernel

This project runs Metal compute kernels from Go via cgo. Follow these steps to add a new kernel and invoke it from Go.

## Prerequisites
- macOS with Xcode Command Line Tools (Metal frameworks).
- Go 1.24+ with `CGO_ENABLED=1` (default on macOS).

## 1) Add your kernel to the embedded source
All kernels are compiled from `internal/metal/kernels/mm.metal` (embedded by `internal/metal/embed.go`).

Keep the parameter layout identical to the host side:
```metal
typedef struct MatrixParams { int a_rows, a_cols; int b_rows, b_cols; } MatrixParams;

kernel void matrix_multiply_mykernel(
  device const MatrixParams *params,
  constant float *A,
  constant float *B,
  device float *C,
  uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params->a_cols || gid.y >= params->a_rows) return;
  float sum = 0.0;
  for (int k = 0; k < params->a_cols; ++k) {
    sum += A[gid.y * params->a_cols + k] * B[k * params->b_cols + gid.x];
  }
  C[gid.y * params->b_cols + gid.x] = sum;
}
```

Constraints (unless you also change the host code):
- Resource bindings: 0=`params` (setBytes), 1=`A`, 2=`B`, 3=`C`.
- Grid mapping: host sets `threadsPerGrid = (a_cols, a_rows, 1)`.
- `MatrixParams` layout must match the C/Go definitions (four 32‑bit ints).

## 2) Expose the kernel in the Objective‑C host
Declare and implement a wrapper like the existing naive path.

Edit `internal/metal/metal.h`:
```c
void* metal_mult_mykernel(MatrixParams *params);
```

Edit `internal/metal/metal.m`:
```objc
// Add a pipeline field (global in this file)
static id<MTLComputePipelineState> pipelineStateMyKernel;

// In initializePipelineAndCommandQueue(...), after the library is created:
id<MTLFunction> f = [lib newFunctionWithName:@"matrix_multiply_mykernel"];
pipelineStateMyKernel = [device newComputePipelineStateWithFunction:f error:&error];

// Kernel wrapper (mirrors metal_mult_naive)
void* metal_mult_mykernel (MatrixParams *params) {
  return metal_mult(params, pipelineStateMyKernel);
}
```

## 3) Add a Go wrapper
Create a helper in `internal/metal/metal.go` similar to `MultiplyNaive`:
```go
func MultiplyMyKernel(aRows, aCols, bRows, bCols int) []float32 {
    params := C.MatrixParams{C.int(aRows), C.int(aCols), C.int(bRows), C.int(bCols)}
    p := C.metal_mult_mykernel(&params)
    if p == nil { return nil }
    n := aRows * bCols
    res := make([]float32, n)
    tmp := unsafe.Slice((*float32)(p), n)
    copy(res, tmp)
    return res
}
```

If you support non‑mac builds, add a stub in `internal/metal/metal_stub.go` with the same signature returning `nil`.

## 4) Use the kernel
Example (from `cmd/fastgo/main.go` or another app):
```go
import "kylesmith19091/fastgo/internal/metal"

metal.CompileDefault() // compiles internal/metal/kernels/mm.metal
_, _ = metal.InitializeBuffersFloat32(A, B, aRows, aCols, bRows, bCols)
C := metal.MultiplyMyKernel(aRows, aCols, bRows, bCols)
```

## 5) Build and run
- Build: `go build ./...`
- Run example: `go run ./cmd/fastgo`

## Troubleshooting
- Unknown type ‘MatrixParams’: ensure the typedef exists in `mm.metal` and matches `internal/metal/metal.h`.
- Function not found: verify `newFunctionWithName:@"matrix_multiply_mykernel"` and the kernel name.
- Wrong results: confirm buffer indices (0=Params,1=A,2=B,3=C) and grid mapping `(a_cols, a_rows, 1)`.
