//go:build darwin && cgo

package metal

/*
#cgo darwin CFLAGS: -x objective-c -fobjc-arc
#cgo darwin LDFLAGS: -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics
#include <stdlib.h>
#include "metal.h"
*/
import "C"
import (
	"errors"
	"fmt"
	"unsafe"
)

// MatrixParams matches the definitions in metal.h
type MatrixParams struct {
	a_rows, a_cols int32
	b_rows, b_cols int32
}

// Compile compiles the Metal source and initializes pipelines/queue.
func Compile(metalSource string, kernelName string) {
	src := C.CString(metalSource)
	kernel := C.CString(kernelName)
	defer C.free(unsafe.Pointer(src))
	defer C.free(unsafe.Pointer(kernel))
	C.initializePipelineAndCommandQueue(src, kernel)
}

// CompileLibraryFrom compiles the Metal source and initializes the library + queue (no specific kernel).
func CompileLibraryFrom(metalSource string) {
	src := C.CString(metalSource)
	defer C.free(unsafe.Pointer(src))
	C.initializeLibrary(src)
}

// InitializeBuffersFloat32 uploads A and B to GPU buffers and prepares the output buffer.
func InitializeBuffersFloat32(a, b []float32, aRows, aCols, bRows, bCols int) (int, error) {
	if len(a) != aRows*aCols {
		return 0, fmt.Errorf("len(a)=%d does not match dims %dx%d", len(a), aRows, aCols)
	}
	if len(b) != bRows*bCols {
		return 0, fmt.Errorf("len(b)=%d does not match dims %dx%d", len(b), bRows, bCols)
	}
	if aCols != bRows {
		return 0, fmt.Errorf("incompatible shapes: %dx%d * %dx%d", aRows, aCols, bRows, bCols)
	}
	if len(a) == 0 || len(b) == 0 {
		return 0, fmt.Errorf("empty input matrices")
	}

	outCount := aRows * bCols
	C.initializeMTLBuffers(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		C.int(4), // sizeof(float32)
		C.int(len(a)),
		C.int(len(b)),
		C.int(outCount),
	)
	return outCount, nil
}

// MultiplyNaive runs the naive Metal kernel and returns a copy of the result.
func MultiplyNaive(aRows, aCols, bRows, bCols int) []float32 {
	params := C.MatrixParams{
		a_rows: C.int(aRows),
		a_cols: C.int(aCols),
		b_rows: C.int(bRows),
		b_cols: C.int(bCols),
	}
	p := C.metal_mult_naive(&params)
	if p == nil {
		return nil
	}
	n := aRows * bCols
	tmp := unsafe.Slice((*float32)(p), n)
	res := make([]float32, n)
	copy(res, tmp)
	return res
}

// Buffer is a thin wrapper over an MTLBuffer for tensor storage.
type Buffer struct {
	ptr  unsafe.Pointer
	size int
}

// NewBuffer allocates an MTLBuffer of given size in bytes.
func NewBuffer(size int) (*Buffer, error) {
	if size <= 0 {
		return nil, fmt.Errorf("invalid buffer size: %d", size)
	}
	p := C.mtl_new_buffer(C.int(size))
	if p == nil {
		return nil, fmt.Errorf("mtl_new_buffer returned nil")
	}
	return &Buffer{ptr: p, size: size}, nil
}

// Write copies host bytes into the buffer.
func (b *Buffer) Write(src []byte) error {
	if b == nil || b.ptr == nil {
		return fmt.Errorf("nil buffer")
	}
	if len(src) > b.size {
		return fmt.Errorf("write overflow: %d > %d", len(src), b.size)
	}
	if len(src) == 0 {
		return nil
	}
	C.mtl_buffer_write(b.ptr, unsafe.Pointer(&src[0]), C.int(len(src)))
	return nil
}

// Read copies buffer bytes into dst.
func (b *Buffer) Read(dst []byte) error {
	if b == nil || b.ptr == nil {
		return fmt.Errorf("nil buffer")
	}
	if len(dst) > b.size {
		return fmt.Errorf("read overflow: %d > %d", len(dst), b.size)
	}
	if len(dst) == 0 {
		return nil
	}
	C.mtl_buffer_read(b.ptr, unsafe.Pointer(&dst[0]), C.int(len(dst)))
	return nil
}

func (b *Buffer) ReadN(start int, numberBytes int) ([]byte, error) {
	if b == nil {
		return nil, errors.New("nil buffer")
	}
	if start < 0 || start > b.size {
		return nil, fmt.Errorf("%d out of bounds of buffer with size %d", start, b.size)
	}
	if numberBytes < 0 {
		return nil, fmt.Errorf("negative read length: %d", numberBytes)
	}
	if start+numberBytes > b.size {
		return nil, fmt.Errorf("%d can not read more bytes than in buffer with size %d", start+numberBytes, b.size)
	}
	if numberBytes == 0 {
		return []byte{}, nil
	}

	// construct destination buffer with size equal to the number of bytes we want to read
	dst := make([]byte, numberBytes)

	// read starting at the given offset into the dest using C helper
	C.mtl_buffer_read_at(b.ptr, C.int(start), unsafe.Pointer(&dst[0]), C.int(len(dst)))
	return dst, nil
}

// Size returns the buffer length in bytes.
func (b *Buffer) Size() int {
	if b == nil {
		return 0
	}
	return b.size
}

// Ptr returns the underlying pointer for binding into encoders.
func (b *Buffer) Ptr() unsafe.Pointer {
	if b == nil {
		return nil
	}
	return b.ptr
}

// Close releases the underlying MTLBuffer.
func (b *Buffer) Close() error {
	if b == nil || b.ptr == nil {
		return nil
	}
	C.mtl_release_buffer(b.ptr)
	b.ptr = nil
	b.size = 0
	return nil
}

// MultiplyNaiveBuffers runs the naive kernel using provided device buffers.
// Buffers must be sized for A[aRows*aCols], B[bRows*bCols], C[aRows*bCols] with float32 elements.
func MultiplyNaiveBuffers(a, b, c *Buffer, aRows, aCols, bRows, bCols int) error {
	if a == nil || b == nil || c == nil {
		return fmt.Errorf("nil buffer")
	}
	if aCols != bRows {
		return fmt.Errorf("incompatible shapes: %dx%d * %dx%d", aRows, aCols, bRows, bCols)
	}
	params := C.MatrixParams{a_rows: C.int(aRows), a_cols: C.int(aCols), b_rows: C.int(bRows), b_cols: C.int(bCols)}
	_ = C.metal_mult_naive_with_buffers(&params, a.ptr, b.ptr, c.ptr)
	return nil
}
