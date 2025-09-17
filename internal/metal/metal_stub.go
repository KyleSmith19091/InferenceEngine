//go:build !darwin || !cgo

package metal

import "unsafe"

// Stubs for non-macOS or when cgo is disabled, so the package compiles.

type MatrixParams struct {
	a_rows, a_cols int32
	b_rows, b_cols int32
}

func initializePipelineAndCommandQueue(_ *byte)                                           {}
func initializeMTLBuffers(_ unsafe.Pointer, _ unsafe.Pointer, _ int, _ int, _ int, _ int) {}
func metal_mult_naive(_ *MatrixParams) unsafe.Pointer                                     { return nil }
func mps_mult(_ *MatrixParams) unsafe.Pointer                                             { return nil }

// Public API stubs
func InitializeBuffersFloat32(_ []float32, _ []float32, _ int, _ int, _ int, _ int) (int, error) {
	return 0, nil
}
func MultiplyNaive(_ int, _ int, _ int, _ int) []float32 { return nil }

// Buffer stub for non-metal builds
type Buffer struct {
	ptr  unsafe.Pointer
	size int
}

func NewBuffer(size int) (*Buffer, error) { return &Buffer{ptr: nil, size: size}, nil }
func (b *Buffer) Write(_ []byte) error    { return nil }
func (b *Buffer) Read(_ []byte) error     { return nil }
func (b *Buffer) Size() int {
	if b == nil {
		return 0
	}
	return b.size
}
func (b *Buffer) Ptr() unsafe.Pointer {
	if b == nil {
		return nil
	}
	return b.ptr
}
func (b *Buffer) Close() error { return nil }

func MultiplyNaiveBuffers(_ *Buffer, _ *Buffer, _ *Buffer, _ int, _ int, _ int, _ int) error {
	return nil
}
