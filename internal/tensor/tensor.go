package tensor

import (
	"errors"
	"math"
	"unsafe"

	"kylesmith19091/fastgo/internal/metal"
)

// DType represents element types supported by the engine.
type DType int

const (
	Float16 DType = iota
	BFloat16
	Float32
	Int8
	Int4 // packed: 2 values per byte
)

func (dt DType) String() string {
	switch dt {
	case Float16:
		return "float16"
	case BFloat16:
		return "bfloat16"
	case Float32:
		return "float32"
	case Int8:
		return "int8"
	case Int4:
		return "int4"
	default:
		return "unknown"
	}
}

// SizeOf returns the size in bytes for one element (rounded up for packed types).
func (dt DType) SizeOf() int {
	switch dt {
	case Float16, BFloat16:
		return 2
	case Float32:
		return 4
	case Int8:
		return 1
	case Int4:
		return 1 // two elems per byte; use BytesFor for arrays
	default:
		return 0
	}
}

// SizeOf is a convenience free function mirroring DType.SizeOf.
func SizeOf(dt DType) int { return dt.SizeOf() }

// BytesFor returns the number of bytes to store numel elements of dtype.
func BytesFor(dt DType, numel int) int {
	switch dt {
	case Int4:
		return (numel + 1) / 2
	default:
		return dt.SizeOf() * numel
	}
}

// Tensor is a device-resident multi-dimensional array backed by a Metal buffer.
type Tensor struct {
	DT      DType
	Shape   []int
	Strides []int // in bytes
	Offset  int   // in bytes
	buf     *metal.Buffer
	own     bool // owns underlying buffer
}

// New allocates a device buffer and creates a contiguous tensor view.
func New(dt DType, shape ...int) (*Tensor, error) {
	if !IsValidShape(shape) {
		return nil, errors.New("invalid shape")
	}
	numel := Numel(shape)
	nbytes := BytesFor(dt, numel)
	mbuf, err := metal.NewBuffer(nbytes)
	if err != nil {
		return nil, err
	}
	return &Tensor{DT: dt, Shape: append([]int(nil), shape...), Strides: DefaultStridesBytes(dt, shape), Offset: 0, buf: mbuf, own: true}, nil
}

// FromFloat32 packs and uploads float32 data into a new device tensor.
func FromFloat32(dt DType, data []float32, shape ...int) (*Tensor, error) {
	t, err := New(dt, shape...)
	if err != nil {
		return nil, err
	}
	switch dt {
	case Float32:
		bs := unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data)*4)
		if err := t.buf.Write(bs); err != nil {
			_ = t.Close()
			return nil, err
		}
	case Float16:
		packed := PackFP16(data)
		bs := uint16SliceAsBytes(packed)
		if err := t.buf.Write(bs); err != nil {
			_ = t.Close()
			return nil, err
		}
	case BFloat16:
		packed := PackBF16(data)
		bs := uint16SliceAsBytes(packed)
		if err := t.buf.Write(bs); err != nil {
			_ = t.Close()
			return nil, err
		}
	default:
		_ = t.Close()
		return nil, errors.New("unsupported dtype for FromFloat32")
	}
	return t, nil
}

func (t *Tensor) Numel() int { return Numel(t.Shape) }

// ByteSize returns the bytes occupied by this view (not necessarily the buffer size).
func (t *Tensor) ByteSize() int { return BytesFor(t.DT, t.Numel()) }

func (t *Tensor) Buffer() *metal.Buffer { return t.buf }

// Close releases the underlying buffer if owned.
func (t *Tensor) Close() error {
	if t == nil {
		return nil
	}
	if t.own && t.buf != nil {
		err := t.buf.Close()
		t.buf = nil
		t.own = false
		return err
	}
	return nil
}

// Contiguous reports whether the tensor is row-major contiguous view.
func (t *Tensor) Contiguous() bool {
	if t == nil {
		return false
	}
	want := DefaultStridesBytes(t.DT, t.Shape)
	if len(want) != len(t.Strides) {
		return false
	}
	for i := range want {
		if want[i] != t.Strides[i] {
			return false
		}
	}
	return t.Offset == 0
}

// Reshape returns a new view with the same storage.
func (t *Tensor) Reshape(newShape ...int) (*Tensor, error) {
	if t == nil {
		return nil, errors.New("nil tensor")
	}
	if !IsValidShape(newShape) {
		return nil, errors.New("invalid shape")
	}
	if Numel(newShape) != t.Numel() {
		return nil, errors.New("reshape changes numel")
	}
	// Only allow reshape if contiguous view
	if !t.Contiguous() {
		return nil, errors.New("reshape requires contiguous tensor")
	}
	return &Tensor{DT: t.DT, Shape: append([]int(nil), newShape...), Strides: DefaultStridesBytes(t.DT, newShape), Offset: t.Offset, buf: t.buf, own: false}, nil
}

// View creates a byte-wise view into the same storage.
func (t *Tensor) View(offsetBytes int, shape, strides []int) (*Tensor, error) {
	if t == nil {
		return nil, errors.New("nil tensor")
	}
	if !IsValidShape(shape) {
		return nil, errors.New("invalid shape")
	}
	if len(shape) != len(strides) {
		return nil, errors.New("shape/strides rank mismatch")
	}
	// Basic bounds check: last element address must fit into buffer
	maxOff := offsetBytes
	if len(shape) > 0 {
		// conservative bound: offset + (shape[0]-1)*stride[0] + ... + elemSize
		for i := range shape {
			if shape[i] <= 0 {
				continue
			}
			maxOff += (shape[i] - 1) * strides[i]
		}
		maxOff += t.DT.SizeOf()
	}
	if maxOff > t.buf.Size() {
		return nil, errors.New("view out of bounds")
	}
	vv := &Tensor{DT: t.DT, Shape: append([]int(nil), shape...), Strides: append([]int(nil), strides...), Offset: offsetBytes, buf: t.buf, own: false}
	return vv, nil
}

// DownloadFloat32 downloads tensor contents into dst, converting types as needed.
func (t *Tensor) DownloadFloat32(dst []float32) error {
	if t == nil || t.buf == nil {
		return errors.New("nil tensor")
	}
	if t.Numel() != len(dst) {
		return errors.New("len(dst) mismatch")
	}
	switch t.DT {
	case Float32:
		bs := unsafe.Slice((*byte)(unsafe.Pointer(&dst[0])), len(dst)*4)
		return t.buf.Read(bs)
	case Float16:
		tmp := make([]uint16, len(dst))
		if err := t.buf.Read(uint16SliceAsBytes(tmp)); err != nil {
			return err
		}
		out := UnpackFP16(tmp)
		copy(dst, out)
		return nil
	case BFloat16:
		tmp := make([]uint16, len(dst))
		if err := t.buf.Read(uint16SliceAsBytes(tmp)); err != nil {
			return err
		}
		out := UnpackBF16(tmp)
		copy(dst, out)
		return nil
	default:
		return errors.New("unsupported dtype for DownloadFloat32")
	}
}

// ---------- Shape/stride helpers ----------

func IsValidShape(shape []int) bool {
	if len(shape) == 0 {
		return false
	}
	for _, d := range shape {
		if d <= 0 {
			return false
		}
	}
	return true
}

func Numel(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

// DefaultStridesBytes computes row-major strides in bytes.
func DefaultStridesBytes(dt DType, shape []int) []int {
	if len(shape) == 0 {
		return nil
	}
	elem := dt.SizeOf()
	// For Int4, element addressability is still 1 byte; we keep stride=1 byte at last dim.
	if dt == Int4 {
		elem = 1
	}
	rank := len(shape)
	strides := make([]int, rank)
	strides[rank-1] = elem
	for i := rank - 2; i >= 0; i-- {
		mul := shape[i+1]
		if dt == Int4 && i == rank-2 { // approximate; contiguous works for even sizes
			// keep simple; exact packing handled by BytesFor
		}
		strides[i] = strides[i+1] * mul
	}
	return strides
}

// ---------- FP16/BF16 conversion ----------

func PackBF16(src []float32) []uint16 {
	out := make([]uint16, len(src))
	for i, f := range src {
		bits := math.Float32bits(f)
		// Round-to-nearest-even: add 0x7FFF + lsb of top 16 bits
		lsb := (bits >> 16) & 1
		rounding := uint32(0x7FFF) + lsb
		out[i] = uint16((bits + rounding) >> 16)
	}
	return out
}

func UnpackBF16(src []uint16) []float32 {
	out := make([]float32, len(src))
	for i, b := range src {
		bits := uint32(b) << 16
		out[i] = math.Float32frombits(bits)
	}
	return out
}

// PackFP16 converts float32 slice to IEEE 754 half precision (round-to-nearest-even).
func PackFP16(src []float32) []uint16 {
	out := make([]uint16, len(src))
	for i, f := range src {
		out[i] = float32ToFloat16Bits(f)
	}
	return out
}

// UnpackFP16 converts IEEE 754 half to float32.
func UnpackFP16(src []uint16) []float32 {
	out := make([]float32, len(src))
	for i, h := range src {
		out[i] = float16BitsToFloat32(h)
	}
	return out
}

func float32ToFloat16Bits(f float32) uint16 {
	x := math.Float32bits(f)
	sign := uint16((x >> 16) & 0x8000)
	mant := x & 0x007fffff
	exp := (x >> 23) & 0xff
	if exp == 0xff { // Inf or NaN
		if mant != 0 { // NaN
			return uint16(sign | 0x7e00)
		}
		return uint16(sign | 0x7c00)
	}
	// normal/denorm
	exp32 := int(exp) - 127
	exp16 := exp32 + 15
	if exp16 >= 0x1f { // overflow -> Inf
		return uint16(sign | 0x7c00)
	}
	if exp16 <= 0 { // denorm or underflow
		if exp16 < -10 {
			return uint16(sign) // underflow to zero
		}
		// denorm: shift mantissa
		mant |= 0x00800000
		shift := uint32(14 - exp16)
		// round to nearest
		out := mant >> shift
		if (mant>>(shift-1))&1 == 1 {
			out += 1
		}
		return uint16(sign | uint16(out&0x03ff))
	}
	// normal case
	outExp := uint16(exp16) & 0x1f
	outMant := uint16(mant >> 13)
	// round
	if (mant>>12)&1 == 1 {
		outMant += 1
	}
	if outMant >= 0x0400 { // mantissa overflow
		outMant = 0
		outExp += 1
		if outExp >= 0x1f { // overflow to Inf
			return uint16(sign | 0x7c00)
		}
	}
	return uint16(sign | (outExp << 10) | (outMant & 0x03ff))
}

func float16BitsToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := (h >> 10) & 0x1f
	mant := uint32(h & 0x03ff)
	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		// subnormal
		e := int32(-14)
		m := float64(float32(mant) / 1024.0)
		f := math.Ldexp(m, int(e))
		if sign != 0 {
			f = -f
		}
		return float32(f)
	}
	if exp == 0x1f {
		// Inf/NaN
		if mant == 0 {
			return math.Float32frombits(sign | 0x7f800000)
		}
		return math.Float32frombits(sign | 0x7fc00000)
	}
	// normalized
	e := int32(exp) - 15 + 127
	bits := sign | (uint32(e) << 23) | (mant << 13)
	return math.Float32frombits(bits)
}

func uint16SliceAsBytes(s []uint16) []byte {
	if len(s) == 0 {
		return nil
	}
	hdr := unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)*2)
	return hdr
}
