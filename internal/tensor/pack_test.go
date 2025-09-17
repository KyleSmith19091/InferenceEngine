package tensor

import (
	"math"
	"testing"
)

func TestBytesForAndSizeOf(t *testing.T) {
	if SizeOf(Float16) != 2 || SizeOf(BFloat16) != 2 || SizeOf(Float32) != 4 || SizeOf(Int8) != 1 {
		t.Fatalf("unexpected SizeOf results")
	}
	cases := []struct {
		dt   DType
		n    int
		want int
	}{
		{Float32, 1, 4}, {Float32, 7, 28},
		{Float16, 3, 6}, {BFloat16, 5, 10},
		{Int8, 5, 5},
		{Int4, 1, 1}, {Int4, 2, 1}, {Int4, 3, 2}, {Int4, 5, 3}, {Int4, 8, 4},
	}
	for _, c := range cases {
		got := BytesFor(c.dt, c.n)
		if got != c.want {
			t.Fatalf("BytesFor(%v,%d)=%d want %d", c.dt, c.n, got, c.want)
		}
	}
}

func TestDefaultStridesBytes(t *testing.T) {
	shape := []int{2, 3, 4}
	s32 := DefaultStridesBytes(Float32, shape)
	want32 := []int{48, 16, 4}
	for i := range s32 {
		if s32[i] != want32[i] {
			t.Fatalf("stride32[%d]=%d want %d", i, s32[i], want32[i])
		}
	}

	s16 := DefaultStridesBytes(Float16, shape)
	want16 := []int{24, 8, 2}
	for i := range s16 {
		if s16[i] != want16[i] {
			t.Fatalf("stride16[%d]=%d want %d", i, s16[i], want16[i])
		}
	}
}

func TestFP16BF16Conversions(t *testing.T) {
	vals := []float32{0, 1, -1, 0.5, -0.5, 65504, 1e-3, 1e-4}
	// BF16 should preserve high 16 bits exactly
	bf := PackBF16(vals)
	round := UnpackBF16(bf)
	for i := range vals {
		// BF16 precision depends on magnitude; use relative tolerance.
		diff := math.Abs(float64(vals[i] - round[i]))
		rel := 1e-3 * math.Max(1.0, math.Abs(float64(vals[i])))
		if diff > rel {
			t.Fatalf("bf16 roundtrip @%d: got %v want ~%v (diff=%g relTol=%g)", i, round[i], vals[i], diff, rel)
		}
	}

	// FP16 has less precision; allow larger epsilon
	hf := PackFP16(vals)
	round2 := UnpackFP16(hf)
	for i := range vals {
		if math.Abs(float64(vals[i]-round2[i])) > 1e-2 && !(math.IsInf(float64(round2[i]), 0) || math.IsInf(float64(vals[i]), 0)) {
			t.Fatalf("fp16 roundtrip @%d: got %v want ~%v", i, round2[i], vals[i])
		}
	}
}
