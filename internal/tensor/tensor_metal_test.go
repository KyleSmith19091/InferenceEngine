//go:build darwin && cgo

package tensor

import (
	"math"
	"testing"
)

func TestTensorUploadDownloadFloat32(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6}
    tt, err := FromFloat32(Float32, data, 2, 3)
    if err != nil {
        t.Skipf("skipping: Metal buffer unavailable: %v", err)
    }
	defer tt.Close()

	got := make([]float32, len(data))
	if err := tt.DownloadFloat32(got); err != nil {
		t.Fatalf("DownloadFloat32: %v", err)
	}
	for i := range data {
		if got[i] != data[i] {
			t.Fatalf("mismatch @%d: got %v want %v", i, got[i], data[i])
		}
	}
}

func TestTensorUploadDownloadFloat16(t *testing.T) {
	data := []float32{0.0, 1.0, -1.0, 0.5, -0.25, 3.14159}
    tt, err := FromFloat32(Float16, data, 2, 3)
    if err != nil {
        t.Skipf("skipping: Metal buffer unavailable: %v", err)
    }
	defer tt.Close()

	got := make([]float32, len(data))
	if err := tt.DownloadFloat32(got); err != nil {
		t.Fatalf("DownloadFloat32: %v", err)
	}
	for i := range data {
		if math.Abs(float64(got[i]-data[i])) > 1e-2 {
			t.Fatalf("fp16 mismatch @%d: got %v want ~%v", i, got[i], data[i])
		}
	}
}
