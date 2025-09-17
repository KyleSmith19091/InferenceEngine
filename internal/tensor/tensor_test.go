package tensor

import "testing"

func TestIsValidShapeAndNumel(t *testing.T) {
	if IsValidShape(nil) || IsValidShape([]int{}) || IsValidShape([]int{2, 0}) || !IsValidShape([]int{1}) {
		t.Fatalf("IsValidShape failed basic checks")
	}
	if Numel([]int{2, 3, 4}) != 24 {
		t.Fatalf("Numel wrong")
	}
}

func TestContiguousAndReshape(t *testing.T) {
	tt, err := New(Float32, 2, 3)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer tt.Close()
	if !tt.Contiguous() {
		t.Fatalf("expected contiguous")
	}

	r, err := tt.Reshape(3, 2)
	if err != nil {
		t.Fatalf("Reshape: %v", err)
	}
	if !r.Contiguous() {
		t.Fatalf("reshape should be contiguous")
	}

	// Make a manual non-contiguous view (transpose-like)
	// bytes per elem for float32 is 4; simulate strides for (2,3) transposed: [4, 8]
	v, err := tt.View(0, []int{3, 2}, []int{4, 12})
	if err != nil {
		t.Fatalf("View: %v", err)
	}
	if v.Contiguous() {
		t.Fatalf("expected non-contiguous view")
	}
}
