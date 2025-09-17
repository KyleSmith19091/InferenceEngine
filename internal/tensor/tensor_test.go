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
    // Avoid allocating a real Metal buffer in tests; construct a logical tensor.
    tt := &Tensor{
        DT:      Float32,
        Shape:   []int{2, 3},
        Strides: DefaultStridesBytes(Float32, []int{2, 3}),
        Offset:  0,
        buf:     nil,
        own:     false,
    }
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

    // Construct a manual non-contiguous view (transpose-like) without calling View
    v := &Tensor{DT: Float32, Shape: []int{3, 2}, Strides: []int{4, 12}, Offset: 0}
    if v.Contiguous() {
        t.Fatalf("expected non-contiguous view")
    }
}
