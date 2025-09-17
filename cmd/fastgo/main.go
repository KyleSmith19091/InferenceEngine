package main

import (
	"fmt"
	"kylesmith19091/fastgo/internal/metal"
	"kylesmith19091/fastgo/internal/tensor"
)

func main() {
	// compile kernel
	metal.CompileDefault("matrix_multiply_naive")

	// prepare data
	aRows, aCols := 2, 3
	bRows, bCols := 3, 2
	A := []float32{1, 2, 3, 4, 6, 6}
	B := []float32{7, 8, 9, 10, 11, 12}

	// allocate tensors on device and upload
	tA, err := tensor.FromFloat32(tensor.Float32, A, aRows, aCols)
	if err != nil {
		fmt.Println("tA error:", err)
		return
	}
	defer tA.Close()
	tB, err := tensor.FromFloat32(tensor.Float32, B, bRows, bCols)
	if err != nil {
		fmt.Println("tB error:", err)
		return
	}
	defer tB.Close()
	tC, err := tensor.New(tensor.Float32, aRows, bCols)
	if err != nil {
		fmt.Println("tC error:", err)
		return
	}
	defer tC.Close()

	// execute using tensor buffers
	if err := metal.MultiplyNaiveBuffers(tA.Buffer(), tB.Buffer(), tC.Buffer(), aRows, aCols, bRows, bCols); err != nil {
		fmt.Println("kernel error:", err)
		return
	}
	C := make([]float32, aRows*bCols)
	if err := tC.DownloadFloat32(C); err != nil {
		fmt.Println("download error:", err)
		return
	}

	fmt.Println("A:")
	fmt.Printf("%v\n%v\n", A[:aCols], A[aCols:])
	fmt.Println("B:")
	fmt.Printf("%v\n%v\n%v\n", B[:bCols], B[bCols:2*bCols], B[2*bCols:])
	fmt.Println("C = A x B:")
	fmt.Printf("%v\n%v\n", C[:bCols], C[bCols:])
}
