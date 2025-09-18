package main

import (
	"fmt"
	"math"

	"kylesmith19091/fastgo/internal/metal"
	"kylesmith19091/fastgo/internal/tensor"
)

func main() {
	const (
		batch = 2
		rows  = 3
		inner = 4
		cols  = 5
	)

	tA, err := tensor.NewRandom3D(tensor.Float32, batch, rows, inner)
	if err != nil {
		fmt.Printf("A alloc/init error: %v\n", err)
		return
	}
	defer tA.Close()

	tB, err := tensor.NewRandom3D(tensor.Float32, batch, inner, cols)
	if err != nil {
		fmt.Printf("B alloc/init error: %v\n", err)
		return
	}
	defer tB.Close()

	hA := make([]float32, tA.Numel())
	if err := tA.DownloadFloat32(hA); err != nil {
		fmt.Printf("A download error: %v\n", err)
		return
	}

	hB := make([]float32, tB.Numel())
	if err := tB.DownloadFloat32(hB); err != nil {
		fmt.Printf("B download error: %v\n", err)
		return
	}

	tC, err := tensor.New(tensor.Float32, batch, rows, cols)
	if err != nil {
		fmt.Printf("C alloc error: %v\n", err)
		return
	}
	defer tC.Close()

	metal.CompileLibrary()
	metal.EnsureKernel("matrix_multiply_batched_naive")

	if err := metal.MatMulBatchedBuffers(tA.Buffer(), tB.Buffer(), tC.Buffer(), batch, rows, inner, cols); err != nil {
		fmt.Printf("kernel error: %v\n", err)
		return
	}

	gpuOut := make([]float32, tC.Numel())
	if err := tC.DownloadFloat32(gpuOut); err != nil {
		fmt.Printf("download error: %v\n", err)
		return
	}

	hostRef := batchedMatMulCPU(batch, rows, inner, cols, hA, hB)
	maxErr := maxAbsDiff(gpuOut, hostRef)

	preview := len(gpuOut)
	if preview > 10 {
		preview = 10
	}
	fmt.Printf("Batched MatMul: B=%d M=%d K=%d N=%d\n", batch, rows, inner, cols)
	fmt.Printf("First few GPU C values: %v\n", gpuOut[:preview])
	fmt.Printf("First few CPU Cref values: %v\n", hostRef[:preview])
	fmt.Printf("Max |GPU-CPU| error: %g\n", maxErr)
}

func batchedMatMulCPU(batches, rows, inner, cols int, a, b []float32) []float32 {
	c := make([]float32, batches*rows*cols)
	batchStrideA := rows * inner
	batchStrideB := inner * cols
	batchStrideC := rows * cols

	for batch := 0; batch < batches; batch++ {
		aBatch := a[batch*batchStrideA : (batch+1)*batchStrideA]
		bBatch := b[batch*batchStrideB : (batch+1)*batchStrideB]
		cBatch := c[batch*batchStrideC : (batch+1)*batchStrideC]

		for r := 0; r < rows; r++ {
			rowOffset := r * inner
			for cIdx := 0; cIdx < cols; cIdx++ {
				sum := float32(0)
				for k := 0; k < inner; k++ {
					sum += aBatch[rowOffset+k] * bBatch[k*cols+cIdx]
				}
				cBatch[r*cols+cIdx] = sum
			}
		}
	}

	return c
}

func maxAbsDiff(a, b []float32) float32 {
	maxErr := float32(0)
	for i, v := range a {
		diff := float32(math.Abs(float64(v - b[i])))
		if diff > maxErr {
			maxErr = diff
		}
	}
	return maxErr
}
