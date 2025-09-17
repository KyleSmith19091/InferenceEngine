package examples

import (
	"fmt"
	"kylesmith19091/fastgo/internal/metal"
	"kylesmith19091/fastgo/internal/tensor"
)

func main() {
	// compile kernel
	metal.CompileDefault("matrix_multiply_naive")

	// allocate tensors on device and upload
	tA, err := tensor.NewRandom2D(tensor.Float32, 3, 3)
	if err != nil {
		fmt.Println("tA error:", err)
		return
	}
	defer tA.Close()

	tB, err := tensor.NewRandom2D(tensor.Float32, 3, 3)
	if err != nil {
		fmt.Println("tB error:", err)
		return
	}
	defer tB.Close()

	tC, err := tensor.New(tensor.Float32, 3, 3)
	if err != nil {
		fmt.Println("tC error:", err)
		return
	}
	defer tC.Close()

	// execute using tensor buffers
	if err := metal.MultiplyNaiveBuffers(tA.Buffer(), tB.Buffer(), tC.Buffer(), 3, 3, 3, 3); err != nil {
		fmt.Println("kernel error:", err)
		return
	}
	C := make([]float32, 3*3)
	if err := tC.DownloadFloat32(C); err != nil {
		fmt.Println("download error:", err)
		return
	}

	fmt.Printf("%v\n", C)
}
