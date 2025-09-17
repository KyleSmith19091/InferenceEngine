// fastgo is a tiny example CLI that demonstrates
// basic usage of the internal tensor package.
// On macOS with cgo enabled it can exercise the
// Metal-backed implementation; otherwise stubs are used.
package main

import (
	"fmt"
	"kylesmith19091/fastgo/internal/tensor"
)

func main() {
	// Allocate a 3x3 Float32 tensor filled with random values.
	// Depending on build tags and platform, this may be backed by
	// GPU resources (Metal) or a CPU-only stub for portability.
	tA, err := tensor.NewRandom2D(tensor.Float32, 3, 3)
	if err != nil {
		// Propagate any allocation/initialization errors to the user.
		fmt.Println("tA error:", err)
		return
	}
	// Always release any underlying resources when done (GPU buffers, etc.).
	defer tA.Close()

	// Placeholder print to show the program ran; replace with real ops
	// (e.g., matmul, activation) or pretty-printing the tensor as needed.
	fmt.Printf("%v\n", tA.String())

}
