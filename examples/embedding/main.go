package main

import (
	"fmt"
	"kylesmith19091/fastgo/internal/layers"
	"kylesmith19091/fastgo/internal/tensor"
)

func main() {
	// Create a small [vocab, dim] embedding matrix
	vocab, dim := 5, 4
	weights, err := tensor.NewRandom2D(tensor.Float32, vocab, dim)
	if err != nil {
		fmt.Println("alloc error:", err)
		return
	}
	defer weights.Close()

	// Wrap in an Embedding layer and look up an id
	emb := layers.NewEmbedding(*weights)
	id := 2
	vec, err := emb.Lookup(id)
	if err != nil {
		fmt.Println("lookup error:", err)
		return
	}
	fmt.Println("Embedding: ", emb)

	// Print the 1D tensor view (values rendered by Tensor.String)
	fmt.Println("Embedding for id", id, ":", vec)

	// Access an element of the vector
	if val, err := vec.At(1); err == nil {
		fmt.Println("vec[1] =", val)
	}
}
