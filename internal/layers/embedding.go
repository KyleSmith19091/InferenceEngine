package layers

import (
	"kylesmith19091/fastgo/internal/tensor"
	"strings"
)

type Embedding struct {
	inner tensor.Tensor
}

func NewEmbedding(tensor tensor.Tensor) *Embedding {
	return &Embedding{
		inner: tensor,
	}
}

// Lookup returns the embedding vector for the given token id.
// Expects the underlying tensor to be shaped [vocab, dim] and returns
// a 1D view representing the row at index id.
func (e *Embedding) Lookup(id int) (*tensor.Tensor, error) {
	return e.inner.Row(id)
}

func (e *Embedding) String() string {
	var sb strings.Builder
	sb.WriteString("Embedding(\n")
	for idx := range e.inner.Shape[0] {
		t, err := e.inner.Row(idx)
		if err != nil {
			panic("could not print embedding due to row")
		}
		sb.WriteString(t.String())
		sb.WriteString("\n")
	}
	sb.WriteString(")")
	return sb.String()
}
