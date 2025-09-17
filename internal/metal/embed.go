package metal

import _ "embed"

//go:embed kernels/mm.metal
var embeddedSource string

// Source returns the embedded Metal kernel source.
func Source() string { return embeddedSource }
