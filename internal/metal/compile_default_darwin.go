//go:build darwin && cgo

package metal

// CompileDefault compiles the embedded Metal source on supported platforms.
func CompileDefault(kernelName string) { Compile(Source(), kernelName) }
