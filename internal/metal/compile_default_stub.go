//go:build !darwin || !cgo

package metal

// CompileDefault is a no-op on unsupported platforms.
func CompileDefault(kernelName string) {}

// CompileLibrary is a no-op on unsupported platforms.
func CompileLibrary() {}