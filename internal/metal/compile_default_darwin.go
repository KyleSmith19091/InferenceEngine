//go:build darwin && cgo

package metal

// CompileDefault compiles the embedded Metal source on supported platforms.
func CompileDefault(kernelName string) { Compile(Source(), kernelName) }

// CompileLibrary compiles the embedded Metal source and initializes the library/queue
// without selecting a specific kernel; use EnsureKernel later per name.
func CompileLibrary() { CompileLibraryFrom(Source()) }