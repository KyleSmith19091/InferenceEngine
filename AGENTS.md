# Repository Guidelines

## Project Structure & Module Organization
- `cmd/fastgo/main.go`: CLI entry; compiles kernels and runs example output.
- `internal/metal/metal.go` (`darwin && cgo`): Go↔C bridge and helpers (generic multi‑kernel runner).
- `internal/metal/metal_stub.go` (`!darwin || !cgo`): Cross‑platform no‑ops.
- `internal/metal/metal.m` + `internal/metal/metal.h`: Objective‑C host + C header (library init, pipeline registry, generic runner).
- `internal/metal/kernels/mm.metal`: Metal Shading Language (2D and batched 3D matmul).
- `internal/metal/embed.go`: Embeds kernel source; `Source()` accessor.
- `internal/metal/compile_default_*.go`: `CompileDefault()` and `CompileLibrary()` platform variants.

## Build, Test, and Development Commands
- Build (macOS): `go build ./...`
- Run examples:
  - `go run ./examples/mm` (2D matmul)
  - `go run ./examples/batched_mm` (batched 3D matmul)
  - `go run ./examples/embedding` (toy layer)
- Vet/format: `go vet ./...` and `gofmt -s -w .`
- Cross‑platform (no Metal): `CGO_ENABLED=0 go build` (stubs; no GPU execution).

Requirements: macOS with Xcode Command Line Tools for Metal frameworks.

## Coding Style & Naming Conventions
- Go formatting via `gofmt -s -w .` (tabs, idiomatic Go).
- Build tags: `//go:build darwin && cgo` for Metal code; complementary stubs.
- File naming: `cmd/<app>/main.go`, `internal/<pkg>`, platform suffixes `*_stub.go`.
- Keep cgo preambles directly above `import "C"` blocks.

## Modern LLM Roadmap Expectations
- The repository is evolving into a GPU-first transformer inference engine targeting models such as Qwen3 and Gemma3.
- Align new features and fixes with the tasks tracked in `TODO.md`; update the list whenever scopes change.
- CPU reference paths are out of scope—focus effort on Metal kernels, GPU tensor ops, and related tooling.

## Agent Workflow Notes
- When editing Objective-C bridge files, immediately re-read the touched sections to ensure no diff markers (`+/-`) remain.
- After Objective-C or cgo changes, run `go build ./...` to confirm the bridge still compiles.

## Testing Guidelines
- Use standard Go `testing` in `*_test.go`.
- Recommend verifying GPU results vs. CPU for small matrices (e.g., 2×3·3×2) and small batches.
- Run tests: `go test ./...` (no tests yet; contributions welcome).

## Commit & Pull Request Guidelines
- Commits: imperative, concise subjects (e.g., "Add batched Metal matmul", "Expose generic kernel runner").
- PRs: include description, linked issues, and macOS build/run output.
- Pre‑submit checklist: `gofmt -s -w .`, `go vet ./...`, `go build ./...`, run one of the examples.

## Security & Configuration Tips
- Only embed trusted kernels under version control (`internal/metal/kernels`).
- For macOS runs keep `CGO_ENABLED=1`; frameworks `Metal`, `Foundation`, `MetalPerformanceShaders` must be available.