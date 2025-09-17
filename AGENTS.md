# Repository Guidelines

## Project Structure & Module Organization
- `cmd/fastgo/main.go`: CLI entry; compiles kernels and runs example output.
- `internal/metal/metal.go` (`darwin && cgo`): Go↔C bridge and helpers.
- `internal/metal/metal_stub.go` (`!darwin || !cgo`): Cross‑platform no‑ops.
- `internal/metal/metal.m` + `internal/metal/metal.h`: Objective‑C host + C header.
- `internal/metal/kernels/mm.metal`: Metal Shading Language (naive matmul).
- `internal/metal/embed.go`: Embeds kernel source; `Source()` accessor.
- `internal/metal/compile_default_*.go`: `CompileDefault()` platform variants.

## Build, Test, and Development Commands
- Build (macOS): `go build ./...`
- Run example: `go run ./cmd/fastgo`
- Vet/format: `go vet ./...` and `gofmt -s -w .`
- Cross‑platform (no Metal): `CGO_ENABLED=0 go build` (stubs; no GPU execution).

Requirements: macOS with Xcode Command Line Tools for Metal frameworks.

## Coding Style & Naming Conventions
- Go formatting via `gofmt -s -w .` (tabs, idiomatic Go).
- Build tags: `//go:build darwin && cgo` for Metal code; complementary stubs.
- File naming: `cmd/<app>/main.go`, `internal/<pkg>`, platform suffixes `*_stub.go`.
- Keep cgo preambles directly above `import "C"` blocks.

## Testing Guidelines
- Use standard Go `testing` in `*_test.go`.
- Recommend verifying GPU results vs. CPU for small matrices (e.g., 2×3·3×2).
- Run tests: `go test ./...` (no tests yet; contributions welcome).

## Commit & Pull Request Guidelines
- Commits: imperative, concise subjects (e.g., "Add naive Metal matmul").
- PRs: include description, linked issues, and macOS build/run output.
- Pre‑submit checklist: `gofmt -s -w .`, `go vet ./...`, `go build ./...`, `go run ./cmd/fastgo`.

## Security & Configuration Tips
- Only embed trusted kernels under version control (`internal/metal/kernels`).
- For macOS runs keep `CGO_ENABLED=1`; frameworks `Metal`, `Foundation`, `MetalPerformanceShaders` must be available.
