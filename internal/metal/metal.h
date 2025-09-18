// Matches with MatrixParams type in metal.go
typedef struct MatrixParams {
  int a_rows, a_cols;
  int b_rows, b_cols;
} MatrixParams;

// Initializes library + queue for a single kernel (back-compat)
void initializePipelineAndCommandQueue(char* source_path, char* kernel_name);

// Generic initialization and multi-kernel support
void initializeLibrary(char* source_path);
void ensurePipelineFor(char* kernel_name);

void initializeMTLBuffers(
  void* a,
  void* b,
  int data_size_bytes,
  int a_array_size,
  int b_array_size,
  int out_array_size
);

void* metal_mult_naive(MatrixParams *params);
void* mps_mult(MatrixParams *params);

// Generic buffer/IO helpers for tensors
void* mtl_new_buffer(int length_bytes);
void  mtl_release_buffer(void* buf);
void  mtl_buffer_write(void* buf, void* src, int length_bytes);
void  mtl_buffer_read(void* buf, void* dst, int length_bytes);
// Read from buffer starting at byte offset into dst for length_bytes.
void  mtl_buffer_read_at(void* buf, int offset_bytes, void* dst, int length_bytes);

// Kernel invocation using provided buffers (2D naive)
void* metal_mult_naive_with_buffers(MatrixParams *params, void* bufA, void* bufB, void* bufC);

// Generic named-kernel runner for up to 3 buffers and explicit grid sizes.
void* mtl_run_kernel_named_3(
  char* kernel_name,
  void* params,
  int params_len,
  void* buf0,
  void* buf1,
  void* buf2,
  int gridX,
  int gridY,
  int gridZ
);