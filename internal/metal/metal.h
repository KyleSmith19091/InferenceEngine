// Matches with MatrixParams type in metal.go
typedef struct MatrixParams {
  int a_rows, a_cols;
  int b_rows, b_cols;
} MatrixParams;

void initializePipelineAndCommandQueue(char* source_path, char* kernel_name);
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

// Kernel invocation using provided buffers
void* metal_mult_naive_with_buffers(MatrixParams *params, void* bufA, void* bufB, void* bufC);
