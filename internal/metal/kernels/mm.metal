// Mirror of the host-side MatrixParams; keep 32-bit ints and order.
typedef struct MatrixParams {
  int a_rows, a_cols;
  int b_rows, b_cols;
} MatrixParams;

// Params for batched 3D matmul: [B,M,K]x[B,K,N]->[B,M,N]
typedef struct MatMul3DParams {
  int batch, m, k, n;
} MatMul3DParams;

kernel void matrix_multiply_naive(
  device const MatrixParams *params,
  device const float *A,
  device const float *B,
  device float *C,
  uint2 gid [[thread_position_in_grid]]
) {
  // Expect grid = (b_cols, a_rows, 1)
  if (gid.x >= params->b_cols || gid.y >= params->a_rows) {
    return;
  }

  float sum = 0.0;
  int kk;
  // Loop unrolling; improves performance by a notable margin
  for (kk = 0; kk <= params->a_cols - 4; kk += 4) {
    sum += A[gid.y * params->a_cols + kk] 
       * B[kk * params->b_cols + gid.x];
    sum += A[gid.y * params->a_cols + kk + 1] 
       * B[(kk + 1) * params->b_cols + gid.x];
    sum += A[gid.y * params->a_cols + kk + 2] 
       * B[(kk + 2) * params->b_cols + gid.x];
    sum += A[gid.y * params->a_cols + kk + 3] 
       * B[(kk + 3) * params->b_cols + gid.x];
  }
  // Handle any remaining elements
  for (; kk < params->a_cols; ++kk) {
    sum += A[gid.y * params->a_cols + kk] * B[kk * params->b_cols + gid.x];
  }
  C[gid.y * params->b_cols + gid.x] = sum;
}

// Batched 3D matmul kernel: each thread computes one C[b, m, n]
kernel void matrix_multiply_batched_naive(
  device const MatMul3DParams *params,
  device const float *A,
  device const float *B,
  device float *C,
  uint3 gid [[thread_position_in_grid]]
) {
  // Expect grid = (n, m, batch)
  if (gid.x >= params->n || gid.y >= params->m || gid.z >= params->batch) {
    return;
  }
  const int b = gid.z;
  const int row = gid.y;
  const int col = gid.x;
  const int M = params->m;
  const int K = params->k;
  const int N = params->n;

  const int aBase = b * M * K;
  const int bBase = b * K * N;
  const int cBase = b * M * N;

  float sum = 0.0f;
  for (int kk = 0; kk < K; ++kk) {
    sum += A[aBase + row * K + kk] * B[bBase + kk * N + col];
  }
  C[cBase + row * N + col] = sum;
}