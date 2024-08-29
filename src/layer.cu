/* Last Updated: 24.08.27. 18:30 */
#include "layer.h"

#include <cstdio>
#include <mma.h>
using namespace nvcuda;
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32
#define BLOCK_SIZE 32
#define NUM_WARP ((WMMA_M * WMMA_N) / (WARP_SIZE))
#define C_LAYOUT wmma::mem_row_major

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

/* Linear
 * @param [in1]  in: [M, K]
 * @param [in2]   w: [N, K]
 * @param [in3]   b: [N]
 * @param [out] out: [M, N]
 */
void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t M = out->shape[0];
  size_t N = out->shape[1];
  size_t K = w->shape[1];

  for (size_t m = 0; m < M; m++) {
    for (size_t n = 0; n < N; n++) {
      out->buf[m * N + n] = 0;
      for (size_t k = 0; k < K; k++) {
        out->buf[m * N + n] += in->buf[m * K + k] * w->buf[n * K + k];
      }
      out->buf[m * N + n] += b->buf[n];
    }
  }
}
static __global__ void Linear_tensor_kernel(half *in, half *w, half *b, half *out, 
                              size_t M, size_t N, size_t K) {
  int gj = blockIdx.x;
  int gi = blockIdx.y;
  if (gi * BLOCK_SIZE >= M || gj * BLOCK_SIZE >= N) return;  // boundary check
  int lj = threadIdx.x;
  int li = threadIdx.y;
  int warpId = li;

  __shared__ half Alocal[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ half Blocal[BLOCK_SIZE * BLOCK_SIZE];

    // Declare the fragments
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);
  
  int A_row_index = (gi * BLOCK_SIZE + li);
  int B_col_index = (gj * BLOCK_SIZE + lj);

  for (int bk = 0; bk < K; bk += BLOCK_SIZE) {
    
    for(int offset = 0 ; offset < NUM_WARP ; ++offset){
      int A_col_index = bk + lj;
      Alocal[(li + offset * blockDim.y) * BLOCK_SIZE + lj] = 
        ((A_row_index + offset * blockDim.y) < M && A_col_index < K)
        ? in[(A_row_index + offset * blockDim.y) * K + A_col_index]
        : (half)(0.0);

      int B_row_index = bk + li + (offset * blockDim.y);
      Blocal[(li + offset * blockDim.y) * BLOCK_SIZE + lj] = 
      (B_row_index < K && B_col_index < N)
        ? w[B_row_index + B_col_index * K]
        : (half)(0.0);  
    }
    __syncthreads();

    for (int i = 0; i < BLOCK_SIZE; i += WMMA_K) {
      int aCol = i;
      int aRow = (warpId / 2) * WMMA_M;
      int bCol = (warpId % 2) * WMMA_N;
      int bRow = i;

      wmma::load_matrix_sync(a_frag, Alocal + aCol + aRow * BLOCK_SIZE, BLOCK_SIZE);
      wmma::load_matrix_sync(b_frag, Blocal + bCol + bRow * BLOCK_SIZE, BLOCK_SIZE);
      
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    __syncthreads();
  }

  int cRow = (warpId / 2) * WMMA_M + blockIdx.y * blockDim.y * NUM_WARP;
  int cCol = (warpId % 2) * WMMA_N + blockIdx.x * blockDim.x;

  if(cRow + WMMA_M <= M && cCol + WMMA_N <= N){
    float * temp;
    wmma::store_matrix_sync(out + cCol + cRow * N, c_frag, N, C_LAYOUT);
  }

  half bias = b[B_col_index];
  for(int offset = 0 ; offset < NUM_WARP ; ++offset){
    int C_row_index = A_row_index + offset * blockDim.y;
    if (C_row_index < M && B_col_index < N) {
      out[C_row_index * N + B_col_index] += bias;
    }
  }
}
__global__ void Linear_kernel(half *in, half *w, half *b, half *out, 
                              size_t M, size_t N, size_t K) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  if (i >= N || j >= M) return;
  
  out[j * N + i] = 0;
  for (size_t k = 0; k < K; k++) {
    out[j * N + i] += in[j * K + k] * w[i * K + k];
  }
  out[j * N + i] += b[i];
}
void Linear_cuda(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t M = out->shape[0];
  size_t N = out->shape[1];
  size_t K = w->shape[1];

  half *in_gpu, *w_gpu, *b_gpu, *out_gpu;
  CHECK_CUDA(cudaMalloc(&in_gpu, M * K * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&w_gpu, N * K * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&b_gpu, N * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&out_gpu, M * N * sizeof(half)));

  CHECK_CUDA(cudaMemcpy(in_gpu, in->buf, M * K * sizeof(half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(w_gpu, w->buf, N * K * sizeof(half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(b_gpu, b->buf, N * sizeof(half), cudaMemcpyHostToDevice));

  int block_size = 32;
  // dim3 blockDim(block_size, block_size);
  // dim3 gridDim((N+block_size-1)/block_size, (M+block_size-1)/block_size);
  // Linear_kernel<<<gridDim, blockDim>>>(in_gpu, w_gpu, b_gpu, out_gpu, M, N, K);
  dim3 blockDim(block_size, 4);
  dim3 gridDim((N+block_size-1)/block_size, (M+block_size-1)/block_size);
  Linear_tensor_kernel<<<gridDim, blockDim>>>(in_gpu, w_gpu, b_gpu, out_gpu, M, N, K);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(out->buf, out_gpu, M * N * sizeof(half), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(in_gpu));
  CHECK_CUDA(cudaFree(w_gpu));
  CHECK_CUDA(cudaFree(b_gpu));
  CHECK_CUDA(cudaFree(out_gpu));
}

/* Reshape 
 * @param [in]   in: [N, D]
 * @param [out] out: [N, C, H, W]
 * 'N' is the number of input tensors.
 * 'D' is the dimension of the input tensor.
 * 'C' is the number of channels.
 * 'H' is the height of the output tensor.
 * 'W' is the width of the output tensor.
 */
void Reshape(Tensor *in, Tensor *out) {
  size_t N = in->shape[0];
  size_t D = in->shape[1];
  size_t C = out->shape[1];
  size_t H = out->shape[2];
  size_t W = out->shape[3];

  for (size_t n = 0; n < N; n++) {
    for (size_t c = 0; c < C; c++) {
      for (size_t h = 0; h < H; h++) {
        for (size_t w = 0; w < W; w++) {
          out->buf[n * C * H * W + c * H * W + h * W + w] =
              in->buf[n * D + c * H * W + h * W + w];
        }
      }
    }
  }
}

/* ConvTranspose2d
 * @param [in1]     in: [N, C, H, W]
 * @param [in2] weight: [C, K, R, S]
 * @param [in3]   bias: [K]
 * @param [out]    out: [N, K, OH, OW]
 *    
 *    OH = (H - 1) * stride - 2 * pad + dilation * (R - 1) + output_pad + 1
 *    OW = (W - 1) * stride - 2 * pad + dilation * (S - 1) + output_pad + 1
 *    In this model, R = S = 3, stride = 2, pad = 1, dilation = 1, output_pad = 1
 *
 * 'N' is the number of input tensors.
 * 'C' is the number of input channels.
 * 'H' is the height of the input tensor.
 * 'W' is the width of the input tensor.
 * 'K' is the number of output channels.
 * 'R' is the height of the filter.
 * 'S' is the width of the filter.
 * 'OH' is the height of the output tensor.
 * 'OW' is the width of the output tensor.
 */
void ConvTranspose2d(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out) {
  size_t C = in->shape[1];
  size_t H = in->shape[2];
  size_t W = in->shape[3];
  size_t K = weight->shape[1];
  size_t R = weight->shape[2];
  size_t S = weight->shape[3];
  size_t OH = out->shape[2];
  size_t OW = out->shape[3];
 
  const size_t stride = 2;
  const size_t pad = 1;
  const size_t dilation = 1;

#pragma omp parallel for
  for (size_t oc = 0; oc < K; ++oc) {
    for (size_t oh = 0; oh < OH; ++oh) {
      for (size_t ow = 0; ow < OW; ++ow) {
        half_cpu o = bias->buf[oc];
        for (size_t c = 0; c < C; ++c) {
          for (size_t r = 0; r < R; ++r) {
            for (size_t s = 0; s < S; ++s) {
              if ((oh - (r * dilation - pad)) % stride != 0) continue;
              if ((ow - (s * dilation - pad)) % stride != 0) continue;
              size_t h = (oh - (r * dilation - pad)) / stride;
              size_t w = (ow - (s * dilation - pad)) / stride;
              if (h >= H || w >= W) continue;
              o += in->buf[c * H * W + h * W + w] * 
                weight->buf[c * K * R * S + oc * R * S + r * S + s];
            }
          }
        }
        out->buf[oc * OH * OW + oh * OW + ow] = o;
      }
    }
  }
}
__global__ void ConvTranspose2d_kernel(half *in, half *weight, half *bias, half *out,
                      size_t C, size_t H, size_t W, 
                      size_t K, size_t R, size_t S, size_t OH, size_t OW, 
                      size_t stride, size_t pad, size_t dilation) {

  int ow = blockDim.x * blockIdx.x + threadIdx.x;
  int tidx = blockDim.y * blockIdx.y + threadIdx.y;
  const int oh = tidx % OH;
  const int k = tidx / OH;

  if (k >= K || oh >= OH || ow >= OW) return;

  half o = bias[k];
  for (size_t c = 0; c < C; ++c) {
    for (size_t r = 0; r < R; ++r) {
      for (size_t s = 0; s < S; ++s) {
        if ((oh - (r * dilation - pad)) % stride != 0) continue;
        if ((ow - (s * dilation - pad)) % stride != 0) continue;
        size_t h = (oh - (r * dilation - pad)) / stride;
        size_t w = (ow - (s * dilation - pad)) / stride;
        if (h >= H || w >= W) continue;
        o += in[c * H * W + h * W + w] * 
          weight[c * K * R * S + k * R * S + r * S + s];
      }
    }
  }
  out[k * OH * OW + oh * OW + ow] = o;

}
void ConvTranspose2d_cuda(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out) {
  size_t C = in->shape[1];
  size_t H = in->shape[2];
  size_t W = in->shape[3];
  size_t K = weight->shape[1];
  size_t R = weight->shape[2];
  size_t S = weight->shape[3];
  size_t OH = out->shape[2];
  size_t OW = out->shape[3];
 
  const size_t stride = 2;
  const size_t pad = 1;
  const size_t dilation = 1;

  half *in_gpu, *weight_gpu, *bias_gpu, *out_gpu;
  CHECK_CUDA(cudaMalloc(&in_gpu,      C * H * W * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&weight_gpu,  C * K * R * S * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&bias_gpu,    K * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&out_gpu,     K * OH * OW * sizeof(half)));

  CHECK_CUDA(cudaMemcpy(in_gpu,     in->buf,      C * H * W * sizeof(half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(weight_gpu, weight->buf,  C * K * R * S * sizeof(half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(bias_gpu,   bias->buf,    K * sizeof(half), cudaMemcpyHostToDevice));

  int block_size = 32;
  dim3 blockDim(block_size, block_size);
  dim3 gridDim((OW+block_size-1)/block_size, (K*OH+block_size-1)/block_size);
  ConvTranspose2d_kernel<<<gridDim, blockDim>>>(in_gpu, weight_gpu, bias_gpu, out_gpu,
                                                C, H, W, K, R, S, OH, OW,
                                                stride, pad, dilation);
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(out->buf, out_gpu, K * OH * OW * sizeof(half), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(in_gpu));
  CHECK_CUDA(cudaFree(weight_gpu));
  CHECK_CUDA(cudaFree(bias_gpu));
  CHECK_CUDA(cudaFree(out_gpu));
}

/* BatchNorm2d (track_running_stats=False)
 * @param [in1]     in: [N, C, H, W]
 * @param [in2] weight: [C]
 * @param [in3]   bias: [C]
 * @param [out]    out: [N, C, H, W]  
 * 
 *    out = weight * (in - mean) / sqrt(var + 1e-5) + bias 
 * 
 * 'N' is the number of input tensors.
 * 'C' is the number of channels.
 * 'H' is the height of the input tensor.
 * 'W' is the width of the input tensor.
 */
void BatchNorm2d(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out) {
  size_t C = in->shape[1];
  size_t H = in->shape[2];
  size_t W = in->shape[3];

  const float eps = 1e-5f;

  for (size_t c = 0; c < C; c++) {
    // 1. Caculate mean for each channel
    float mean = 0.0f;
    float var = 0.0f;
    for (size_t h = 0; h < H; h++) {
      for (size_t w = 0; w < W; w++) {
        half_cpu val = in->buf[c * H * W + h * W + w];
        mean += static_cast<float>(val); /* Cast to float */
      }
    }
    mean /= static_cast<float>(H * W);

    // 2. Caculate variance for each channel
    for (size_t h = 0; h < H; h++) {
      for (size_t w = 0; w < W; w++) {
        half_cpu val = in->buf[c * H * W + h * W + w];
        var += (static_cast<float>(val) - mean) * 
          (static_cast<float>(val) - mean); /* Cast to float */
      }
    }
    var /= static_cast<float>(H * W);

    // 3. Normalize with the calculated mean and variance
    for (size_t h = 0; h < H; h++) {
      for (size_t w = 0; w < W; w++) {
        out->buf[c * H * W + h * W + w] =
          weight->buf[c] * 
          (in->buf[c * H * W + h * W + w] - 
          half_cpu(mean)) / /* Cast to half */
          half_cpu(sqrt(var + eps)) + /* Cast to half */
          bias->buf[c];
      }
    }
  }
}
__global__ void BatchNorm2d_kernel(half *in, half *weight, half *bias, half *out,
                                    size_t C, size_t H, size_t W) {
  
  int c = blockDim.x * blockIdx.x + threadIdx.x;
  const float eps = 1e-5f;

  // 1. Caculate mean for each channel
  float mean = 0.0f;
  float var = 0.0f;
  for (size_t h = 0; h < H; h++) {
    for (size_t w = 0; w < W; w++) {
      half val = in[c * H * W + h * W + w];
      mean += static_cast<float>(val); /* Cast to float */
    }
  }
  mean /= static_cast<float>(H * W);

  // 2. Caculate variance for each channel
  for (size_t h = 0; h < H; h++) {
    for (size_t w = 0; w < W; w++) {
      half val = in[c * H * W + h * W + w];
      var += (static_cast<float>(val) - mean) * 
        (static_cast<float>(val) - mean); /* Cast to float */
    }
  }
  var /= static_cast<float>(H * W);

  // 3. Normalize with the calculated mean and variance
  for (size_t h = 0; h < H; h++) {
    for (size_t w = 0; w < W; w++) {
      out[c * H * W + h * W + w] =
        weight[c] * 
        (in[c * H * W + h * W + w] - 
        half(mean)) / /* Cast to half */
        half(sqrt(var + eps)) + /* Cast to half */
        bias[c];
    }
  }
}
void BatchNorm2d_cuda(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out) {
  size_t C = in->shape[1];
  size_t H = in->shape[2];
  size_t W = in->shape[3];

  half *in_gpu, *weight_gpu, *bias_gpu, *out_gpu;

  CHECK_CUDA(cudaMalloc(&in_gpu,      C * H * W * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&weight_gpu,  C * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&bias_gpu,    C * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&out_gpu,     C * H * W * sizeof(half)));

  CHECK_CUDA(cudaMemcpy(in_gpu,     in->buf,      C * H * W * sizeof(half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(weight_gpu, weight->buf,  C * sizeof(half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(bias_gpu,   bias->buf,    C * sizeof(half), cudaMemcpyHostToDevice));

  int block_size = 32;
  dim3 blockDim((C+block_size-1)/block_size);
  dim3 gridDim(block_size);
  BatchNorm2d_kernel<<<gridDim, blockDim>>>(in_gpu, weight_gpu, bias_gpu, out_gpu,
                                        C, H, W);

  CHECK_CUDA(cudaMemcpy(out->buf, out_gpu, C * H * W * sizeof(half), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(in_gpu));
  CHECK_CUDA(cudaFree(weight_gpu));
  CHECK_CUDA(cudaFree(bias_gpu));
  CHECK_CUDA(cudaFree(out_gpu));
}

/* LeakyReLU
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
void LeakyReLU(Tensor *inout) {
  size_t N = inout->num_elem();

  const half_cpu alpha = 0.01_h;

  for (size_t i = 0; i < N; i++) {
    if (inout->buf[i] < 0) { inout->buf[i] *= alpha; }
  }
}

/* LeakyReLU GPU kernel
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
__global__ void LeakyReLU_kernel(half *inout, size_t N, half alpha) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    if (inout[idx] < half(0)) { inout[idx] *= alpha; }
  }
}

/* LeakyReLU using CUDA GPU
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
void LeakyReLU_cuda(Tensor *inout) {
  size_t N = inout->num_elem();

  const half alpha = 0.01;
  
  half *d_inout;

  CHECK_CUDA(cudaMalloc(&d_inout, N * sizeof(half)));
  CHECK_CUDA(cudaMemcpy(d_inout, inout->buf, N * sizeof(half), cudaMemcpyHostToDevice));

  LeakyReLU_kernel<<<(N + 255) / 256, 256>>>(d_inout, N, alpha);
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(inout->buf, d_inout, N * sizeof(half), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_inout));
}

/* Conv2d
 * @param [in1]     in: [N, C, H, W]
 * @param [in2] weight: [K, C, R, S]
 * @param [in3]   bias: [K]
 * @param [out]    out: [N, K, OH, OW]
 *
 *   OH = (H + 2 * pad - dilation * (R - 1) - 1) / stride + 1
 *   OW = (W + 2 * pad - dilation * (S - 1) - 1) / stride + 1
 *   In this model, R = S = 3, stride = 1, pad = 1, dilation = 1
 *
 * 'N' is the number of input tensors.
 * 'C' is the number of input channels.
 * 'H' is the height of the input tensor.
 * 'W' is the width of the input tensor.
 * 'K' is the number of output channels.
 * 'R' is the height of the filter.
 * 'S' is the width of the filter.
 * 'OH' is the height of the output tensor.
 * 'OW' is the width of the output tensor.
 */
void Conv2d(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out) {
  size_t N = in->shape[0];
  size_t C = in->shape[1];
  size_t H = in->shape[2];
  size_t W = in->shape[3];
  size_t K = weight->shape[0];
  size_t R = weight->shape[2];
  size_t S = weight->shape[3];
  size_t OH = out->shape[2];
  size_t OW = out->shape[3];

  const size_t stride = 1;
  const size_t pad = 1;
  const size_t dilation = 1;

  for (size_t n = 0; n < N; n++) {
    for (size_t oc = 0; oc < K; oc++) {
      for (size_t oh = 0; oh < OH; oh++) {
        for (size_t ow = 0; ow < OW; ow++) {
          half_cpu o = bias->buf[oc];
          for (size_t c = 0; c < C; c++) {
            for (size_t r = 0; r < R; r++) {
              for (size_t s = 0; s < S; s++) {
                size_t h = oh * stride - pad + r * dilation;
                size_t w = ow * stride - pad + s * dilation;
                if (h >= H || w >= W) continue;
                o += in->buf[n * C * H * W + c * H * W + h * W + w] *
                  weight->buf[oc * C * R * S + c * R * S + r * S + s];
              }
            }
          }
          out->buf[n * K * OH * OW + oc * OH * OW + oh * OW + ow] = o;
        }
      }
    }
  }
}
__global__ void Conv2d_kernel(half *in, half *weight, half *bias, half *out,
                              size_t N, size_t C, size_t H, size_t W, 
                              size_t K, size_t R, size_t S, size_t OH, size_t OW, 
                              size_t stride, size_t pad, size_t dilation) {
  
  int w = blockDim.x * blockIdx.x + threadIdx.x;
  int tidx = blockDim.y * blockIdx.y + threadIdx.y;
  int temp = tidx;
  const int h = tidx % OH;
  temp /= OH;
  const int k = temp % K;
  const int n = temp / K;
  // int w = blockDim.x * blockIdx.x + threadIdx.x;
  // const int tidx = blockDim.y * blockIdx.y + threadIdx.y;
  // const int n = tidx / (K * OH);
  // const int k = (tidx / (OH)) % K;
  // const int h = (tidx) % OH;

  if (n >= N || k >= K || h >= OH || w >= OW) return;

  half sum = bias[k];
  for (int c = 0; c < C; ++c) {
    for (int r = 0; r < R; ++r) {
      for (int s = 0; s < S; ++s) {
        const int ih = h * stride - pad + r * dilation;
        const int iw = w * stride - pad + s * dilation;
        if (ih >= H || iw >= W) continue;
        sum += (in[((n * C + c) * H + ih) * W + iw]) 
                * (weight[((k * C + c) * R + r) * S + s]);
      }
    }
  }
  out[((n * K + k) * OH + h) * OW + w] = sum;
}
void Conv2d_cuda(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out) {
  size_t N = in->shape[0];
  size_t C = in->shape[1];
  size_t H = in->shape[2];
  size_t W = in->shape[3];
  size_t K = weight->shape[0];
  size_t R = weight->shape[2];
  size_t S = weight->shape[3];
  size_t OH = out->shape[2];
  size_t OW = out->shape[3];
  
  const size_t stride = 1;
  const size_t pad = 1;
  const size_t dilation = 1;
  
  half *in_gpu, *weight_gpu, *bias_gpu, *out_gpu;
  CHECK_CUDA(cudaMalloc(&in_gpu,      N * C * H * W * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&weight_gpu,  K * C * R * S * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&bias_gpu,    K * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&out_gpu,     N * K * OH * OW * sizeof(half)));

  CHECK_CUDA(cudaMemcpy(in_gpu,     in->buf,      N * C * H * W * sizeof(half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(weight_gpu, weight->buf,  K * C * R * S * sizeof(half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(bias_gpu,   bias->buf,    K * sizeof(half), cudaMemcpyHostToDevice));

  int block_size = 32;
  dim3 blockDim(block_size, block_size);
  dim3 gridDim((OW+block_size-1)/block_size, (N*K*OH+block_size-1)/block_size);
  Conv2d_kernel<<<gridDim, blockDim>>>(in_gpu, weight_gpu, bias_gpu, out_gpu,
                                        N, C, H, W, K, R, S, OH, OW,
                                        stride, pad, dilation);
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(out->buf, out_gpu, N * K * OH * OW * sizeof(half), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(in_gpu));
  CHECK_CUDA(cudaFree(weight_gpu));
  CHECK_CUDA(cudaFree(bias_gpu));
  CHECK_CUDA(cudaFree(out_gpu));
} 

/* Tanh 
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
void Tanh(Tensor *inout) {
  size_t N = inout->num_elem();

  for (size_t i = 0; i < N; i++) {
    inout->buf[i] = tanh(inout->buf[i]);
  }
}

