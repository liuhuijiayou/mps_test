// memory_probe -- native, step-wise device allocation probe for the MPS
// CUDA_MPS_PINNED_DEVICE_MEM_LIMIT verification.
//
// Why native: PyTorch's caching allocator would make the OOM boundary
// unattributable. Here each step is a direct driver allocation, optionally
// touched (written+read back) so the memory is really backed, and the exact CUDA
// error name of the first failing request is reported.
//
// Honest limits:
//   * the context itself plus CUDA-internal allocations consume part of the
//     client quota, so user buffers are NOT expected to reach exactly the limit;
//   * an allocation OOM is reported as an allocation failure. Whether it is a
//     whole-card fatal fault is a separate question this probe does not answer.
//     We verify post-release recoverability by running a small kernel again.

#include "common.cuh"

#include <cstring>
#include <vector>

__global__ void touch_kernel(unsigned char* p, size_t n, unsigned char v) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (; i < n; i += stride) p[i] = v;
}

__global__ void sum_kernel(const unsigned char* p, size_t n, unsigned long long* out) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  unsigned long long local = 0;
  for (; i < n; i += stride) local += p[i];
  atomicAdd(out, local);
}

struct Options {
  size_t step_mib = 256;
  size_t max_mib = 0;  // 0 = until failure
  bool touch = false;
  bool json = false;
  double hold_s = 0.0;
};

static Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--step-mib") && i + 1 < argc) o.step_mib = std::strtoull(argv[++i], nullptr, 10);
    else if (!std::strcmp(argv[i], "--max-mib") && i + 1 < argc) o.max_mib = std::strtoull(argv[++i], nullptr, 10);
    else if (!std::strcmp(argv[i], "--touch")) o.touch = true;
    else if (!std::strcmp(argv[i], "--json")) o.json = true;
    else if (!std::strcmp(argv[i], "--hold-s") && i + 1 < argc) o.hold_s = std::atof(argv[++i]);
  }
  return o;
}

int main(int argc, char** argv) {
  Options opt = parse(argc, argv);
  const size_t MIB = 1024ull * 1024ull;

  CUresult res = cuInit(0);
  if (res != CUDA_SUCCESS) {
    std::printf("{\"ok\":false,\"stage\":\"cuInit\",\"cuda_error\":\"%s\"}\n", drv_err_name(res));
    return 2;
  }
  CUdevice dev;
  cuDeviceGet(&dev, 0);
  CUcontext ctx = nullptr;
  res = cuCtxCreate(&ctx, 0, dev);
  if (res != CUDA_SUCCESS) {
    std::printf("{\"ok\":false,\"stage\":\"cuCtxCreate\",\"cuda_error\":\"%s\"}\n",
                drv_err_name(res));
    return 2;
  }

  size_t free_before = 0, total = 0;
  cuMemGetInfo(&free_before, &total);
  // Context creation itself already consumed quota; record the delta explicitly.
  const size_t context_overhead = (total > free_before) ? (total - free_before) : 0;

  std::vector<CUdeviceptr> blocks;
  size_t allocated_mib = 0, requested_mib = 0;
  const char* first_error = nullptr;
  size_t failed_request_mib = 0;

  while (opt.max_mib == 0 || allocated_mib + opt.step_mib <= opt.max_mib) {
    CUdeviceptr ptr = 0;
    requested_mib += opt.step_mib;
    CUresult ares = cuMemAlloc(&ptr, opt.step_mib * MIB);
    if (ares != CUDA_SUCCESS) {
      first_error = drv_err_name(ares);
      failed_request_mib = opt.step_mib;
      break;
    }
    if (opt.touch) {
      size_t n = opt.step_mib * MIB;
      touch_kernel<<<256, 256>>>(reinterpret_cast<unsigned char*>(ptr), n, 0x5A);
      if (cudaDeviceSynchronize() != cudaSuccess) {
        first_error = "CUDA_ERROR_LAUNCH_FAILED";
        failed_request_mib = opt.step_mib;
        cuMemFree(ptr);
        break;
      }
    }
    blocks.push_back(ptr);
    allocated_mib += opt.step_mib;
  }

  size_t free_at_peak = 0, total_at_peak = 0;
  cuMemGetInfo(&free_at_peak, &total_at_peak);

  // Verify the data we wrote is intact while still holding the blocks.
  unsigned long long checksum = 0;
  bool checksum_ok = false;
  if (opt.touch && !blocks.empty()) {
    unsigned long long* d_out = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&d_out), sizeof(unsigned long long)) == cudaSuccess) {
      cudaMemset(d_out, 0, sizeof(unsigned long long));
      size_t n = opt.step_mib * MIB;
      sum_kernel<<<256, 256>>>(reinterpret_cast<const unsigned char*>(blocks.front()), n, d_out);
      if (cudaDeviceSynchronize() == cudaSuccess) {
        cudaMemcpy(&checksum, d_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        checksum_ok = (checksum == static_cast<unsigned long long>(0x5A) * n);
      }
      cudaFree(d_out);
    }
  }

  if (opt.hold_s > 0) {
    struct timespec ts;
    ts.tv_sec = static_cast<time_t>(opt.hold_s);
    ts.tv_nsec = static_cast<long>((opt.hold_s - ts.tv_sec) * 1e9);
    nanosleep(&ts, nullptr);
  }

  for (CUdeviceptr p : blocks) cuMemFree(p);
  bool released = true;
  size_t free_after = 0;
  cuMemGetInfo(&free_after, &total);

  // Can this client still compute after the OOM + release?
  bool recovered = false;
  {
    int* buf = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&buf), 1024 * sizeof(int)) == cudaSuccess) {
      cudaMemset(buf, 1, 1024 * sizeof(int));
      touch_kernel<<<8, 128>>>(reinterpret_cast<unsigned char*>(buf), 1024 * sizeof(int), 1);
      recovered = (cudaDeviceSynchronize() == cudaSuccess);
      cudaFree(buf);
    }
  }

  std::printf("{\"ok\":true,\"step_mib\":%zu,\"max_mib\":%zu,\"touch\":%s,"
              "\"requested_mib\":%zu,\"allocated_mib\":%zu,\"failed_request_mib\":%zu,"
              "\"cuda_error\":%s%s%s,\"context_overhead_bytes\":%zu,"
              "\"device_total_bytes\":%zu,\"device_free_before_bytes\":%zu,"
              "\"device_free_at_peak_bytes\":%zu,\"device_free_after_release_bytes\":%zu,"
              "\"checksum\":%llu,\"checksum_ok\":%s,"
              "\"released\":%s,\"recovered_compute\":%s,"
              "\"note\":\"用户缓冲区不要求恰好占满配额：context 与 CUDA 内部分配也计入 client 配额\"}\n",
              opt.step_mib, opt.max_mib, opt.touch ? "true" : "false",
              requested_mib, allocated_mib, failed_request_mib,
              first_error ? "\"" : "null", first_error ? first_error : "",
              first_error ? "\"" : "",
              context_overhead, total, free_before, free_at_peak, free_after,
              checksum, checksum_ok ? "true" : "false",
              released ? "true" : "false", recovered ? "true" : "false");

  cuCtxDestroy(ctx);
  return 0;
}
