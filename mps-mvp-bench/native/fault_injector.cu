// fault_injector -- small standalone CUDA helper that triggers REAL device-side
// faults, so a CPU-side exception is never passed off as a device error.
//
// Modes:
//   illegal-access : genuine device illegal memory access (F7)
//   device-assert  : genuine device-side assert (F8). The build strips -DNDEBUG,
//                    and `--verify-assert-enabled` checks that at runtime.
//   long-kernel    : BOUNDED long-running kernel (F9). Never an infinite loop:
//                    the kernel exits after its own deadline, and the process has
//                    a watchdog on top.
//   api-error      : non-fatal CUDA API parameter error (F2), reported with its
//                    actual error type.
//   busy           : keeps GPU work in flight and publishes a device-side progress
//                    marker, so an external SIGKILL (F6) can be proven to have
//                    landed while work was submitted-but-incomplete -- rather than
//                    assuming "we slept, therefore the kernel ran".
//
// This process replaces one workload worker; it never adds a third GPU client.

#include "common.cuh"

#include <cassert>
#include <cstring>
#include <ctime>

__global__ void illegal_access_kernel(int* victim) {
  // Deliberately dereference an invalid device pointer.
  int* bad = reinterpret_cast<int*>(0xdeadbeef00ull);
  *bad = 42;
  if (victim) *victim = *bad;
}

__global__ void assert_kernel(int value) {
  // Real device assert. If NDEBUG were defined this would vanish, which is why
  // the build explicitly removes it and `--verify-assert-enabled` checks it.
  assert(value != 0);
}

// Bounded spin: stops on the clock deadline OR the iteration cap, whichever
// comes first. Writes a monotonically increasing progress marker so the host can
// prove work was actually executing on the device.
__global__ void bounded_long_kernel(long long clock_budget, long long iter_cap,
                                    volatile unsigned int* progress) {
  long long start = clock64();
  long long iters = 0;
  float acc = 1.0f;
  while (iters < iter_cap) {
    acc = acc * 1.0000001f + 1e-7f;
    ++iters;
    if ((iters & 0xFFFF) == 0) {
      if (threadIdx.x == 0 && blockIdx.x == 0 && progress) {
        atomicAdd(reinterpret_cast<unsigned int*>(const_cast<unsigned int*>(progress)), 1u);
      }
      if (clock64() - start > clock_budget) break;
    }
  }
  if (acc == 0.0f && progress) *const_cast<unsigned int*>(progress) = 0;  // keep acc live
}

static double now_s() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
  const char* mode = nullptr;
  double duration_s = 2.0;
  double watchdog_s = 30.0;
  bool verify_assert = false;
  const char* marker_path = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--mode") && i + 1 < argc) mode = argv[++i];
    else if (!std::strcmp(argv[i], "--duration-s") && i + 1 < argc) duration_s = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--watchdog-s") && i + 1 < argc) watchdog_s = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--verify-assert-enabled")) verify_assert = true;
    else if (!std::strcmp(argv[i], "--marker-path") && i + 1 < argc) marker_path = argv[++i];
  }
  if (mode == nullptr) {
    std::printf("{\"ok\":false,\"error\":\"--mode is required "
                "(illegal-access|device-assert|long-kernel|api-error|busy)\"}\n");
    return 64;
  }

#ifdef NDEBUG
  if (verify_assert) {
    std::printf("{\"ok\":false,\"error\":\"NDEBUG 已定义：device assert 会被编译掉，"
                "F8 用例无效。请检查构建配置\"}\n");
    return 65;
  }
#endif
  if (verify_assert) {
    std::printf("{\"assert_enabled\":true}\n");
  }

  if (cuInit(0) != CUDA_SUCCESS) {
    std::printf("{\"ok\":false,\"stage\":\"cuInit\"}\n");
    return 2;
  }
  CUdevice dev;
  cuDeviceGet(&dev, 0);
  CUcontext ctx = nullptr;
  CUresult cres = cuCtxCreate(&ctx, 0, dev);
  if (cres != CUDA_SUCCESS) {
    std::printf("{\"ok\":false,\"stage\":\"cuCtxCreate\",\"cuda_error\":\"%s\"}\n",
                drv_err_name(cres));
    return 2;
  }
  std::printf("{\"stage\":\"ready\",\"pid\":%d,\"device_uuid\":\"%s\",\"mode\":\"%s\"}\n",
              static_cast<int>(getpid()), device_uuid(dev).c_str(), mode);
  std::fflush(stdout);

  if (!std::strcmp(mode, "api-error")) {
    // Non-fatal, caught at API validation: report the ACTUAL error, not a guess.
    void* p = nullptr;
    cudaError_t e1 = cudaMalloc(&p, 0xFFFFFFFFFFFFFFFFull);
    cudaError_t e2 = cudaMemcpy(nullptr, nullptr, 16, cudaMemcpyDeviceToDevice);
    cudaError_t after = cudaGetLastError();
    std::printf("{\"ok\":true,\"mode\":\"api-error\",\"injected\":true,"
                "\"malloc_error\":\"%s\",\"memcpy_error\":\"%s\",\"sticky_error\":\"%s\","
                "\"fatal\":false}\n",
                cudaGetErrorName(e1), cudaGetErrorName(e2), cudaGetErrorName(after));
    cuCtxDestroy(ctx);
    return 0;
  }

  unsigned int* d_progress = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_progress), sizeof(unsigned int));
  cudaMemset(d_progress, 0, sizeof(unsigned int));

  if (!std::strcmp(mode, "illegal-access")) {
    illegal_access_kernel<<<1, 32>>>(nullptr);
    cudaError_t sync = cudaDeviceSynchronize();
    std::printf("{\"ok\":true,\"mode\":\"illegal-access\",\"injected\":%s,"
                "\"device_error\":\"%s\",\"note\":\"设备端真实错误，非 CPU 异常冒充\"}\n",
                sync != cudaSuccess ? "true" : "false", cudaGetErrorName(sync));
    return sync == cudaSuccess ? 1 : 0;
  }

  if (!std::strcmp(mode, "device-assert")) {
    assert_kernel<<<1, 32>>>(0);
    cudaError_t sync = cudaDeviceSynchronize();
    std::printf("{\"ok\":true,\"mode\":\"device-assert\",\"injected\":%s,"
                "\"device_error\":\"%s\"}\n",
                sync != cudaSuccess ? "true" : "false", cudaGetErrorName(sync));
    return sync == cudaSuccess ? 1 : 0;
  }

  if (!std::strcmp(mode, "long-kernel") || !std::strcmp(mode, "busy")) {
    // ~1.4 GHz SM clock assumption is only used to size the budget; the kernel
    // also has a hard iteration cap, and the host has a watchdog. Nothing here
    // can run forever.
    const long long clock_budget = static_cast<long long>(duration_s * 1.4e9);
    const long long iter_cap = 20000000000ll;
    double t_launch = now_s();
    bounded_long_kernel<<<64, 256>>>(clock_budget, iter_cap, d_progress);

    unsigned int seen = 0;
    bool progressed = false;
    // Poll the device progress marker: this is the evidence that work was
    // submitted AND executing, rather than merely slept over.
    while (now_s() - t_launch < watchdog_s) {
      unsigned int host_val = 0;
      cudaMemcpyAsync(&host_val, d_progress, sizeof(unsigned int), cudaMemcpyDeviceToHost, 0);
      // do not synchronize the whole device; peek via a separate stream-0 copy
      if (host_val > seen) {
        seen = host_val;
        progressed = true;
      }
      if (marker_path) {
        FILE* fh = std::fopen(marker_path, "w");
        if (fh) {
          std::fprintf(fh, "{\"progress\":%u,\"submitted\":true,\"completed\":false,"
                           "\"elapsed_s\":%.3f}\n", seen, now_s() - t_launch);
          std::fclose(fh);
        }
      }
      if (cudaStreamQuery(0) == cudaSuccess) break;
      struct timespec ts{0, 100000000L};
      nanosleep(&ts, nullptr);
    }
    cudaError_t sync = cudaDeviceSynchronize();
    double elapsed = now_s() - t_launch;
    std::printf("{\"ok\":true,\"mode\":\"%s\",\"injected\":true,"
                "\"kernel_elapsed_s\":%.3f,\"device_progress_observed\":%u,"
                "\"progress_proven\":%s,\"device_error\":\"%s\","
                "\"note\":\"有界 kernel：延迟/超时干扰，非硬件故障\"}\n",
                mode, elapsed, seen, progressed ? "true" : "false", cudaGetErrorName(sync));
    cudaFree(d_progress);
    cuCtxDestroy(ctx);
    return 0;
  }

  std::printf("{\"ok\":false,\"error\":\"unknown mode\"}\n");
  cuCtxDestroy(ctx);
  return 64;
}
