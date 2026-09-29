#include <unistd.h>
// device_probe -- proves, by execution, what preflight can only mark `unknown`:
//   * how many devices the container actually sees
//   * each device's physical UUID, total memory and SM count
//   * whether a CUDA context can be created and a trivial kernel completes
//   * which MPS client env vars this process received (recorded verbatim)
//
// Seeing the MPS daemon or finding a symbol proves nothing; this probe creating a
// context that appears in the MPS client list is what proves attachment.

#include "common.cuh"

#include <cstring>
#include <string>
#include <vector>

__global__ void add_one(int* data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] += 1;
}

static std::string env_or_null(const char* name) {
  const char* v = std::getenv(name);
  if (v == nullptr) return std::string("null");
  return std::string("\"") + json_escape(v) + "\"";
}

int main(int argc, char** argv) {
  bool hold = false;
  double hold_s = 0.0;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--hold-s") == 0 && i + 1 < argc) {
      hold = true;
      hold_s = std::atof(argv[++i]);
    }
  }

  CUresult res = cuInit(0);
  if (res != CUDA_SUCCESS) {
    std::printf("{\"ok\":false,\"stage\":\"cuInit\",\"cuda_error\":\"%s\"}\n", drv_err_name(res));
    return 2;
  }

  int count = 0;
  cuDeviceGetCount(&count);
  std::printf("{\"ok\":true,\"visible_device_count\":%d,", count);
  std::printf("\"env\":{\"CUDA_MPS_PIPE_DIRECTORY\":%s,"
              "\"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE\":%s,"
              "\"CUDA_MPS_CLIENT_PRIORITY\":%s,"
              "\"CUDA_MPS_PINNED_DEVICE_MEM_LIMIT\":%s,"
              "\"CUDA_VISIBLE_DEVICES\":%s},",
              env_or_null("CUDA_MPS_PIPE_DIRECTORY").c_str(),
              env_or_null("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE").c_str(),
              env_or_null("CUDA_MPS_CLIENT_PRIORITY").c_str(),
              env_or_null("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT").c_str(),
              env_or_null("CUDA_VISIBLE_DEVICES").c_str());
  std::printf("\"pid\":%d,\"devices\":[", static_cast<int>(getpid()));

  bool kernel_ok = false;
  for (int d = 0; d < count; ++d) {
    CUdevice dev;
    cuDeviceGet(&dev, d);
    char name[128] = {0};
    cuDeviceGetName(name, sizeof(name), dev);
    size_t total = 0;
    cuDeviceTotalMem(&total, dev);
    int sm = 0, major = 0, minor = 0;
    cuDeviceGetAttribute(&sm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    std::string uuid = device_uuid(dev);

    CUcontext ctx = nullptr;
    CUresult cres = cuCtxCreate(&ctx, 0, dev);
    const char* ctx_err = (cres == CUDA_SUCCESS) ? "CUDA_SUCCESS" : drv_err_name(cres);
    size_t free_b = 0, total_b = 0;
    if (cres == CUDA_SUCCESS) {
      cuMemGetInfo(&free_b, &total_b);
      int* buf = nullptr;
      const int n = 1024;
      if (cudaMalloc(reinterpret_cast<void**>(&buf), n * sizeof(int)) == cudaSuccess) {
        cudaMemset(buf, 0, n * sizeof(int));
        add_one<<<(n + 127) / 128, 128>>>(buf, n);
        // Blocking sync: an async launch returning is not a completed kernel.
        kernel_ok = (cudaDeviceSynchronize() == cudaSuccess);
        cudaFree(buf);
      }
      if (hold && hold_s > 0) {
        // Stay alive so the host can read the MPS client list while this context
        // exists (this is the actual attachment evidence).
        struct timespec ts;
        ts.tv_sec = static_cast<time_t>(hold_s);
        ts.tv_nsec = static_cast<long>((hold_s - ts.tv_sec) * 1e9);
        nanosleep(&ts, nullptr);
      }
      cuCtxDestroy(ctx);
    }

    std::printf("%s{\"index\":%d,\"name\":\"%s\",\"uuid\":\"%s\","
                "\"total_memory_bytes\":%zu,\"free_memory_bytes\":%zu,"
                "\"sm_count\":%d,\"capability\":\"%d.%d\",\"context\":\"%s\"}",
                d ? "," : "", d, json_escape(name).c_str(),
                uuid.empty() ? "" : uuid.c_str(), total, free_b, sm, major, minor, ctx_err);
  }
  std::printf("],\"kernel_ok\":%s}\n", kernel_ok ? "true" : "false");
  return kernel_ok ? 0 : 1;
}
