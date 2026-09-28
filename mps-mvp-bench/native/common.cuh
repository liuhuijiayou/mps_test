// Shared helpers for the native probes.
//
// These probes intentionally use the CUDA *driver* API for allocation so the
// framework caching allocator never sits between us and the MPS client quota.
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>

inline const char* drv_err_name(CUresult res) {
  const char* name = nullptr;
  if (cuGetErrorName(res, &name) != CUDA_SUCCESS || name == nullptr) {
    return "CUDA_ERROR_UNKNOWN";
  }
  return name;
}

inline std::string json_escape(const std::string& in) {
  std::string out;
  for (char c : in) {
    if (c == '"' || c == '\\') {
      out.push_back('\\');
      out.push_back(c);
    } else if (c == '\n') {
      out += "\\n";
    } else {
      out.push_back(c);
    }
  }
  return out;
}

// Report the device UUID so the caller can verify the container really only sees
// the physical GPU the user named. Returns an empty string when unavailable
// (never a fabricated value).
inline std::string device_uuid(CUdevice dev) {
  CUuuid uuid{};
  if (cuDeviceGetUuid(&uuid, dev) != CUDA_SUCCESS) return std::string();
  char buf[64];
  const unsigned char* b = reinterpret_cast<const unsigned char*>(uuid.bytes);
  std::snprintf(buf, sizeof(buf),
                "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11],
                b[12], b[13], b[14], b[15]);
  return std::string(buf);
}
