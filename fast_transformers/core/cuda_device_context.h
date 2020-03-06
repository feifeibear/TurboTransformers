#pragma once
#include <memory.h>

#include <map>

#include "fast_transformers/core/cuda_error.h"
#include "macros.h"

namespace fast_transformers {
namespace core {

class CublasHandleHolder {
 public:
  explicit CublasHandleHolder(cudaStream_t stream) {
    cublasCreate(&handle_);
    cublasSetStream(handle_, stream);
  }

  ~CublasHandleHolder() { cublasDestroy(handle_); }

  template <typename Callback>
  inline void Call(Callback&& callback) const {
    callback(handle_);
  }

 private:
  DISABLE_COPY_AND_ASSIGN(CublasHandleHolder);

  cublasHandle_t handle_;
};

class CUDADeviceContext {
 public:
  CUDADeviceContext();

  ~CUDADeviceContext();

  static CUDADeviceContext& GetInstance() {
    static CUDADeviceContext instance;
    return instance;
  }

  void Wait() const;

  template <typename Callback>
  inline void CublasCall(Callback&& callback) const {
    cublas_handle_->Call(std::forward<Callback>(callback));
  }

  cudaStream_t stream() const;

 private:
  cudaStream_t stream_;

  // TODO(jiaruifang)其实应该对cublas handle和cuda stream分别写两个deletor，然后用unique_ptr
  // 管理就好了。不用cublas的callback
  std::unique_ptr<CublasHandleHolder> cublas_handle_;

  DISABLE_COPY_AND_ASSIGN(CUDADeviceContext);
};

}  // namespace core
}  // namespace fast_transformers