#include "javernn/synced_memory.h"
#include "javernn/util/cuda.h"

namespace javernn {
SyncedMemory::SyncedMemory()
    : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
    own_cpu_data_(false), own_gpu_data_(false) {
#ifdef GPU
  device_ = cuda_get_device();
#endif
}

SyncedMemory::SyncedMemory(size_t size)
  : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
    own_cpu_data_(false),  own_gpu_data_(false) {
#ifdef GPU
  device_ = cuda_get_device();
#endif
}

SyncedMemory::~SyncedMemory() {
  check_device();
  if (cpu_ptr_ && own_cpu_data_) {
    free(cpu_ptr_);
  }

#ifdef GPU
  if (gpu_ptr_ && own_gpu_data_) {
    cuda_free(gpu_ptr_);
  }
#endif  // GPU
}

inline void SyncedMemory::to_cpu() {
  check_device();
  switch (head_) {
  case UNINITIALIZED:
    cpu_ptr_ = malloc(size_);
    memset(cpu_ptr_, 0, size_ );
    head_ = JAVERNN_CPU;
    own_cpu_data_ = true;
    break;
  case JAVERNN_GPU:
#ifdef GPU
    if (cpu_ptr_ == NULL) {
      cuda_malloc(&cpu_ptr_, size_);
      own_cpu_data_ = true;
    }
    cuda_pull_array(gpu_ptr_, cpu_ptr_,size_);
    head_ = SYNCED;
#endif
    break;
  case JAVERNN_CPU:
  case SYNCED:
    break;
  }
}

inline void SyncedMemory::to_gpu() {
  check_device();
#ifdef GPU
  switch (head_) {
  case UNINITIALIZED:
    cuda_malloc(&gpu_ptr_, size_);
    cuda_memset(gpu_ptr_, 0, size_);
    head_ = JAVERNN_GPU;
    own_gpu_data_ = true;
    break;
  case JAVERNN_CPU:
    if (gpu_ptr_ == NULL) {
      cuda_malloc(&gpu_ptr_, size_);
      own_gpu_data_ = true;
    }
    cuda_push_array(gpu_ptr_, cpu_ptr_,size_);
    head_ = SYNCED;
    break;
  case JAVERNN_GPU:
  case SYNCED:
    break;
  }
#endif
}

const void* SyncedMemory::cpu_data() {
  check_device();
  to_cpu();
  return (const void*)cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data) {
  check_device();
  if(!data){
    return;
  }
  if (own_cpu_data_) {
    free(cpu_ptr_);
  }
  cpu_ptr_ = data;
  head_ = JAVERNN_CPU;
  own_cpu_data_ = false;
}

const void* SyncedMemory::gpu_data() {
  check_device();
#ifdef GPU
  to_gpu();
  return (const void*)gpu_ptr_;
#endif
}

void SyncedMemory::set_gpu_data(void* data) {
  check_device();
#ifdef GPU
  if(!data){
    return;
  }
  if (own_gpu_data_) {
    cudaFree(gpu_ptr_);
  }
  gpu_ptr_ = data;
  head_ = JAVERNN_GPU;
  own_gpu_data_ = false;
#endif
}

void* SyncedMemory::mutable_cpu_data() {
  check_device();
  to_cpu();
  head_ = JAVERNN_CPU;
  return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data() {
  check_device();
#ifdef GPU
  to_gpu();
  head_ = JAVERNN_GPU;
  return gpu_ptr_;
#endif
  return NULL;
}

#ifdef GPU
void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {
  check_device();
  if(head_ != JAVERNN_CPU){
    return;
  }
  if (gpu_ptr_ == NULL) {
    cuda_malloc(&gpu_ptr_, size_);
    own_gpu_data_ = true;
  }
  cuda_async_push_array(gpu_ptr_,cpu_ptr_,size_,stream);
  // Assume caller will synchronize on the stream before use
  head_ = SYNCED;
}
#endif

void SyncedMemory::check_device() {
#ifdef GPU
#ifdef DEBUG
  int device = cuda_get_device();
  if(device == device_){

  }else{
    
  }
  if (gpu_ptr_ && own_gpu_data_) {
    cudaPointerAttributes attributes;
    cuda_pointer_get_attributes(&attributes,gpu_ptr_);
    if(attributes.device == device_){

    }else{

    }
  }
#endif
#endif
}

}  // namespace caffe

