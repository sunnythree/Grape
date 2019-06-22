#ifndef _javernn_synced_memory_h_
#define _javernn_synced_memory_h_

#include <cstdint>
#include "javernn/util/cuda.h"

namespace javernn {
/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
class SyncedMemory {
public:
    SyncedMemory();
    explicit SyncedMemory(uint32_t size);
    ~SyncedMemory();
    const void* cpu_data();
    void set_cpu_data(void* data);
    const void* gpu_data();
    void set_gpu_data(void* data);
    void* mutable_cpu_data();
    void* mutable_gpu_data();
    enum SyncedHead { UNINITIALIZED, JAVERNN_CPU, JAVERNN_GPU, SYNCED };
    SyncedHead head() const { return head_; }
    uint32_t size() const { return size_; }

#ifdef GPU
  void async_gpu_push(const cudaStream_t& stream);
#endif

 private:
  void check_device();

  void to_cpu();
  void to_gpu();
  void* cpu_ptr_;
  void* gpu_ptr_;
  size_t size_;
  SyncedHead head_;
  bool own_cpu_data_;
  bool own_gpu_data_;
  int device_;
};  // class SyncedMemory

}  

#endif
