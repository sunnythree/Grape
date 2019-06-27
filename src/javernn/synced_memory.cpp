#include <stdlib.h>
#include <memory.h>
#include "javernn/synced_memory.h"
#include "javernn/util/cuda.h"
#include "javernn/error.h"
#include "javernn/global_config.h"
#include "javernn/log.h"

namespace javernn {

    static std::string TAG = "SyncedMemory";

    SyncedMemory::SyncedMemory(uint32_t size,CAL_MODE mode)
    :size_(size),mode_(mode) {
        if(mode_ == CPU_MODE){
            cpu_ptr_ = malloc(size_);
            memset(cpu_ptr_, 0, size_ );
        }else{
    #ifdef GPU
            device_ = cuda_get_device();
            if(device_>0 && gDeviceId < device_){
                cuda_set_device(gDeviceId);
            }else{
                throw new Error("no gpu device checked");
            }
            mode_ = GPU_MODE;
    #endif
        }
    }

    SyncedMemory::~SyncedMemory() {
        if (cpu_ptr_ != nullptr) {
            free(cpu_ptr_);
        }
    #ifdef GPU
        if (gpu_ptr_ != nullptr) {
            cuda_free(gpu_ptr_);
        }
    #endif  // GPU
    }

    void SyncedMemory::to_cpu() {
    switch (mode_) {
            case GPU_MODE:
    #ifdef GPU
            if (cpu_ptr_ == NULL) {
                cuda_malloc(&cpu_ptr_, size_);
            }
            cuda_pull_array(gpu_ptr_, cpu_ptr_,size_);
    #endif
            break;
        case CPU_MODE:
        break;
    }
    }

    void SyncedMemory::to_gpu() {
    #ifdef GPU
        switch (mode_) {
        case CPU_MODE:
            if (gpu_ptr_ == NULL) {
                cuda_malloc(&gpu_ptr_, size_);
            }
            cuda_push_array(gpu_ptr_, cpu_ptr_,size_);
            break;
        case GPU_MODE:
            break;
        }
    #endif
    }

    const void* SyncedMemory::cpu_data() {
        to_cpu();
        return (const void*)cpu_ptr_;
    }

    const void* SyncedMemory::gpu_data() {
    #ifdef GPU
        to_gpu();
        return (const void*)gpu_ptr_;
    #endif
    }

    void* SyncedMemory::mutable_cpu_data() {
        to_cpu();
        return cpu_ptr_;
    }

    void* SyncedMemory::mutable_gpu_data() {
    #ifdef GPU
        to_gpu();
        return gpu_ptr_;
    #endif
    return NULL;
    }
}  // namespace caffe

