#include <memory.h>
#include <stdlib.h>
#include "grape/synced_memory.h"
#include "grape/util/cuda.h"


namespace Grape {
    SyncedMemory::SyncedMemory(): 
    cpu_ptr_(nullptr), 
    gpu_ptr_(nullptr), 
    size_(0), 
    head_(UNINITIALIZED),
    own_cpu_data_(false), 
    own_gpu_data_(false) 
    {
        
    }

    SyncedMemory::SyncedMemory(uint32_t size)
    : cpu_ptr_(nullptr), 
    gpu_ptr_(nullptr), 
    size_(size),
    head_(UNINITIALIZED),
    own_cpu_data_(false),
    own_gpu_data_(false) 
    {
       
    }

    SyncedMemory::~SyncedMemory()
    {
        if (cpu_ptr_ && own_cpu_data_) {
            free(cpu_ptr_);
        }

        #ifdef GPU
        if (gpu_ptr_ && own_gpu_data_) {
            cuda_free(gpu_ptr_);
        }
        #endif  // GPU
    }

    inline void SyncedMemory::to_cpu() 
    {
        switch (head_) {
        case UNINITIALIZED:
            cpu_ptr_ = malloc(size_);
            memset(cpu_ptr_, 0, size_);
            head_ = HEAD_AT_CPU;
            own_cpu_data_ = true;
            break;
        case HEAD_AT_GPU:
        #ifdef GPU
            if (cpu_ptr_ == nullptr) {
                cpu_ptr_ = malloc(size_);
                own_cpu_data_ = true;
            }
            cuda_pull_array(gpu_ptr_, cpu_ptr_, size_);
            head_ = SYNCED;
        #endif
            break;
        case HEAD_AT_CPU:
        case SYNCED:
            break;
        }
    }

    inline void SyncedMemory::to_gpu() 
    {
    #ifdef GPU
        switch (head_) {
        case UNINITIALIZED:
            cuda_malloc(&gpu_ptr_, size_);
            cuda_memset(gpu_ptr_,0, size_);
            head_ = HEAD_AT_GPU;
            own_gpu_data_ = true;
            break;
        case HEAD_AT_CPU:
            if (gpu_ptr_ == NULL) {
                cuda_malloc(&gpu_ptr_, size_);
                own_gpu_data_ = true;
            }
            cuda_push_array(gpu_ptr_, cpu_ptr_, size_);
            head_ = SYNCED;
            break;
        case HEAD_AT_GPU:
        case SYNCED:
            break;
        }
    #endif
    }

    const void* SyncedMemory::cpu_data()
    {
        to_cpu();
        return (const void*)cpu_ptr_;
    }

    void SyncedMemory::set_cpu_data(void* data) 
    {
        if (own_cpu_data_) {
            free(cpu_ptr_);
        }
        cpu_ptr_ = data;
        head_ = HEAD_AT_CPU;
        own_cpu_data_ = false;
    }

    const void* SyncedMemory::gpu_data() 
    {
    #ifdef GPU
        to_gpu();
        return (const void*)gpu_ptr_;
    #endif
        return NULL;
    }

    void SyncedMemory::set_gpu_data(void* data) 
    {
    #ifdef GPU
        if (own_gpu_data_) {
            cuda_free(gpu_ptr_);
        }
        gpu_ptr_ = data;
        head_ = HEAD_AT_GPU;
        own_gpu_data_ = false;
    #endif
    }

    void* SyncedMemory::mutable_cpu_data() 
    {
        to_cpu();
        head_ = HEAD_AT_CPU;
        return cpu_ptr_;
    }

    void* SyncedMemory::mutable_gpu_data() 
    {
    #ifdef GPU
        to_gpu();
        head_ = HEAD_AT_GPU;
        return gpu_ptr_;
    #endif
        return nullptr;
    }


}  // namespace Grape

