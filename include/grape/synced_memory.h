#ifndef __GRAPE_SYNCED_MEMORY_H__
#define __GRAPE_SYNCED_MEMORY_H__

#include <cstdint>
#include "grape/util/cuda.h"
#include "grape/global_config.h"

namespace Grape {
    class SyncedMemory {
    public:
        explicit SyncedMemory();
        explicit SyncedMemory(uint32_t size);
        virtual ~SyncedMemory();
        const void* cpu_data();
        const void* gpu_data();
        void set_cpu_data(void* data);
        void set_gpu_data(void* data);
        void* mutable_cpu_data();
        void* mutable_gpu_data();
        enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
        SyncedHead head() const { return head_; }
        uint32_t size() const { return size_; }

    private:
        void to_cpu();
        void to_gpu();
        void* cpu_ptr_;
        void* gpu_ptr_;
        uint32_t size_;
        SyncedHead head_;
        bool own_cpu_data_;
        bool own_gpu_data_;
        int device_;
    };  // class SyncedMemory
}  

#endif
