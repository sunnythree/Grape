#ifndef __GRAPE_SYNCED_MEMORY_H__
#define __GRAPE_SYNCED_MEMORY_H__

#include <cstdint>
#include "grape/util/cuda.h"
#include "grape/global_config.h"

namespace Grape {
    class SyncedMemory {
    public:
        explicit SyncedMemory(uint32_t size,CAL_MODE mode);
        virtual ~SyncedMemory();
        const void* cpu_data();
        const void* gpu_data();
        void* mutable_cpu_data();
        void* mutable_gpu_data();
        uint32_t size() const { return size_; }
        void to_cpu();
        void to_gpu();
    private:
        SyncedMemory() = delete;

        void* cpu_ptr_;
        void* gpu_ptr_;
        uint32_t size_;
        uint32_t device_;
        CAL_MODE mode_;
    };  // class SyncedMemory
}  

#endif
