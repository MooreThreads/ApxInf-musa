#pragma once

#if (defined(__MUSA_ARCH__) && (__MUSA_ARCH__ == 310)) ||      \
    (!defined(__MUSA_ARCH__) && defined(__MUSA_ARCH_LIST__) && \
     (__MUSA_ARCH_LIST__ == 310))
#include "gemm_mp31.h"
#elif (defined(__MUSA_ARCH__) && (__MUSA_ARCH__ == 320)) ||    \
    (!defined(__MUSA_ARCH__) && defined(__MUSA_ARCH_LIST__) && \
     (__MUSA_ARCH_LIST__ == 320))
#include "gemm_mp32.h"
#elif (defined(__MUSA_ARCH__) && (__MUSA_ARCH__ == 220)) ||    \
    (!defined(__MUSA_ARCH__) && defined(__MUSA_ARCH_LIST__) && \
     (__MUSA_ARCH_LIST__ == 220))
#include "gemm_mp22.h"
#else
// No matching architecture found
#endif
