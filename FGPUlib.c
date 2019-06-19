#include "clctypes.h"

#define TYPE_CHECK(TYPE,X,MSG) ({ TYPE __dummy; __typeof__(X) __dummy2; (void)(&__dummy == &__dummy2); 1; })


#define __local __attribute__((address_space(1)))

typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef unsigned int uint32_t;
typedef uint8_t uint8x128 __attribute__((ext_vector_type(128)));


inline int get_group_id(const int dim)
{
    int res;
    __asm__ __volatile__(
        "wgid %0, %1"
        : "=r"(res)
        : "I"(dim)
    );
    return res;
}

inline int get_local_size(const int dim)
{
    int res;
    __asm__ __volatile__(
        "wgsize %0, %1"
        : "=r"(res)
        : "I"(dim)
    );
    return res;
}

inline int get_global_size(const int dim)
{
    int res;
    __asm__ __volatile__(
        "size %0, %1"
        : "=r"(res)
        : "I"(dim)
    );
    return res;
}

inline int get_local_id(const int dim)
{
    int lid;
    __asm__ __volatile__(
        "lid %0, %1"
        : "=r"(lid)
        : "I"(dim)
    );
    return lid;
}

inline int get_global_id(const int dim)
{
    int index, lid;
    __asm__ __volatile__(
        "lid %0, %1"
        : "=r"(lid)
        : "I"(dim)
    );
    __asm__ __volatile__(
        "wgoff %0, %1"
        : "=r"(index)
        : "I"(dim)
    );
    return index + lid;
}


inline void local_sync()
{
    __asm__ __volatile__("lsync":::"memory");
}

inline void global_sync()
{
    __asm__ __volatile__("gsync":::"memory");
}


// FIXME: ahsu
// maybe the compiler should be able to infer these instructions?
inline float load_if_zero_fp32(float val, int cond, __global float* base, int offset)
{
    __asm__ __volatile__(
        "lwz %0, %1, [%2 + %3]"
        : "+r"(val)
        : "r"(cond), "r"(base), "I"(offset)
    );
    return val;
}

inline void store_if_zero_fp32(float val, int cond, __global float* base, int offset)
{
    __asm__ __volatile__(
        "swz %0, %1, [%2 + %3]"
        :
        : "r"(val), "r"(cond), "r"(base), "I"(offset)
    );
}

inline int8_t load_if_zero_int8(int8_t val, int cond, __global int8_t* base, int offset)
{
    __asm__ __volatile__(
        "lbz %0, %1, [%2 + %3]"
        : "+r"(val)
        : "r"(cond), "r"(base), "I"(offset)
    );
    return val;
}

inline void store_if_zero_int8(int8_t val, int cond, __global int8_t* base, int offset)
{
    __asm__ __volatile__(
        "sbz %0, %1, [%2 + %3]"
        :
        : "r"(val), "r"(cond), "r"(base), "I"(offset)
    );
}


// FIXME: ahsu
// dot functions defined here as macros instead of the wrapper functions commented below
// when invoking the inline assembly via a function call, the vector types exhibit a weird type splitting issue that
// does not occur when using inline assembly directly
// just using macros with static type checks for now instead of fixing the compiler issue
#define _dot(acc, srcA, srcB_ptr, srcB_off)                                                         \
{                                                                                                   \
    TYPE_CHECK(uint8x128,          srcA    , dot_srcA_is_expected_to_be_uint8x128);                 \
    TYPE_CHECK(__local uint8x128*, srcB_ptr, dot_srcB_ptr_is_expected_to_be_local_uint8x128_ptr);   \
    TYPE_CHECK(int,                srcB_off, dot_srcB_off_is_expected_to_be_int);                   \
    __asm__("vdot %0, %1, [%2 + %3]"                                                                \
        : "+r"(acc) : "f"(srcA), "r"(srcB_ptr), "I"(srcB_off) : "memory");                          \
}

#define dot_fp32(acc, srcA, srcB_ptr, srcB_off)                                                     \
{                                                                                                   \
    TYPE_CHECK(float,              acc,      dot_accumulator_is_expected_to_be_fp32);               \
    _dot(acc, srcA, srcB_ptr, srcB_off);                                                            \
}

#define dot_int32(acc, srcA, srcB_ptr, srcB_off)                                                    \
{                                                                                                   \
    TYPE_CHECK(int,                acc,      dot_accumulator_is_expected_to_be_int32);              \
    _dot(acc, srcA, srcB_ptr, srcB_off);                                                            \
}

//inline float dot_fp32(float acc, uint8x128 srcA, __local uint8x128* srcB_ptr, int srcB_off)
//{
//    __asm__ __volatile__(
//        "vdot %0, %1, [%2 + %3]"
//        : "+r"(acc)
//        : "f"(srcA), "r"(srcB_ptr), "I"(srcB_off)
//    );
//    return acc;
//}

//inline int dot_int32(int acc, uint8x128 srcA, __local uint8x128* srcB_ptr, int srcB_off)
//{
//    __asm__ __volatile__(
//        "vdot %0, %1, [%2 + %3]"
//        : "+r"(acc)
//        : "f"(srcA), "r"(srcB_ptr), "I"(srcB_off)
//    );
//    return acc;
//}


inline float relu_fp32(float in)
{
    float out;
    __asm__ __volatile__(
        "relu %0, %1, 0"
        : "=r"(out)
        : "r"(in)
    );
    return out;
}

inline int relu_int32(int in)
{
    int out;
    __asm__ __volatile__(
        "relu %0, %1, 0"
        : "=r"(out)
        : "r"(in)
    );
    return out;
}

inline float sigmoid_fp32(float in)
{
    float out;
    __asm__ __volatile__(
        "sigmoid %0, %1, 0"
        : "=r"(out)
        : "r"(in)
    );
    return out;
}

inline float tanh_fp32(float in)
{
    float out;
    __asm__ __volatile__(
        "tanh %0, %1, 0"
        : "=r"(out)
        : "r"(in)
    );
    return out;
}
