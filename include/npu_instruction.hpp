#ifndef _NPU_INSTRUCTION_HPP_
#define _NPU_INSTRUCTION_HPP_

// public
#define NRAM_SIZE 1024 * 256 // 256KB
#define TCSR_SIZE 64 // 64

const long TCSR_DIM0 = 0;const long TCSR_STRIDE0 = 1;
const long TCSR_DIM1 = 2;const long TCSR_STRIDE1 = 3;
const long TCSR_DIM2 = 4;const long TCSR_STRIDE2 = 5;
const long TCSR_DIM3 = 6;const long TCSR_STRIDE3 = 7;

#include <stdio.h>
#include <stdint.h>
namespace RVT_EMU_FP8 {
    union fp32_int32 {
        float f;
        uint32_t bits;
    };

    template<int N>
    constexpr int exp_i2() {
        return 1 << N;
    }

    template<int N>
    constexpr float exp_f2() {
        if constexpr (N>0) return exp_f2<N-1>()*2;
        if constexpr (N<0) return exp_f2<N+1>()/2;
        if constexpr (N==0) return 1.;
        return 0.; // This should never happen, but just in case.
    }

    template<int _E> //, int M=7-E>  1.7 bits!
    struct FP8 {
        uint8_t bits;
        using type = FP8<_E>;
        static constexpr int E      = _E;
        static constexpr int M      = (7-_E);
        static constexpr int E_BIAS = exp_i2<E-1>()-1;
        static constexpr float MAX  = (2-exp_f2<-M+1>())*exp_f2<exp_i2<E-1>()>();
        static constexpr float MIN  = exp_f2<-M>()*exp_f2<2-exp_i2<E-1>()>();
    };

    // template<int E>
    // inline float to_float(const FP8<E>& in) {
    //     fp32_int32 out = {0};
    //     out.bits = in.bits & 0x80;
    //     out.bits <<= 24;
    //     uint32_t _bits = in.bits & 0x7F;
    //     _bits <<= (23-FP8<E>::M);
    //     out.bits |= _bits;
    //     out.f *= exp_f2<127-FP8<E>::E_BIAS>();
    //     return out.f;
    // }
    

    // TODO: Fixme, Not working in RISCV
    static int isubnorm[8] = {
                0x00000000,
                0x3b000000,
                0x3b800000,
                0x3bc00000,
                0x3c000000,
                0x3c200000,
                0x3c400000,
                0x3c600000};
    template<int E>
    inline float to_float(const FP8<E>& in) {
        fp32_int32 out = {0};
        out.bits = in.bits & 0x80;
        out.bits <<= 24;
        if(E == 4) {
            if((in.bits & 0x7f) < 0x08) {
                out.bits |= isubnorm[in.bits & 0x7f];
                return out.f;
            }
        }
        uint32_t _bits = in.bits & 0x7F;
        _bits <<= (23-FP8<E>::M);
        out.bits |= _bits;
        out.f *= exp_f2<127-FP8<E>::E_BIAS>();
        return out.f;
    }
    // template<int E>
    // inline float fp8Mul(const FP8<E>& a, const FP8<E>& b) {
    //     fp32_int32 out = {0};
    //     out.bits = (a.bits ^ b.bits) & 0x80;
    //     out.bits <<= 24;

    //     // Calculate mantissa
    //     long mant_a = (a.bits & ((1 << FP8<E>::M) - 1));
    //     long mant_b = (b.bits & ((1 << FP8<E>::M) - 1));

    //     // Calculate exponent
    //     long exp_a = (a.bits >> FP8<E>::M) & ((1 << E) - 1);
    //     long exp_b = (a.bits >> FP8<E>::M) & ((1 << E) - 1);

    //     // Handle subnormal numbers
    //     if(exp_a) mant_a |= (1 << FP8<E>::M); // mant_a is Q1.M
    //     if(exp_b) mant_b |= (1 << FP8<E>::M); // mant_b is Q1.M
    //     long exp = exp_a + exp_b - FP8<E>::E_BIAS;

    //     long mant = mant_a * mant_b; // mant is Q2.2M
    //     mant = (mant >> FP8<E>::M) + (mant & ((1 << FP8<E>::M) - 1)) ? 1 : 0; // mant is Q2.M
    //     long mant_clz = __builtin_clz(mant);
    //     const machine_width = sizeof(long) * 8;
    //     long width = machine_width - mant_clz;
    //     if(width == 0) return 0.0f;

    //     return out.f;
    // }
    template<int E>
    inline uint8_t from_float(float value) {
        FP8<E> out;
        fp32_int32 in = {value};
        out.bits = (in.bits >> 24) & 0x80;
        in.bits &= 0x7fffffff;
        if (in.f >= FP8<E>::MAX) {
            out.bits |= 0x7E;
        } else if (in.f < FP8<E>::MIN) { // => 0.
        } else if(E == 4 && in.bits < 0x3C800000) { // 2^-6
            if(in.bits >= 0x3c500000)      out.bits |= 0x7;
            else if(in.bits >= 0x3c300000) out.bits |= 0x6;
            else if(in.bits >= 0x3c100000) out.bits |= 0x5;
            else if(in.bits >= 0x3be00000) out.bits |= 0x4;
            else if(in.bits >= 0x3ba00000) out.bits |= 0x3;
            else if(in.bits >= 0x3b400000) out.bits |= 0x2;
            else if(in.bits >  0x3a000000) out.bits |= 0x1;
            return out.bits;
        } else {
            in.f *= exp_f2<FP8<E>::E_BIAS-127>();
            uint32_t eps = (0x3fffff>>FP8<E>::M) + ((in.bits >> (23-FP8<E>::M)) & 0x1);
            in.bits += eps;
            out.bits |= (in.bits >> (23-FP8<E>::M)) & 0x7F;
        }
        return out.bits;
    }
}

#ifdef __EMULATED_RVT_EXT__
// #include <simsimd/simsimd.h>
#define MAX_KTILE 256
extern "C" void exit(int code);
#define ASSERT(x) \
    do { \
        if (!(x)) { \
            printf("Assertion failed: %s\n", #x); \
            exit(1); \
        } \
    } while (0)

class TCFG{
public:
    long dim0;
    long stride0;
    long dim1;
    long stride1;
    long dim2;
    long stride2;
    long dim3;
    long stride3;
    long get_offset(long column /* DIM0 */, long row /* DIM1 */) {
        return (column * stride0 + row * stride1);
    }
    long get_offset(long dim0 /* DIM0 */, long dim1 /* DIM1 */, long dim2 /* DIM2 */) {
        return (dim0 * stride0 + dim1 * stride1 + dim2 * stride2);
    }
    long get_offset(long dim0 /* DIM0 */, long dim1 /* DIM1 */, long dim2 /* DIM2 */, long dim3 /* DIM3 */) {
        return (dim0 * stride0 + dim1 * stride1 + dim2 * stride2 + dim3 * stride3);
    }
};

class RVTEnv {
    public:
        char* aram;
        char* nram;
        long* tcsr;
        TCFG* tcfg;
        template <typename T>
        T* get_ptr(void* raw_ptr) {
            long lraw_ptr = (long)raw_ptr;
            if(lraw_ptr & (1ul << 31)) {
                lraw_ptr = lraw_ptr & 0x1fffffff; // No High 4 bits
                ASSERT(lraw_ptr < NRAM_SIZE);
                return (T*)(aram + lraw_ptr);
            } else {
                ASSERT(lraw_ptr < NRAM_SIZE);
                return (T*)(nram + lraw_ptr);
            }
        }
};
extern RVTEnv* __rvt_env;
static inline void __tx_fence() {
    // Do nothing
}

template <long tcfgrs1, long tcfgrs2, long tcfgrs3>
static inline void __tx_matmul_fix_s8_s8_s32(void* rs1, void* rs2, void* rs3, long& rd) {
    TCFG* tcfg1 = __rvt_env->tcfg +tcfgrs1;
    TCFG* tcfg2 = __rvt_env->tcfg +tcfgrs2;
    TCFG* tcfg3 = __rvt_env->tcfg +tcfgrs3;
    ASSERT(tcfg1->dim1 == tcfg2->dim0);
    ASSERT(tcfg1->dim0 == tcfg3->dim0);
    ASSERT(tcfg2->dim1 == tcfg3->dim1);
    for(int m =0; m < tcfg3->dim0; m+=1) {
        for(int n = 0 ; n < tcfg3->dim1; n+=1) {
            int result = 0;
            for(int k = 0; k < tcfg1->dim1; k += 1) {
                long offset1 = tcfg1->get_offset(m, k);
                long offset2 = tcfg2->get_offset(k, n);
                signed char rs1_val = *(__rvt_env->get_ptr<signed char>(rs1) + offset1);
                signed char rs2_val = *(__rvt_env->get_ptr<signed char>(rs2) + offset2);
                result += (int)rs1_val * (int)rs2_val;
            }
            long offset3 = tcfg3->get_offset(m, n);
            *(__rvt_env->get_ptr<int>(rs3) + offset3) += result;
        }
    }
}
template <long tcfgrs1, long tcfgrs2, long tcfgrs3>
static inline void __tx_matmul_fix_s8_s8_s32(void* rs1, void* rs2, void* rs3) {
    long rd = 0;
    __tx_matmul_fix_s8_s8_s32<tcfgrs1, tcfgrs2, tcfgrs3>(rs1, rs2, rs3, rd);
}

template <long tcfgrs1, long tcfgrs2, long tcfgrs3>
static inline void __tx_matmul_subfloat_e4m3_e4m3_bf16(void* rs1, void* rs2, void* rs3, long& rd) {
    printf("Not Implemented: __tx_matmul_subfloat_e4m3_e4m3_bf16\n");
}
template <long tcfgrs1, long tcfgrs2, long tcfgrs3>
static inline void __tx_matmul_subfloat_e4m3_e4m3_bf16(void* rs1, void* rs2, void* rs3) {
    long rd = 0;
    __tx_matmul_subfloat_e4m3_e4m3_bf16<tcfgrs1, tcfgrs2, tcfgrs3>(rs1, rs2, rs3, rd);
}

// union fp16_int16 {
//     simsimd_f16_t f;
//     uint16_t bits;
// };
// template<int E>
// static inline simsimd_f16_t to_float16(const RVT_EMU_FP8::FP8<E>& in) {
//     fp16_int16 out;
//     unsigned int e = (in.bits & 0x78) >> 3; // Extract exponent bits
//     unsigned int m = in.bits & 0x07;        // Extract mantissa bits
//     unsigned short result = ((in.bits & 0x80) << 8) | // Sign bit
//                            (((e - 7 + 15) << 10) | (m << 7)); // DONOT Handle denormalized numbers
//     out.bits = result;
//     return out.f;
// }
// union bfp16_int16 {
//     simsimd_bf16_t f;
//     uint16_t bits;
// };
// template<int E>
// static inline simsimd_bf16_t to_bfloat16(const RVT_EMU_FP8::FP8<E>& in) {
//     bfp16_int16 out;
//     unsigned int e = (in.bits & 0x78) >> 3; // Extract exponent bits
//     unsigned int m = in.bits & 0x07;        // Extract mantissa bits
//     unsigned short result = ((in.bits & 0x80) << 8) | // Sign bit
//                            (((e - 7 + 127) << 7) | (m << 4)); // DONOT Handle denormalized numbers
//     out.bits = result;
//     return out.f;
// }

template <long tcfgrs1, long tcfgrs2, long tcfgrs3>
static inline void __tx_matmul_subfloat_e4m3_e4m3_f32(void* rs1, void* rs2, void* rs3, long& rd) {
    TCFG* tcfg1 = __rvt_env->tcfg + tcfgrs1;
    TCFG* tcfg2 = __rvt_env->tcfg + tcfgrs2;
    TCFG* tcfg3 = __rvt_env->tcfg + tcfgrs3;
    ASSERT(tcfg1->dim1 == tcfg2->dim0);
    ASSERT(tcfg1->dim0 == tcfg3->dim0);
    ASSERT(tcfg2->dim1 == tcfg3->dim1);
    // Optimize for performance
    RVT_EMU_FP8::FP8<4>* a = __rvt_env->get_ptr<RVT_EMU_FP8::FP8<4>>(rs1);
    RVT_EMU_FP8::FP8<4>* b = __rvt_env->get_ptr<RVT_EMU_FP8::FP8<4>>(rs2);
    // simsimd_bf16_t aExp[MAX_KTILE];
    // simsimd_bf16_t bExp[MAX_KTILE];
    float* c = __rvt_env->get_ptr<float>(rs3);
    if(tcfg1->stride0 == 1) {
        // Not support yet
        ASSERT(false);
    } else {
        for(int m =0; m < tcfg3->dim0; m+=1) {
            for(int n = 0 ; n < tcfg3->dim1; n+=1) {
                RVT_EMU_FP8::FP8<4>* aRow = &a[m * tcfg1->stride0];
                RVT_EMU_FP8::FP8<4>* bRow = &b[n * tcfg2->stride1];
                float acc = c[n * tcfg3->stride1 + m];
                for(int k = 0; k < tcfg1->dim1; k += 1) {
                    // aExp[k] = to_bfloat16(aRow[k]);
                    // bExp[k] = to_bfloat16(bRow[k]);
                    float aExp = RVT_EMU_FP8::to_float(aRow[k]);
                    float bExp = RVT_EMU_FP8::to_float(bRow[k]);
                    float cExp = aExp * bExp;
                    acc += cExp;
                }
                // double acc = 0;
                // simsimd_dot_bf16(aExp, bExp, tcfg1->dim1, &acc);
                // printf("m: %ld, n: %ld, f: %lf\n", m, n, acc);
                c[n * tcfg3->stride1 + m] = acc;
            }
        }
    }
}
template <long tcfgrs1, long tcfgrs2, long tcfgrs3>
static inline void __tx_matmul_subfloat_e4m3_e4m3_f32(void* rs1, void* rs2, void* rs3) {
    long rd = 0;
    __tx_matmul_subfloat_e4m3_e4m3_f32<tcfgrs1, tcfgrs2, tcfgrs3>(rs1, rs2, rs3, rd);
}


static inline void* __emu_get_dram_ptr(long in) {
    return (void*)in;
}
static inline void* __emu_get_nram_ptr(long in) {
    return (void*)(__rvt_env->get_ptr<char>((void*)in));
}

template <typename T>
static inline void __emu_raw_copy(long src, long dst, TCFG* tcfgsrc, TCFG* tcfgdst, void*(* get_src_ptr)(long), void*(* get_dst_ptr)(long)) {
    ASSERT(tcfgsrc->dim0 * tcfgsrc->dim1 * tcfgsrc->dim2 * tcfgsrc->dim3 <= NRAM_SIZE);
    ASSERT(tcfgsrc->dim0 * tcfgsrc->dim1 * tcfgsrc->dim2 * tcfgsrc->dim3 == tcfgdst->dim0 * tcfgdst->dim1 * tcfgdst->dim2 * tcfgdst->dim3);
    long src0 = 0, src1 = 0, src2 = 0, src3 = 0;
    long dst0 = 0, dst1 = 0, dst2 = 0, dst3 = 0;
    while(1) {
        // One Element One Time
        *(T*)get_dst_ptr(dst + sizeof(T) * tcfgdst->get_offset(dst0, dst1, dst2, dst3)) = 
        *(T*)get_src_ptr(src + sizeof(T) * tcfgsrc->get_offset(src0, src1, src2, src3));
        src0 += 1;
        dst0 += 1;
        if(src0 >= tcfgsrc->dim0) {src0 = 0; src1 += 1;}
        if(src1 >= tcfgsrc->dim1) {src1 = 0; src2 += 1;}
        if(src2 >= tcfgsrc->dim2) {src2 = 0; src3 += 1;}
        if(src3 >= tcfgsrc->dim3) {src3 = 0; break;}
        if(dst0 >= tcfgdst->dim0) {dst0 = 0; dst1 += 1;}
        if(dst1 >= tcfgdst->dim1) {dst1 = 0; dst2 += 1;}
        if(dst2 >= tcfgdst->dim2) {dst2 = 0; dst3 += 1;}
    }
}

template <long tcfgrs1, long tcfgrs3, long log2w>
static inline void __tx_async_nram_load(void* rs1, void* rs3) {
    TCFG* tcfg1 = __rvt_env->tcfg + tcfgrs1;
    TCFG* tcfg3 = __rvt_env->tcfg + tcfgrs3;
    static_assert(log2w >= 0 && log2w <= 3, "log2w must be between 0 and 3");
    if(log2w == 0) __emu_raw_copy<char>((long)rs1, (long)rs3, tcfg1, tcfg3,  &__emu_get_dram_ptr, &__emu_get_nram_ptr);
    if(log2w == 1) __emu_raw_copy<short>((long)rs1, (long)rs3, tcfg1, tcfg3, &__emu_get_dram_ptr, &__emu_get_nram_ptr);
    if(log2w == 2) __emu_raw_copy<int>((long)rs1, (long)rs3, tcfg1, tcfg3,   &__emu_get_dram_ptr, &__emu_get_nram_ptr);
    if(log2w == 3) __emu_raw_copy<long>((long)rs1, (long)rs3, tcfg1, tcfg3,  &__emu_get_dram_ptr, &__emu_get_nram_ptr);
}
template <long tcfgrs1, long tcfgrs3, long log2w>
static inline void __tx_async_nram_load(void* rs1, void* rs3, long& rd) {
    __tx_async_nram_load<tcfgrs1, tcfgrs3, log2w>(rs1, rs3, rd);
}

template <long tcfgrs1, long tcfgrs3, long log2w>
static inline void __tx_async_nram_store(void* rs1, void* rs3, long& rd) {
    TCFG* tcfg1 = __rvt_env->tcfg + tcfgrs1;
    TCFG* tcfg3 = __rvt_env->tcfg + tcfgrs3;
    static_assert(log2w >= 0 && log2w <= 3, "log2w must be between 0 and 3");
    if(log2w == 0) __emu_raw_copy<char>((long)rs1, (long)rs3, tcfg1, tcfg3,  &__emu_get_nram_ptr, &__emu_get_dram_ptr);
    if(log2w == 1) __emu_raw_copy<short>((long)rs1, (long)rs3, tcfg1, tcfg3, &__emu_get_nram_ptr, &__emu_get_dram_ptr);
    if(log2w == 2) __emu_raw_copy<int>((long)rs1, (long)rs3, tcfg1, tcfg3,   &__emu_get_nram_ptr, &__emu_get_dram_ptr);
    if(log2w == 3) __emu_raw_copy<long>((long)rs1, (long)rs3, tcfg1, tcfg3,  &__emu_get_nram_ptr, &__emu_get_dram_ptr);
}
template <long tcfgrs1, long tcfgrs3, long log2w>
static inline void __tx_async_nram_store(void* rs1, void* rs3) {
    long rd = 0;
    __tx_async_nram_store<tcfgrs1, tcfgrs3, log2w>(rs1, rs3, rd);
}


template <long tcsrid>
static inline void __tx_tcsr_xchg_nmask(long rs1, long rs3, long &rd) {
    rd = __rvt_env->tcsr[tcsrid];
    __rvt_env->tcsr[tcsrid] = rs1 & ~rs3 | (__rvt_env->tcsr[tcsrid] & rs3);
}

template <long tcsrid>
static inline void __tx_tcsr_xchg_mask(long rs1, long rs3, long &rd) {
    rd = __rvt_env->tcsr[tcsrid];
    __rvt_env->tcsr[tcsrid] = rs1 & rs3 | (__rvt_env->tcsr[tcsrid] & ~rs3);
}

template <long tcsrid>
static inline void __tx_tcsr_write(long rs1) {
    long rd = 0;
    __tx_tcsr_xchg_nmask<tcsrid>(rs1, 0, rd);
}

template <long tcsrid>
static inline void __tx_tcsr_xchg(long rs1, long &rd) {
    __tx_tcsr_xchg_nmask<tcsrid>(rs1, 0, rd);
}

#else

namespace NPUInstBuilder {
    template <long encoding>
    __attribute__((always_inline))
    static inline void __custom_64b_instruction3r(long rs1, long rs2, long rs3, long& rd) 
    {
        asm volatile (
            "\n"
            ".ifc %0,zero\n.set _R0,0\n.endif\n.ifc %0,ra\n.set _R0,1\n.endif\n.ifc %0,sp\n.set _R0,2\n.endif\n.ifc %0,gp\n.set _R0,3\n.endif\n.ifc %0,tp\n.set _R0,4\n.endif\n.ifc %0,t0\n.set _R0,5\n.endif\n.ifc %0,t1\n.set _R0,6\n.endif\n.ifc %0,t2\n.set _R0,7\n.endif\n.ifc %0,s0\n.set _R0,8\n.endif\n.ifc %0,s1\n.set _R0,9\n.endif\n.ifc %0,a0\n.set _R0,10\n.endif\n.ifc %0,a1\n.set _R0,11\n.endif\n.ifc %0,a2\n.set _R0,12\n.endif\n.ifc %0,a3\n.set _R0,13\n.endif\n.ifc %0,a4\n.set _R0,14\n.endif\n.ifc %0,a5\n.set _R0,15\n.endif\n.ifc %0,a6\n.set _R0,16\n.endif\n.ifc %0,a7\n.set _R0,17\n.endif\n.ifc %0,s2\n.set _R0,18\n.endif\n.ifc %0,s3\n.set _R0,19\n.endif\n.ifc %0,s4\n.set _R0,20\n.endif\n.ifc %0,s5\n.set _R0,21\n.endif\n.ifc %0,s6\n.set _R0,22\n.endif\n.ifc %0,s7\n.set _R0,23\n.endif\n.ifc %0,s8\n.set _R0,24\n.endif\n.ifc %0,s9\n.set _R0,25\n.endif\n.ifc %0,s10\n.set _R0,26\n.endif\n.ifc %0,s11\n.set _R0,27\n.endif\n.ifc %0,t3\n.set _R0,28\n.endif\n.ifc %0,t4\n.set _R0,29\n.endif\n.ifc %0,t5\n.set _R0,30\n.endif\n.ifc %0,t6\n.set _R0,31\n.endif\n.ifc %1,zero\n.set _R1,0\n.endif\n.ifc %1,ra\n.set _R1,1\n.endif\n.ifc %1,sp\n.set _R1,2\n.endif\n.ifc %1,gp\n.set _R1,3\n.endif\n.ifc %1,tp\n.set _R1,4\n.endif\n.ifc %1,t0\n.set _R1,5\n.endif\n.ifc %1,t1\n.set _R1,6\n.endif\n.ifc %1,t2\n.set _R1,7\n.endif\n.ifc %1,s0\n.set _R1,8\n.endif\n.ifc %1,s1\n.set _R1,9\n.endif\n.ifc %1,a0\n.set _R1,10\n.endif\n.ifc %1,a1\n.set _R1,11\n.endif\n.ifc %1,a2\n.set _R1,12\n.endif\n.ifc %1,a3\n.set _R1,13\n.endif\n.ifc %1,a4\n.set _R1,14\n.endif\n.ifc %1,a5\n.set _R1,15\n.endif\n.ifc %1,a6\n.set _R1,16\n.endif\n.ifc %1,a7\n.set _R1,17\n.endif\n.ifc %1,s2\n.set _R1,18\n.endif\n.ifc %1,s3\n.set _R1,19\n.endif\n.ifc %1,s4\n.set _R1,20\n.endif\n.ifc %1,s5\n.set _R1,21\n.endif\n.ifc %1,s6\n.set _R1,22\n.endif\n.ifc %1,s7\n.set _R1,23\n.endif\n.ifc %1,s8\n.set _R1,24\n.endif\n.ifc %1,s9\n.set _R1,25\n.endif\n.ifc %1,s10\n.set _R1,26\n.endif\n.ifc %1,s11\n.set _R1,27\n.endif\n.ifc %1,t3\n.set _R1,28\n.endif\n.ifc %1,t4\n.set _R1,29\n.endif\n.ifc %1,t5\n.set _R1,30\n.endif\n.ifc %1,t6\n.set _R1,31\n.endif\n.ifc %2,zero\n.set _R2,0\n.endif\n.ifc %2,ra\n.set _R2,1\n.endif\n.ifc %2,sp\n.set _R2,2\n.endif\n.ifc %2,gp\n.set _R2,3\n.endif\n.ifc %2,tp\n.set _R2,4\n.endif\n.ifc %2,t0\n.set _R2,5\n.endif\n.ifc %2,t1\n.set _R2,6\n.endif\n.ifc %2,t2\n.set _R2,7\n.endif\n.ifc %2,s0\n.set _R2,8\n.endif\n.ifc %2,s1\n.set _R2,9\n.endif\n.ifc %2,a0\n.set _R2,10\n.endif\n.ifc %2,a1\n.set _R2,11\n.endif\n.ifc %2,a2\n.set _R2,12\n.endif\n.ifc %2,a3\n.set _R2,13\n.endif\n.ifc %2,a4\n.set _R2,14\n.endif\n.ifc %2,a5\n.set _R2,15\n.endif\n.ifc %2,a6\n.set _R2,16\n.endif\n.ifc %2,a7\n.set _R2,17\n.endif\n.ifc %2,s2\n.set _R2,18\n.endif\n.ifc %2,s3\n.set _R2,19\n.endif\n.ifc %2,s4\n.set _R2,20\n.endif\n.ifc %2,s5\n.set _R2,21\n.endif\n.ifc %2,s6\n.set _R2,22\n.endif\n.ifc %2,s7\n.set _R2,23\n.endif\n.ifc %2,s8\n.set _R2,24\n.endif\n.ifc %2,s9\n.set _R2,25\n.endif\n.ifc %2,s10\n.set _R2,26\n.endif\n.ifc %2,s11\n.set _R2,27\n.endif\n.ifc %2,t3\n.set _R2,28\n.endif\n.ifc %2,t4\n.set _R2,29\n.endif\n.ifc %2,t5\n.set _R2,30\n.endif\n.ifc %2,t6\n.set _R2,31\n.endif\n.ifc %3,zero\n.set _R3,0\n.endif\n.ifc %3,ra\n.set _R3,1\n.endif\n.ifc %3,sp\n.set _R3,2\n.endif\n.ifc %3,gp\n.set _R3,3\n.endif\n.ifc %3,tp\n.set _R3,4\n.endif\n.ifc %3,t0\n.set _R3,5\n.endif\n.ifc %3,t1\n.set _R3,6\n.endif\n.ifc %3,t2\n.set _R3,7\n.endif\n.ifc %3,s0\n.set _R3,8\n.endif\n.ifc %3,s1\n.set _R3,9\n.endif\n.ifc %3,a0\n.set _R3,10\n.endif\n.ifc %3,a1\n.set _R3,11\n.endif\n.ifc %3,a2\n.set _R3,12\n.endif\n.ifc %3,a3\n.set _R3,13\n.endif\n.ifc %3,a4\n.set _R3,14\n.endif\n.ifc %3,a5\n.set _R3,15\n.endif\n.ifc %3,a6\n.set _R3,16\n.endif\n.ifc %3,a7\n.set _R3,17\n.endif\n.ifc %3,s2\n.set _R3,18\n.endif\n.ifc %3,s3\n.set _R3,19\n.endif\n.ifc %3,s4\n.set _R3,20\n.endif\n.ifc %3,s5\n.set _R3,21\n.endif\n.ifc %3,s6\n.set _R3,22\n.endif\n.ifc %3,s7\n.set _R3,23\n.endif\n.ifc %3,s8\n.set _R3,24\n.endif\n.ifc %3,s9\n.set _R3,25\n.endif\n.ifc %3,s10\n.set _R3,26\n.endif\n.ifc %3,s11\n.set _R3,27\n.endif\n.ifc %3,t3\n.set _R3,28\n.endif\n.ifc %3,t4\n.set _R3,29\n.endif\n.ifc %3,t5\n.set _R3,30\n.endif\n.ifc %3,t6\n.set _R3,31\n.endif\n"
            ".insn 8, %4 | 0x3F | (_R0 << 7) | (_R1 << 15) | (_R2 << 20) | (_R3 << 27)\n"
            ".set _R0,z\n.set _R1,z\n.set _R2,z\n.set _R3,z\n"
            :"+r"(rd)
            :"r"(rs1), "r"(rs2), "r"(rs3), "i"(encoding)
        );
    }
    template <long encoding>
    __attribute__((always_inline))
    static inline void __custom_64b_instruction3r(long rs1, long rs2, long rs3) 
    {
        asm volatile (
            "\n"
            ".ifc %0,zero\n.set _R0,0\n.endif\n.ifc %0,ra\n.set _R0,1\n.endif\n.ifc %0,sp\n.set _R0,2\n.endif\n.ifc %0,gp\n.set _R0,3\n.endif\n.ifc %0,tp\n.set _R0,4\n.endif\n.ifc %0,t0\n.set _R0,5\n.endif\n.ifc %0,t1\n.set _R0,6\n.endif\n.ifc %0,t2\n.set _R0,7\n.endif\n.ifc %0,s0\n.set _R0,8\n.endif\n.ifc %0,s1\n.set _R0,9\n.endif\n.ifc %0,a0\n.set _R0,10\n.endif\n.ifc %0,a1\n.set _R0,11\n.endif\n.ifc %0,a2\n.set _R0,12\n.endif\n.ifc %0,a3\n.set _R0,13\n.endif\n.ifc %0,a4\n.set _R0,14\n.endif\n.ifc %0,a5\n.set _R0,15\n.endif\n.ifc %0,a6\n.set _R0,16\n.endif\n.ifc %0,a7\n.set _R0,17\n.endif\n.ifc %0,s2\n.set _R0,18\n.endif\n.ifc %0,s3\n.set _R0,19\n.endif\n.ifc %0,s4\n.set _R0,20\n.endif\n.ifc %0,s5\n.set _R0,21\n.endif\n.ifc %0,s6\n.set _R0,22\n.endif\n.ifc %0,s7\n.set _R0,23\n.endif\n.ifc %0,s8\n.set _R0,24\n.endif\n.ifc %0,s9\n.set _R0,25\n.endif\n.ifc %0,s10\n.set _R0,26\n.endif\n.ifc %0,s11\n.set _R0,27\n.endif\n.ifc %0,t3\n.set _R0,28\n.endif\n.ifc %0,t4\n.set _R0,29\n.endif\n.ifc %0,t5\n.set _R0,30\n.endif\n.ifc %0,t6\n.set _R0,31\n.endif\n.ifc %1,zero\n.set _R1,0\n.endif\n.ifc %1,ra\n.set _R1,1\n.endif\n.ifc %1,sp\n.set _R1,2\n.endif\n.ifc %1,gp\n.set _R1,3\n.endif\n.ifc %1,tp\n.set _R1,4\n.endif\n.ifc %1,t0\n.set _R1,5\n.endif\n.ifc %1,t1\n.set _R1,6\n.endif\n.ifc %1,t2\n.set _R1,7\n.endif\n.ifc %1,s0\n.set _R1,8\n.endif\n.ifc %1,s1\n.set _R1,9\n.endif\n.ifc %1,a0\n.set _R1,10\n.endif\n.ifc %1,a1\n.set _R1,11\n.endif\n.ifc %1,a2\n.set _R1,12\n.endif\n.ifc %1,a3\n.set _R1,13\n.endif\n.ifc %1,a4\n.set _R1,14\n.endif\n.ifc %1,a5\n.set _R1,15\n.endif\n.ifc %1,a6\n.set _R1,16\n.endif\n.ifc %1,a7\n.set _R1,17\n.endif\n.ifc %1,s2\n.set _R1,18\n.endif\n.ifc %1,s3\n.set _R1,19\n.endif\n.ifc %1,s4\n.set _R1,20\n.endif\n.ifc %1,s5\n.set _R1,21\n.endif\n.ifc %1,s6\n.set _R1,22\n.endif\n.ifc %1,s7\n.set _R1,23\n.endif\n.ifc %1,s8\n.set _R1,24\n.endif\n.ifc %1,s9\n.set _R1,25\n.endif\n.ifc %1,s10\n.set _R1,26\n.endif\n.ifc %1,s11\n.set _R1,27\n.endif\n.ifc %1,t3\n.set _R1,28\n.endif\n.ifc %1,t4\n.set _R1,29\n.endif\n.ifc %1,t5\n.set _R1,30\n.endif\n.ifc %1,t6\n.set _R1,31\n.endif\n.ifc %2,zero\n.set _R2,0\n.endif\n.ifc %2,ra\n.set _R2,1\n.endif\n.ifc %2,sp\n.set _R2,2\n.endif\n.ifc %2,gp\n.set _R2,3\n.endif\n.ifc %2,tp\n.set _R2,4\n.endif\n.ifc %2,t0\n.set _R2,5\n.endif\n.ifc %2,t1\n.set _R2,6\n.endif\n.ifc %2,t2\n.set _R2,7\n.endif\n.ifc %2,s0\n.set _R2,8\n.endif\n.ifc %2,s1\n.set _R2,9\n.endif\n.ifc %2,a0\n.set _R2,10\n.endif\n.ifc %2,a1\n.set _R2,11\n.endif\n.ifc %2,a2\n.set _R2,12\n.endif\n.ifc %2,a3\n.set _R2,13\n.endif\n.ifc %2,a4\n.set _R2,14\n.endif\n.ifc %2,a5\n.set _R2,15\n.endif\n.ifc %2,a6\n.set _R2,16\n.endif\n.ifc %2,a7\n.set _R2,17\n.endif\n.ifc %2,s2\n.set _R2,18\n.endif\n.ifc %2,s3\n.set _R2,19\n.endif\n.ifc %2,s4\n.set _R2,20\n.endif\n.ifc %2,s5\n.set _R2,21\n.endif\n.ifc %2,s6\n.set _R2,22\n.endif\n.ifc %2,s7\n.set _R2,23\n.endif\n.ifc %2,s8\n.set _R2,24\n.endif\n.ifc %2,s9\n.set _R2,25\n.endif\n.ifc %2,s10\n.set _R2,26\n.endif\n.ifc %2,s11\n.set _R2,27\n.endif\n.ifc %2,t3\n.set _R2,28\n.endif\n.ifc %2,t4\n.set _R2,29\n.endif\n.ifc %2,t5\n.set _R2,30\n.endif\n.ifc %2,t6\n.set _R2,31\n.endif\n"
            ".insn 8, %3 | 0x3F | (_R0 << 15) | (_R1 << 20) | (_R2 << 27)\n"
            ".set _R0,z\n.set _R1,z\n.set _R2,z\n"
            ::"r"(rs1), "r"(rs2), "r"(rs3), "i"(encoding)
        );
    }
    template <long encoding>
    __attribute__((always_inline))
    static inline void __custom_64b_instruction2r(long rs1, long rs3, long& rd) 
    {
        asm volatile (
            "\n"
            ".ifc %0,zero\n.set _R0,0\n.endif\n.ifc %0,ra\n.set _R0,1\n.endif\n.ifc %0,sp\n.set _R0,2\n.endif\n.ifc %0,gp\n.set _R0,3\n.endif\n.ifc %0,tp\n.set _R0,4\n.endif\n.ifc %0,t0\n.set _R0,5\n.endif\n.ifc %0,t1\n.set _R0,6\n.endif\n.ifc %0,t2\n.set _R0,7\n.endif\n.ifc %0,s0\n.set _R0,8\n.endif\n.ifc %0,s1\n.set _R0,9\n.endif\n.ifc %0,a0\n.set _R0,10\n.endif\n.ifc %0,a1\n.set _R0,11\n.endif\n.ifc %0,a2\n.set _R0,12\n.endif\n.ifc %0,a3\n.set _R0,13\n.endif\n.ifc %0,a4\n.set _R0,14\n.endif\n.ifc %0,a5\n.set _R0,15\n.endif\n.ifc %0,a6\n.set _R0,16\n.endif\n.ifc %0,a7\n.set _R0,17\n.endif\n.ifc %0,s2\n.set _R0,18\n.endif\n.ifc %0,s3\n.set _R0,19\n.endif\n.ifc %0,s4\n.set _R0,20\n.endif\n.ifc %0,s5\n.set _R0,21\n.endif\n.ifc %0,s6\n.set _R0,22\n.endif\n.ifc %0,s7\n.set _R0,23\n.endif\n.ifc %0,s8\n.set _R0,24\n.endif\n.ifc %0,s9\n.set _R0,25\n.endif\n.ifc %0,s10\n.set _R0,26\n.endif\n.ifc %0,s11\n.set _R0,27\n.endif\n.ifc %0,t3\n.set _R0,28\n.endif\n.ifc %0,t4\n.set _R0,29\n.endif\n.ifc %0,t5\n.set _R0,30\n.endif\n.ifc %0,t6\n.set _R0,31\n.endif\n.ifc %1,zero\n.set _R1,0\n.endif\n.ifc %1,ra\n.set _R1,1\n.endif\n.ifc %1,sp\n.set _R1,2\n.endif\n.ifc %1,gp\n.set _R1,3\n.endif\n.ifc %1,tp\n.set _R1,4\n.endif\n.ifc %1,t0\n.set _R1,5\n.endif\n.ifc %1,t1\n.set _R1,6\n.endif\n.ifc %1,t2\n.set _R1,7\n.endif\n.ifc %1,s0\n.set _R1,8\n.endif\n.ifc %1,s1\n.set _R1,9\n.endif\n.ifc %1,a0\n.set _R1,10\n.endif\n.ifc %1,a1\n.set _R1,11\n.endif\n.ifc %1,a2\n.set _R1,12\n.endif\n.ifc %1,a3\n.set _R1,13\n.endif\n.ifc %1,a4\n.set _R1,14\n.endif\n.ifc %1,a5\n.set _R1,15\n.endif\n.ifc %1,a6\n.set _R1,16\n.endif\n.ifc %1,a7\n.set _R1,17\n.endif\n.ifc %1,s2\n.set _R1,18\n.endif\n.ifc %1,s3\n.set _R1,19\n.endif\n.ifc %1,s4\n.set _R1,20\n.endif\n.ifc %1,s5\n.set _R1,21\n.endif\n.ifc %1,s6\n.set _R1,22\n.endif\n.ifc %1,s7\n.set _R1,23\n.endif\n.ifc %1,s8\n.set _R1,24\n.endif\n.ifc %1,s9\n.set _R1,25\n.endif\n.ifc %1,s10\n.set _R1,26\n.endif\n.ifc %1,s11\n.set _R1,27\n.endif\n.ifc %1,t3\n.set _R1,28\n.endif\n.ifc %1,t4\n.set _R1,29\n.endif\n.ifc %1,t5\n.set _R1,30\n.endif\n.ifc %1,t6\n.set _R1,31\n.endif\n.ifc %2,zero\n.set _R2,0\n.endif\n.ifc %2,ra\n.set _R2,1\n.endif\n.ifc %2,sp\n.set _R2,2\n.endif\n.ifc %2,gp\n.set _R2,3\n.endif\n.ifc %2,tp\n.set _R2,4\n.endif\n.ifc %2,t0\n.set _R2,5\n.endif\n.ifc %2,t1\n.set _R2,6\n.endif\n.ifc %2,t2\n.set _R2,7\n.endif\n.ifc %2,s0\n.set _R2,8\n.endif\n.ifc %2,s1\n.set _R2,9\n.endif\n.ifc %2,a0\n.set _R2,10\n.endif\n.ifc %2,a1\n.set _R2,11\n.endif\n.ifc %2,a2\n.set _R2,12\n.endif\n.ifc %2,a3\n.set _R2,13\n.endif\n.ifc %2,a4\n.set _R2,14\n.endif\n.ifc %2,a5\n.set _R2,15\n.endif\n.ifc %2,a6\n.set _R2,16\n.endif\n.ifc %2,a7\n.set _R2,17\n.endif\n.ifc %2,s2\n.set _R2,18\n.endif\n.ifc %2,s3\n.set _R2,19\n.endif\n.ifc %2,s4\n.set _R2,20\n.endif\n.ifc %2,s5\n.set _R2,21\n.endif\n.ifc %2,s6\n.set _R2,22\n.endif\n.ifc %2,s7\n.set _R2,23\n.endif\n.ifc %2,s8\n.set _R2,24\n.endif\n.ifc %2,s9\n.set _R2,25\n.endif\n.ifc %2,s10\n.set _R2,26\n.endif\n.ifc %2,s11\n.set _R2,27\n.endif\n.ifc %2,t3\n.set _R2,28\n.endif\n.ifc %2,t4\n.set _R2,29\n.endif\n.ifc %2,t5\n.set _R2,30\n.endif\n.ifc %2,t6\n.set _R2,31\n.endif\n"
            ".insn 8, %3 | 0x3F | (_R0 << 7) | (_R1 << 15) | (_R2 << 27)\n"
            ".set _R0,z\n.set _R1,z\n.set _R2,z\n"
            :"+r"(rd)
            :"r"(rs1), "r"(rs3), "i"(encoding)
        );
    }
    template <long encoding>
    __attribute__((always_inline))
    static inline void __custom_64b_instruction2r(long rs1, long rs3) 
    {
        asm volatile (
            "\n"
            ".ifc %0,zero\n.set _R0,0\n.endif\n.ifc %0,ra\n.set _R0,1\n.endif\n.ifc %0,sp\n.set _R0,2\n.endif\n.ifc %0,gp\n.set _R0,3\n.endif\n.ifc %0,tp\n.set _R0,4\n.endif\n.ifc %0,t0\n.set _R0,5\n.endif\n.ifc %0,t1\n.set _R0,6\n.endif\n.ifc %0,t2\n.set _R0,7\n.endif\n.ifc %0,s0\n.set _R0,8\n.endif\n.ifc %0,s1\n.set _R0,9\n.endif\n.ifc %0,a0\n.set _R0,10\n.endif\n.ifc %0,a1\n.set _R0,11\n.endif\n.ifc %0,a2\n.set _R0,12\n.endif\n.ifc %0,a3\n.set _R0,13\n.endif\n.ifc %0,a4\n.set _R0,14\n.endif\n.ifc %0,a5\n.set _R0,15\n.endif\n.ifc %0,a6\n.set _R0,16\n.endif\n.ifc %0,a7\n.set _R0,17\n.endif\n.ifc %0,s2\n.set _R0,18\n.endif\n.ifc %0,s3\n.set _R0,19\n.endif\n.ifc %0,s4\n.set _R0,20\n.endif\n.ifc %0,s5\n.set _R0,21\n.endif\n.ifc %0,s6\n.set _R0,22\n.endif\n.ifc %0,s7\n.set _R0,23\n.endif\n.ifc %0,s8\n.set _R0,24\n.endif\n.ifc %0,s9\n.set _R0,25\n.endif\n.ifc %0,s10\n.set _R0,26\n.endif\n.ifc %0,s11\n.set _R0,27\n.endif\n.ifc %0,t3\n.set _R0,28\n.endif\n.ifc %0,t4\n.set _R0,29\n.endif\n.ifc %0,t5\n.set _R0,30\n.endif\n.ifc %0,t6\n.set _R0,31\n.endif\n.ifc %1,zero\n.set _R1,0\n.endif\n.ifc %1,ra\n.set _R1,1\n.endif\n.ifc %1,sp\n.set _R1,2\n.endif\n.ifc %1,gp\n.set _R1,3\n.endif\n.ifc %1,tp\n.set _R1,4\n.endif\n.ifc %1,t0\n.set _R1,5\n.endif\n.ifc %1,t1\n.set _R1,6\n.endif\n.ifc %1,t2\n.set _R1,7\n.endif\n.ifc %1,s0\n.set _R1,8\n.endif\n.ifc %1,s1\n.set _R1,9\n.endif\n.ifc %1,a0\n.set _R1,10\n.endif\n.ifc %1,a1\n.set _R1,11\n.endif\n.ifc %1,a2\n.set _R1,12\n.endif\n.ifc %1,a3\n.set _R1,13\n.endif\n.ifc %1,a4\n.set _R1,14\n.endif\n.ifc %1,a5\n.set _R1,15\n.endif\n.ifc %1,a6\n.set _R1,16\n.endif\n.ifc %1,a7\n.set _R1,17\n.endif\n.ifc %1,s2\n.set _R1,18\n.endif\n.ifc %1,s3\n.set _R1,19\n.endif\n.ifc %1,s4\n.set _R1,20\n.endif\n.ifc %1,s5\n.set _R1,21\n.endif\n.ifc %1,s6\n.set _R1,22\n.endif\n.ifc %1,s7\n.set _R1,23\n.endif\n.ifc %1,s8\n.set _R1,24\n.endif\n.ifc %1,s9\n.set _R1,25\n.endif\n.ifc %1,s10\n.set _R1,26\n.endif\n.ifc %1,s11\n.set _R1,27\n.endif\n.ifc %1,t3\n.set _R1,28\n.endif\n.ifc %1,t4\n.set _R1,29\n.endif\n.ifc %1,t5\n.set _R1,30\n.endif\n.ifc %1,t6\n.set _R1,31\n.endif\n"
            ".insn 8, %2 | 0x3F | (_R0 << 15) | (_R1 << 27)\n"
            ".set _R0,z\n.set _R1,z\n"
            ::"r"(rs1), "r"(rs3), "i"(encoding)
        );
    }
    template <long encoding>
    __attribute__((always_inline))
    static inline void __custom_64b_instruction1r(long rs3, long& rd) 
    {
        asm volatile (
            "\n"
            ".ifc %0,zero\n.set _R0,0\n.endif\n.ifc %0,ra\n.set _R0,1\n.endif\n.ifc %0,sp\n.set _R0,2\n.endif\n.ifc %0,gp\n.set _R0,3\n.endif\n.ifc %0,tp\n.set _R0,4\n.endif\n.ifc %0,t0\n.set _R0,5\n.endif\n.ifc %0,t1\n.set _R0,6\n.endif\n.ifc %0,t2\n.set _R0,7\n.endif\n.ifc %0,s0\n.set _R0,8\n.endif\n.ifc %0,s1\n.set _R0,9\n.endif\n.ifc %0,a0\n.set _R0,10\n.endif\n.ifc %0,a1\n.set _R0,11\n.endif\n.ifc %0,a2\n.set _R0,12\n.endif\n.ifc %0,a3\n.set _R0,13\n.endif\n.ifc %0,a4\n.set _R0,14\n.endif\n.ifc %0,a5\n.set _R0,15\n.endif\n.ifc %0,a6\n.set _R0,16\n.endif\n.ifc %0,a7\n.set _R0,17\n.endif\n.ifc %0,s2\n.set _R0,18\n.endif\n.ifc %0,s3\n.set _R0,19\n.endif\n.ifc %0,s4\n.set _R0,20\n.endif\n.ifc %0,s5\n.set _R0,21\n.endif\n.ifc %0,s6\n.set _R0,22\n.endif\n.ifc %0,s7\n.set _R0,23\n.endif\n.ifc %0,s8\n.set _R0,24\n.endif\n.ifc %0,s9\n.set _R0,25\n.endif\n.ifc %0,s10\n.set _R0,26\n.endif\n.ifc %0,s11\n.set _R0,27\n.endif\n.ifc %0,t3\n.set _R0,28\n.endif\n.ifc %0,t4\n.set _R0,29\n.endif\n.ifc %0,t5\n.set _R0,30\n.endif\n.ifc %0,t6\n.set _R0,31\n.endif\n.ifc %1,zero\n.set _R1,0\n.endif\n.ifc %1,ra\n.set _R1,1\n.endif\n.ifc %1,sp\n.set _R1,2\n.endif\n.ifc %1,gp\n.set _R1,3\n.endif\n.ifc %1,tp\n.set _R1,4\n.endif\n.ifc %1,t0\n.set _R1,5\n.endif\n.ifc %1,t1\n.set _R1,6\n.endif\n.ifc %1,t2\n.set _R1,7\n.endif\n.ifc %1,s0\n.set _R1,8\n.endif\n.ifc %1,s1\n.set _R1,9\n.endif\n.ifc %1,a0\n.set _R1,10\n.endif\n.ifc %1,a1\n.set _R1,11\n.endif\n.ifc %1,a2\n.set _R1,12\n.endif\n.ifc %1,a3\n.set _R1,13\n.endif\n.ifc %1,a4\n.set _R1,14\n.endif\n.ifc %1,a5\n.set _R1,15\n.endif\n.ifc %1,a6\n.set _R1,16\n.endif\n.ifc %1,a7\n.set _R1,17\n.endif\n.ifc %1,s2\n.set _R1,18\n.endif\n.ifc %1,s3\n.set _R1,19\n.endif\n.ifc %1,s4\n.set _R1,20\n.endif\n.ifc %1,s5\n.set _R1,21\n.endif\n.ifc %1,s6\n.set _R1,22\n.endif\n.ifc %1,s7\n.set _R1,23\n.endif\n.ifc %1,s8\n.set _R1,24\n.endif\n.ifc %1,s9\n.set _R1,25\n.endif\n.ifc %1,s10\n.set _R1,26\n.endif\n.ifc %1,s11\n.set _R1,27\n.endif\n.ifc %1,t3\n.set _R1,28\n.endif\n.ifc %1,t4\n.set _R1,29\n.endif\n.ifc %1,t5\n.set _R1,30\n.endif\n.ifc %1,t6\n.set _R1,31\n.endif\n"
            ".insn 8, %2 | 0x3F | (_R0 << 7) | (_R1 << 27)\n"
            ".set _R0,z\n.set _R1,z\n"
            :"+r"(rd)
            :"r"(rs3), "i"(encoding)
        );
    }
    template <long encoding>
    __attribute__((always_inline))
    static inline void __custom_64b_instruction1r(long rs3) 
    {
        asm volatile (
            "\n"
            ".ifc %0,zero\n.set _R0,0\n.endif\n.ifc %0,ra\n.set _R0,1\n.endif\n.ifc %0,sp\n.set _R0,2\n.endif\n.ifc %0,gp\n.set _R0,3\n.endif\n.ifc %0,tp\n.set _R0,4\n.endif\n.ifc %0,t0\n.set _R0,5\n.endif\n.ifc %0,t1\n.set _R0,6\n.endif\n.ifc %0,t2\n.set _R0,7\n.endif\n.ifc %0,s0\n.set _R0,8\n.endif\n.ifc %0,s1\n.set _R0,9\n.endif\n.ifc %0,a0\n.set _R0,10\n.endif\n.ifc %0,a1\n.set _R0,11\n.endif\n.ifc %0,a2\n.set _R0,12\n.endif\n.ifc %0,a3\n.set _R0,13\n.endif\n.ifc %0,a4\n.set _R0,14\n.endif\n.ifc %0,a5\n.set _R0,15\n.endif\n.ifc %0,a6\n.set _R0,16\n.endif\n.ifc %0,a7\n.set _R0,17\n.endif\n.ifc %0,s2\n.set _R0,18\n.endif\n.ifc %0,s3\n.set _R0,19\n.endif\n.ifc %0,s4\n.set _R0,20\n.endif\n.ifc %0,s5\n.set _R0,21\n.endif\n.ifc %0,s6\n.set _R0,22\n.endif\n.ifc %0,s7\n.set _R0,23\n.endif\n.ifc %0,s8\n.set _R0,24\n.endif\n.ifc %0,s9\n.set _R0,25\n.endif\n.ifc %0,s10\n.set _R0,26\n.endif\n.ifc %0,s11\n.set _R0,27\n.endif\n.ifc %0,t3\n.set _R0,28\n.endif\n.ifc %0,t4\n.set _R0,29\n.endif\n.ifc %0,t5\n.set _R0,30\n.endif\n.ifc %0,t6\n.set _R0,31\n.endif\n"
            ".insn 8, %1 | 0x3F | (_R0 << 27)\n"
            ".set _R0,z\n"
            :"r"(rs3), "i"(encoding)
        );
    }

    template <long basetype, long rs1type, long rs2type, long rs3type, long tcfgrs1, long tcfgrs2, long tcfgrs3>
    __attribute__((always_inline))
    static inline void __tx_matmul(void* rs1, void* rs2, void* rs3, long& rd)
    {
        NPUInstBuilder::__custom_64b_instruction3r<
        (basetype << 56ul) | (rs1type << 50ul) | (rs2type << 44ul) | (rs3type << 43ul) | (tcfgrs1 << 32ul) | (tcfgrs2 << 35ul) | (tcfgrs3 << 38ul)>
        ((long)rs1, (long)rs2, (long)rs3, rd);
    }
    template <long basetype, long rs1type, long rs2type, long rs3type, long tcfgrs1, long tcfgrs2, long tcfgrs3>
    __attribute__((always_inline))
    static inline void __tx_matmul(void* rs1, void* rs2, void* rs3)
    {
        NPUInstBuilder::__custom_64b_instruction3r<
        (basetype << 56ul) | (rs1type << 50ul) | (rs2type << 44ul) | (rs3type << 43ul) | (tcfgrs1 << 32ul) | (tcfgrs2 << 35ul) | (tcfgrs3 << 38ul)>
        ((long)rs1, (long)rs2, (long)rs3);
    }

    enum __TX_ELEM_DTYPE_ENUM {
        TX_SUBBYTE_FLOAT = 0,
        TX_U8   = 0b1000000,
        TX_S8   = 0b1000001,
        TX_U16  = 0b1000010,
        TX_S16  = 0b1000011,
        TX_U32  = 0b1000100,
        TX_S32  = 0b1000101,
        TX_U64  = 0b1000110,
        TX_S64  = 0b1000111,
        TX_FP16 = 0b1001010,
        TX_BF16 = 0b1001011,
        TX_FP32 = 0b1001100,
        TX_TF32 = 0b1001101,
        TX_FP64 = 0b1001110
    };

    enum __TX_INT_OP_ENUM{
        TX_INT_ADD   = 0,
        TX_INT_SUB   = 1,
        TX_INT_MUL   = 2,
        TX_INT_SRL   = 3,
        TX_INT_SRA   = 4,
        TX_INT_SLL   = 5,
        TX_INT_LUTOP = 16, // 16 | LUT 
    };
    template <long op, long type, long tcfgrs1, long tcfgrs2, long tcfgrs3, long broadcast>
    __attribute__((always_inline))
    static inline void __tx_int_op_type(void* rs1, void* rs2, void* rs3)
    {
        NPUInstBuilder::__custom_64b_instruction3r<
            (0b00001000ul << 56ul) | // func == 0000 1000
            (op << 50ul)           | // op
            (broadcast << 45ul)    | // broadcast
            (((type >> 3) & 0b1111) << 41ul) | // Type-HI
            (((type >> 0) & 0b0111) << 12ul) | // Type-LO
            (tcfgrs3 << 38ul)      |
            (tcfgrs2 << 35ul)      |
            (tcfgrs1 << 32ul)
        >((long)rs1, (long)rs2, (long)rs3);
    }

    enum __TX_FP_OP_ENUM{
        TX_FP_ADD   = 0,
        TX_FP_SUB   = 1,
        TX_FP_MUL   = 2
    };

    enum __TX_FP_REDUCE_ENUM{
        TX_FP_MAX   = 0,
        TX_FP_MIN   = 1,
        TX_FP_SUM   = 2
    };

    template <long op, long type, long tcfgrs1, long tcfgrs2, long tcfgrs3, long broadcast>
    __attribute__((always_inline))
    static inline void __tx_fp_op_type(void* rs1, void* rs2, void* rs3)
    {
        NPUInstBuilder::__custom_64b_instruction3r<
            (0b00001001ul << 56ul) | // func == 0000 1001
            (op << 50ul)           | // op
            (broadcast << 45ul)    | // broadcast
            (((type >> 3) & 0b1111) << 41ul) | // Type-HI
            (((type >> 0) & 0b0111) << 12ul) | // Type-LO
            (tcfgrs3 << 38ul)      |
            (tcfgrs2 << 35ul)      |
            (tcfgrs1 << 32ul)
        >((long)rs1, (long)rs2, (long)rs3);
    }

    template <long type, long tcfgrs1, long width, long tcfgrs3, long log2Length>
    __attribute__((always_inline))
    static inline void __tx_lut_int_type(long rs1, long rs2, long rs3)
    {
        NPUInstBuilder::__custom_64b_instruction3r<
            (0b00001010ul << 56ul) | // func == 0000 1010
            (log2Length << 50ul)   | // log2Length
            (((type >> 3) & 0b1111) << 41ul) | // Type-HI
            (((type >> 0) & 0b0111) << 12ul) | // Type-LO
            (tcfgrs3 << 38ul)      |
            (width << 35ul)        |
            (tcfgrs1 << 32ul)
        >((long)rs1, (long)rs2, (long)rs3);
    }

    template <long mode, long type, long log2Length, long tcfgrs1, long width, long tcfgrs3>
    __attribute__((always_inline))
    static inline void __tx_lut_fp_mode_type(long rs1, long rs2, long rs3)
    {
        NPUInstBuilder::__custom_64b_instruction3r<
            (0b00001011ul << 56ul) | // func == 0000 1011
            (log2Length << 50ul)   | // log2Length
            (((type >> 3) & 0b1111) << 41ul) | // Type-HI
            (((type >> 0) & 0b0111) << 12ul) | // Type-LO
            (tcfgrs3 << 38ul)      |
            (width << 35ul)        |
            (tcfgrs1 << 32ul)      |
            (mode << 25ul)
        >((long)rs1, (long)rs2, (long)rs3);
    }

    template <long op, long type, long tcfgrs1, long tcfgrs3, long dim>
    __attribute__((always_inline))
    static inline void __tx_int_reduce_op_type(void* rs1, void* rs3)
    {
        NPUInstBuilder::__custom_64b_instruction2r<
            (0b10001100ul << 56ul) | // func == 1000 1100
            (op << 50ul)           | // op
            (((type >> 3) & 0b1111) << 41ul) | // Type-HI
            (((type >> 0) & 0b0111) << 12ul) | // Type-LO
            (tcfgrs3 << 38ul)      |
            (tcfgrs1 << 32ul)      |
            (dim << 25ul)
        >((long)rs1, (long)rs3);
    }

    template <long op, long type, long tcfgrs1, long tcfgrs3, long dim>
    __attribute__((always_inline))
    static inline void __tx_fp_reduce_op_type(void* rs1, void* rs3)
    {
         constexpr long encoding =  (0b10001101ul << 56ul) | // func == 1000 1101
            (op << 50ul)           | // op
            (((type >> 3) & 0b1111) << 41ul) | // Type-HI
            (((type >> 0) & 0b0111) << 12ul) | // Type-LO
            (tcfgrs3 << 38ul)      |
            (tcfgrs1 << 32ul)      |
            (dim << 25ul);
            NPUInstBuilder::__custom_64b_instruction2r<encoding>((long)rs1, (long)rs3);
    }

    enum __TX_CONVERT_MODE{
        TX_CONVERT_ROUND = 0,
        TX_CONVERT_FLOOR = 1,
        TX_CONVERT_CEIL  = 2
    };
    template <long rs1type, long rs3type, long mode, long tcfgrs1, long tcfgrs3>
    __attribute__((always_inline))
    static inline void __tx_convert_rs1type_rs3type_mode(long rs1, long rs3)
    {
        NPUInstBuilder::__custom_64b_instruction3r<
            (0b10001110ul << 56ul) | // func == 1000 1110
            (rs3type << 45ul)      |
            (((rs1type >> 3) & 0b1111) << 41ul) | // Type-HI
            (((rs1type >> 0) & 0b0111) << 12ul) | // Type-LO
            (tcfgrs3 << 38ul)      |
            (tcfgrs1 << 32ul)      |
            (mode << 25ul)
        >((long)rs1, (long)rs3);
    }
}

template <long tcfgrs1, long tcfgrs2, long tcfgrs3>
__attribute__((always_inline))
static inline void __tx_matmul_subfloat_e4m3_e4m3_bf16(void* rs1, void* rs2, void* rs3, long& rd) {
    NPUInstBuilder::__tx_matmul<0x0, 0b011100, 0b011100, 0b1, tcfgrs1, tcfgrs2, tcfgrs3>(rs1, rs2, rs3, rd);
}
template <long tcfgrs1, long tcfgrs2, long tcfgrs3>
__attribute__((always_inline))
static inline void __tx_matmul_subfloat_e4m3_e4m3_bf16(void* rs1, void* rs2, void* rs3)
{
    NPUInstBuilder::__tx_matmul<0x0, 0b011100, 0b011100, 0b1, tcfgrs1, tcfgrs2, tcfgrs3>(rs1, rs2, rs3);
}

template <long tcfgrs1, long tcfgrs2, long tcfgrs3>
__attribute__((always_inline))
static inline void __tx_matmul_subfloat_e4m3_e4m3_32(void* rs1, void* rs2, void* rs3, long& rd) {
    NPUInstBuilder::__tx_matmul<0x0, 0b011100, 0b011100, 0b0, tcfgrs1, tcfgrs2, tcfgrs3>(rs1, rs2, rs3, rd);
}
template <long tcfgrs1, long tcfgrs2, long tcfgrs3>
__attribute__((always_inline))
static inline void __tx_matmul_subfloat_e4m3_e4m3_f32(void* rs1, void* rs2, void* rs3)
{
    NPUInstBuilder::__tx_matmul<0x0, 0b011100, 0b011100, 0b0, tcfgrs1, tcfgrs2, tcfgrs3>(rs1, rs2, rs3);
}
template <long tcfgrs1, long tcfgrs2, long tcfgrs3>
__attribute__((always_inline))
static inline void __tx_matmul_float_f32_f32_f32(void* rs1, void* rs2, void* rs3) {
    NPUInstBuilder::__tx_matmul<0x1, 0b10, 0b10, 0b0, tcfgrs1, tcfgrs2, tcfgrs3>(rs1, rs2, rs3);
}
template <long tcfgrs1, long tcfgrs2, long tcfgrs3>
__attribute__((always_inline))
static inline void __tx_matmul_fix_s8_s8_s32(void* rs1, void* rs2, void* rs3, long& rd) {
    NPUInstBuilder::__tx_matmul<0b00000010, 7, 7, 0b0, tcfgrs1, tcfgrs2, tcfgrs3>(rs1, rs2, rs3, rd);
}
template <long tcfgrs1, long tcfgrs2, long tcfgrs3>
__attribute__((always_inline))
static inline void __tx_matmul_fix_s8_s8_s32(void* rs1, void* rs2, void* rs3)
{
    NPUInstBuilder::__tx_matmul<0b00000010, 7, 7, 0b0, tcfgrs1, tcfgrs2, tcfgrs3>(rs1, rs2, rs3);
}

template <long tcfgrs1, long tcfgrs3, long log2w>
__attribute__((always_inline))
static inline void __tx_async_nram_load(void* rs1, void* rs3, long& rd) {
    NPUInstBuilder::__custom_64b_instruction2r<
        ((long)0b10000110 << 56ul) | (tcfgrs3 << 38ul) | (tcfgrs1 << 32ul) | (log2w << 12)>((long)rs1, (long)rs3, rd);
}


template <long tcfgrs1, long tcfgrs3, long log2w>
__attribute__((always_inline))
static inline void __tx_async_nram_load(void* rs1, void* rs3) {
    NPUInstBuilder::__custom_64b_instruction2r<
        ((long)0b10000110 << 56ul) | (tcfgrs3 << 38ul) | (tcfgrs1 << 32ul) | (log2w << 12)>((long)rs1, (long)rs3);
}

template <long tcfgrs1, long tcfgrs3, long log2w>
__attribute__((always_inline))
static inline void __tx_async_nram_store(void* rs1, void* rs3, long& rd) {
    NPUInstBuilder::__custom_64b_instruction2r<
        ((long)0b10000101 << 56ul) | (tcfgrs3 << 38ul) | (tcfgrs1 << 32ul) | (log2w << 12)>((long)rs1, (long)rs3, rd);
}
template <long tcfgrs1, long tcfgrs3, long log2w>
__attribute__((always_inline))
static inline void __tx_async_nram_store(void* rs1, void* rs3) {
    NPUInstBuilder::__custom_64b_instruction2r<
        ((long)0b10000101 << 56ul) | (tcfgrs3 << 38ul) | (tcfgrs1 << 32ul) | (log2w << 12)>((long)rs1, (long)rs3);
}


template <long tcsrid>
__attribute__((always_inline))
static inline void __tx_tcsr_write(long rs1) {
    NPUInstBuilder::__custom_64b_instruction2r<
        ((long)0b10000000 << 56ul) | (tcsrid << 32)>(rs1, 0);
}
template <long tcsrid>
__attribute__((always_inline))
static inline void __tx_tcsr_xchg(long rs1, long &rd) {
    NPUInstBuilder::__custom_64b_instruction2r<
        ((long)0b10000000 << 56ul) | (tcsrid << 32)>(rs1, 0, rd);
}

__attribute__((always_inline))
static inline void __tx_fence() {
    asm volatile (
        "FENCE iorw, iorw"
    );
}

#define GENERATE_TX_INT_OP_TYPE(opname, op, typename, type) \
template <long tcfgrs1, long tcfgrs2, long tcfgrs3, long broadcast> \
__attribute__((always_inline)) \
static inline void __tx_int_##opname##_##typename(void* rs1, void* rs2, void* rs3) { \
    NPUInstBuilder::__tx_int_op_type< \
        NPUInstBuilder::op, \
        NPUInstBuilder::type, \
        tcfgrs1, tcfgrs2, tcfgrs3, broadcast \
        >(rs1, rs2, rs3); \
}

#define GENERATE_TX_FP_OP_TYPE(opname, op, typename, type) \
template <long tcfgrs1, long tcfgrs2, long tcfgrs3, long broadcast> \
__attribute__((always_inline)) \
static inline void __tx_fp_##opname##_##typename(void* rs1, void* rs2, void* rs3) { \
    NPUInstBuilder::__tx_fp_op_type< \
        NPUInstBuilder::op, \
        NPUInstBuilder::type, \
        tcfgrs1, tcfgrs2, tcfgrs3, broadcast \
        >(rs1, rs2, rs3); \
}

GENERATE_TX_INT_OP_TYPE(add,   TX_INT_ADD,   s32, TX_S32)
GENERATE_TX_INT_OP_TYPE(sub,   TX_INT_SUB,   s32, TX_S32)
GENERATE_TX_INT_OP_TYPE(mul,   TX_INT_MUL,   s32, TX_S32)
GENERATE_TX_INT_OP_TYPE(srl,   TX_INT_SRL,   s32, TX_S32)
GENERATE_TX_INT_OP_TYPE(sra,   TX_INT_SRA,   s32, TX_S32)
GENERATE_TX_INT_OP_TYPE(sll,   TX_INT_SLL,   s32, TX_S32)
GENERATE_TX_INT_OP_TYPE(lutop, TX_INT_LUTOP, s32, TX_S32)

GENERATE_TX_FP_OP_TYPE(add, TX_FP_ADD, fp32, TX_FP32)
GENERATE_TX_FP_OP_TYPE(sub, TX_FP_SUB, fp32, TX_FP32)
GENERATE_TX_FP_OP_TYPE(mul, TX_FP_MUL, fp32, TX_FP32)

#undef GENERATE_TX_INT_OP_TYPE
#undef GENERATE_TX_FP_OP_TYPE

#define GENERATE_TX_CONVERT(rs1typename, rs1type, rs3typename, rs3type, modename, mode) \
template <long tcfgrs1, long tcfgrs3> \
__attribute__((always_inline)) \
static inline void __tx_convert_##rs1typename##_##rs3typename##modename(long rs1, long rs3) { \
    NPUInstBuilder::__tx_convert_rs1type_rs3type_mode< \
        NPUInstBuilder::rs1type, \
        NPUInstBuilder::rs3type, \
        NPUInstBuilder::mode, \
        tcfgrs1, tcfgrs3 \
        >(rs1, rs3); \
}

GENERATE_TX_CONVERT(fp32, TX_FP32, s32 , TX_S32 , _round, TX_CONVERT_ROUND)
GENERATE_TX_CONVERT(fp32, TX_FP32, s32 , TX_S32 , _floor, TX_CONVERT_FLOOR)
GENERATE_TX_CONVERT(fp32, TX_FP32, s32 , TX_S32 , _ceil , TX_CONVERT_CEIL)
GENERATE_TX_CONVERT(s32 , TX_S32 , fp32, TX_FP32,       , TX_CONVERT_ROUND)

#undef GENERATE_TX_CONVERT

#define GENERATE_TX_FP_REDUCE_TYPE(opname, op, typename, type) \
template <long tcfgrs1, long tcfgrs3, long dim> \
__attribute__((always_inline)) \
static inline void __tx_fp_reduce_##opname##_##typename(void* rs1,  void* rs3) { \
NPUInstBuilder::__tx_fp_reduce_op_type< \
    NPUInstBuilder::op, \
    NPUInstBuilder::type, \
    tcfgrs1, tcfgrs3, dim \
    >(rs1, rs3); \
}

GENERATE_TX_FP_REDUCE_TYPE(max, TX_FP_MAX, fp32, TX_FP32)
GENERATE_TX_FP_REDUCE_TYPE(min, TX_FP_MIN, fp32, TX_FP32)
GENERATE_TX_FP_REDUCE_TYPE(add, TX_FP_SUM, fp32, TX_FP32)

#endif

#endif