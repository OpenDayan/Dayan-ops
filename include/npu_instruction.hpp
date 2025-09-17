#ifndef _NPU_INSTRUCTION_HPP_
#define _NPU_INSTRUCTION_HPP_

// public
#define URAM_SIZE 1024 * 256 // 256KB
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

#ifdef __EMULATED_RVT_EXT__
// #include <simsimd/simsimd.h>
#define MAX_KTILE 256
#define MAX_MTILE 32
#define MAX_NTILE 32
extern "C" void exit(int code);
#define ASSERT(x) \
    do { \
        if (!(x)) { \
            printf("Assertion failed: %s\n", #x); \
            exit(1); \
        } \
    } while (0)

class RVTEnv {
    public:
        char* aram;
        b char* uram;
        long* tcsr;
        TCFG* tcfg;
        template <typename T>
        T* get_ptr(void* raw_ptr) {
            long lraw_ptr = (long)raw_ptr;
            if(lraw_ptr & (1ul << 31)) {
                lraw_ptr = lraw_ptr & 0x1fffffff; // No High 4its
                ASSERT(lraw_ptr < URAM_SIZE);
                return (T*)(aram + lraw_ptr);
            } else {
                ASSERT(lraw_ptr < URAM_SIZE);
                return (T*)(uram + lraw_ptr);
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
        // printf("Note: temporarly false implementation of __tx_matmul_subfloat_e4m3_e4m3_f32\n");
        ASSERT(tcfg3->stride1 == tcfg3->dim0);
        memcpy(cExp, c, sizeof(float) * tcfg3->dim0 * tcfg3->dim1);
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
static inline void* __emu_get_uram_ptr(long in) {
    return (void*)(__rvt_env->get_ptr<char>((void*)in));
}

template <typename T>
static inline void __emu_raw_copy(long src, long dst, TCFG* tcfgsrc, TCFG* tcfgdst, void*(* get_src_ptr)(long), void*(* get_dst_ptr)(long)) {
    ASSERT(tcfgsrc->dim0 * tcfgsrc->dim1 * tcfgsrc->dim2 * tcfgsrc->dim3 <= URAM_SIZE);
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
static inline void __tx_async_uram_load(void* rs1, void* rs3) {
    TCFG* tcfg1 = __rvt_env->tcfg + tcfgrs1;
    TCFG* tcfg3 = __rvt_env->tcfg + tcfgrs3;
    static_assert(log2w >= 0 && log2w <= 3, "log2w must be between 0 and 3");
    if(log2w == 0) __emu_raw_copy<char>((long)rs1, (long)rs3, tcfg1, tcfg3,  &__emu_get_dram_ptr, &__emu_get_uram_ptr);
    if(log2w == 1) __emu_raw_copy<short>((long)rs1, (long)rs3, tcfg1, tcfg3, &__emu_get_dram_ptr, &__emu_get_uram_ptr);
    if(log2w == 2) __emu_raw_copy<int>((long)rs1, (long)rs3, tcfg1, tcfg3,   &__emu_get_dram_ptr, &__emu_get_uram_ptr);
    if(log2w == 3) __emu_raw_copy<long>((long)rs1, (long)rs3, tcfg1, tcfg3,  &__emu_get_dram_ptr, &__emu_get_uram_ptr);
}
template <long tcfgrs1, long tcfgrs3, long log2w>
static inline void __tx_async_uram_load(void* rs1, void* rs3, long& rd) {
    __tx_async_uram_load<tcfgrs1, tcfgrs3, log2w>(rs1, rs3, rd);
}

template <long tcfgrs1, long tcfgrs3, long log2w>
static inline void __tx_async_uram_store(void* rs1, void* rs3, long& rd) {
    TCFG* tcfg1 = __rvt_env->tcfg + tcfgrs1;
    TCFG* tcfg3 = __rvt_env->tcfg + tcfgrs3;
    static_assert(log2w >= 0 && log2w <= 3, "log2w must be between 0 and 3");
    if(log2w == 0) __emu_raw_copy<char>((long)rs1, (long)rs3, tcfg1, tcfg3,  &__emu_get_uram_ptr, &__emu_get_dram_ptr);
    if(log2w == 1) __emu_raw_copy<short>((long)rs1, (long)rs3, tcfg1, tcfg3, &__emu_get_uram_ptr, &__emu_get_dram_ptr);
    if(log2w == 2) __emu_raw_copy<int>((long)rs1, (long)rs3, tcfg1, tcfg3,   &__emu_get_uram_ptr, &__emu_get_dram_ptr);
    if(log2w == 3) __emu_raw_copy<long>((long)rs1, (long)rs3, tcfg1, tcfg3,  &__emu_get_uram_ptr, &__emu_get_dram_ptr);
}
template <long tcfgrs1, long tcfgrs3, long log2w>
static inline void __tx_async_uram_store(void* rs1, void* rs3) {
    long rd = 0;
    __tx_async_uram_store<tcfgrs1, tcfgrs3, log2w>(rs1, rs3, rd);
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
    static inline void __custom_32b_instruction3r(long rs1, long rs2, long rs3)
    {
        asm volatile (
            "\n"
            ".ifc %0,zero\n.set _R0,0\n.endif\n.ifc %0,ra\n.set _R0,1\n.endif\n.ifc %0,sp\n.set _R0,2\n.endif\n.ifc %0,gp\n.set _R0,3\n.endif\n.ifc %0,tp\n.set _R0,4\n.endif\n.ifc %0,t0\n.set _R0,5\n.endif\n.ifc %0,t1\n.set _R0,6\n.endif\n.ifc %0,t2\n.set _R0,7\n.endif\n.ifc %0,s0\n.set _R0,8\n.endif\n.ifc %0,s1\n.set _R0,9\n.endif\n.ifc %0,a0\n.set _R0,10\n.endif\n.ifc %0,a1\n.set _R0,11\n.endif\n.ifc %0,a2\n.set _R0,12\n.endif\n.ifc %0,a3\n.set _R0,13\n.endif\n.ifc %0,a4\n.set _R0,14\n.endif\n.ifc %0,a5\n.set _R0,15\n.endif\n.ifc %0,a6\n.set _R0,16\n.endif\n.ifc %0,a7\n.set _R0,17\n.endif\n.ifc %0,s2\n.set _R0,18\n.endif\n.ifc %0,s3\n.set _R0,19\n.endif\n.ifc %0,s4\n.set _R0,20\n.endif\n.ifc %0,s5\n.set _R0,21\n.endif\n.ifc %0,s6\n.set _R0,22\n.endif\n.ifc %0,s7\n.set _R0,23\n.endif\n.ifc %0,s8\n.set _R0,24\n.endif\n.ifc %0,s9\n.set _R0,25\n.endif\n.ifc %0,s10\n.set _R0,26\n.endif\n.ifc %0,s11\n.set _R0,27\n.endif\n.ifc %0,t3\n.set _R0,28\n.endif\n.ifc %0,t4\n.set _R0,29\n.endif\n.ifc %0,t5\n.set _R0,30\n.endif\n.ifc %0,t6\n.set _R0,31\n.endif\n.ifc %1,zero\n.set _R1,0\n.endif\n.ifc %1,ra\n.set _R1,1\n.endif\n.ifc %1,sp\n.set _R1,2\n.endif\n.ifc %1,gp\n.set _R1,3\n.endif\n.ifc %1,tp\n.set _R1,4\n.endif\n.ifc %1,t0\n.set _R1,5\n.endif\n.ifc %1,t1\n.set _R1,6\n.endif\n.ifc %1,t2\n.set _R1,7\n.endif\n.ifc %1,s0\n.set _R1,8\n.endif\n.ifc %1,s1\n.set _R1,9\n.endif\n.ifc %1,a0\n.set _R1,10\n.endif\n.ifc %1,a1\n.set _R1,11\n.endif\n.ifc %1,a2\n.set _R1,12\n.endif\n.ifc %1,a3\n.set _R1,13\n.endif\n.ifc %1,a4\n.set _R1,14\n.endif\n.ifc %1,a5\n.set _R1,15\n.endif\n.ifc %1,a6\n.set _R1,16\n.endif\n.ifc %1,a7\n.set _R1,17\n.endif\n.ifc %1,s2\n.set _R1,18\n.endif\n.ifc %1,s3\n.set _R1,19\n.endif\n.ifc %1,s4\n.set _R1,20\n.endif\n.ifc %1,s5\n.set _R1,21\n.endif\n.ifc %1,s6\n.set _R1,22\n.endif\n.ifc %1,s7\n.set _R1,23\n.endif\n.ifc %1,s8\n.set _R1,24\n.endif\n.ifc %1,s9\n.set _R1,25\n.endif\n.ifc %1,s10\n.set _R1,26\n.endif\n.ifc %1,s11\n.set _R1,27\n.endif\n.ifc %1,t3\n.set _R1,28\n.endif\n.ifc %1,t4\n.set _R1,29\n.endif\n.ifc %1,t5\n.set _R1,30\n.endif\n.ifc %1,t6\n.set _R1,31\n.endif\n.ifc %2,zero\n.set _R2,0\n.endif\n.ifc %2,ra\n.set _R2,1\n.endif\n.ifc %2,sp\n.set _R2,2\n.endif\n.ifc %2,gp\n.set _R2,3\n.endif\n.ifc %2,tp\n.set _R2,4\n.endif\n.ifc %2,t0\n.set _R2,5\n.endif\n.ifc %2,t1\n.set _R2,6\n.endif\n.ifc %2,t2\n.set _R2,7\n.endif\n.ifc %2,s0\n.set _R2,8\n.endif\n.ifc %2,s1\n.set _R2,9\n.endif\n.ifc %2,a0\n.set _R2,10\n.endif\n.ifc %2,a1\n.set _R2,11\n.endif\n.ifc %2,a2\n.set _R2,12\n.endif\n.ifc %2,a3\n.set _R2,13\n.endif\n.ifc %2,a4\n.set _R2,14\n.endif\n.ifc %2,a5\n.set _R2,15\n.endif\n.ifc %2,a6\n.set _R2,16\n.endif\n.ifc %2,a7\n.set _R2,17\n.endif\n.ifc %2,s2\n.set _R2,18\n.endif\n.ifc %2,s3\n.set _R2,19\n.endif\n.ifc %2,s4\n.set _R2,20\n.endif\n.ifc %2,s5\n.set _R2,21\n.endif\n.ifc %2,s6\n.set _R2,22\n.endif\n.ifc %2,s7\n.set _R2,23\n.endif\n.ifc %2,s8\n.set _R2,24\n.endif\n.ifc %2,s9\n.set _R2,25\n.endif\n.ifc %2,s10\n.set _R2,26\n.endif\n.ifc %2,s11\n.set _R2,27\n.endif\n.ifc %2,t3\n.set _R2,28\n.endif\n.ifc %2,t4\n.set _R2,29\n.endif\n.ifc %2,t5\n.set _R2,30\n.endif\n.ifc %2,t6\n.set _R2,31\n.endif\n"
            ".align 2\n"
            ".insn 4, %3 | 0xb | (_R0 << 7) | (_R1 << 15) | (_R2 << 20)\n"
            ".set _R0,z\n.set _R1,z\n.set _R2,z\n"
            ::"r"(rs1), "r"(rs2), "r"(rs3), "i"(encoding)
        );
    }
    template <long encoding>
    __attribute__((always_inline))
    static inline void __custom_32b_instruction2r(long rs1, long rs2)
    {
        asm volatile (
            "\n"
            ".ifc %0,zero\n.set _R0,0\n.endif\n.ifc %0,ra\n.set _R0,1\n.endif\n.ifc %0,sp\n.set _R0,2\n.endif\n.ifc %0,gp\n.set _R0,3\n.endif\n.ifc %0,tp\n.set _R0,4\n.endif\n.ifc %0,t0\n.set _R0,5\n.endif\n.ifc %0,t1\n.set _R0,6\n.endif\n.ifc %0,t2\n.set _R0,7\n.endif\n.ifc %0,s0\n.set _R0,8\n.endif\n.ifc %0,s1\n.set _R0,9\n.endif\n.ifc %0,a0\n.set _R0,10\n.endif\n.ifc %0,a1\n.set _R0,11\n.endif\n.ifc %0,a2\n.set _R0,12\n.endif\n.ifc %0,a3\n.set _R0,13\n.endif\n.ifc %0,a4\n.set _R0,14\n.endif\n.ifc %0,a5\n.set _R0,15\n.endif\n.ifc %0,a6\n.set _R0,16\n.endif\n.ifc %0,a7\n.set _R0,17\n.endif\n.ifc %0,s2\n.set _R0,18\n.endif\n.ifc %0,s3\n.set _R0,19\n.endif\n.ifc %0,s4\n.set _R0,20\n.endif\n.ifc %0,s5\n.set _R0,21\n.endif\n.ifc %0,s6\n.set _R0,22\n.endif\n.ifc %0,s7\n.set _R0,23\n.endif\n.ifc %0,s8\n.set _R0,24\n.endif\n.ifc %0,s9\n.set _R0,25\n.endif\n.ifc %0,s10\n.set _R0,26\n.endif\n.ifc %0,s11\n.set _R0,27\n.endif\n.ifc %0,t3\n.set _R0,28\n.endif\n.ifc %0,t4\n.set _R0,29\n.endif\n.ifc %0,t5\n.set _R0,30\n.endif\n.ifc %0,t6\n.set _R0,31\n.endif\n.ifc %1,zero\n.set _R1,0\n.endif\n.ifc %1,ra\n.set _R1,1\n.endif\n.ifc %1,sp\n.set _R1,2\n.endif\n.ifc %1,gp\n.set _R1,3\n.endif\n.ifc %1,tp\n.set _R1,4\n.endif\n.ifc %1,t0\n.set _R1,5\n.endif\n.ifc %1,t1\n.set _R1,6\n.endif\n.ifc %1,t2\n.set _R1,7\n.endif\n.ifc %1,s0\n.set _R1,8\n.endif\n.ifc %1,s1\n.set _R1,9\n.endif\n.ifc %1,a0\n.set _R1,10\n.endif\n.ifc %1,a1\n.set _R1,11\n.endif\n.ifc %1,a2\n.set _R1,12\n.endif\n.ifc %1,a3\n.set _R1,13\n.endif\n.ifc %1,a4\n.set _R1,14\n.endif\n.ifc %1,a5\n.set _R1,15\n.endif\n.ifc %1,a6\n.set _R1,16\n.endif\n.ifc %1,a7\n.set _R1,17\n.endif\n.ifc %1,s2\n.set _R1,18\n.endif\n.ifc %1,s3\n.set _R1,19\n.endif\n.ifc %1,s4\n.set _R1,20\n.endif\n.ifc %1,s5\n.set _R1,21\n.endif\n.ifc %1,s6\n.set _R1,22\n.endif\n.ifc %1,s7\n.set _R1,23\n.endif\n.ifc %1,s8\n.set _R1,24\n.endif\n.ifc %1,s9\n.set _R1,25\n.endif\n.ifc %1,s10\n.set _R1,26\n.endif\n.ifc %1,s11\n.set _R1,27\n.endif\n.ifc %1,t3\n.set _R1,28\n.endif\n.ifc %1,t4\n.set _R1,29\n.endif\n.ifc %1,t5\n.set _R1,30\n.endif\n.ifc %1,t6\n.set _R1,31\n.endif\n"
            ".align 2\n"
            ".insn 4, %2 | 0xb | (_R0 << 7) | (_R1 << 15)\n"
            ".set _R0,z\n.set _R1,z\n"
            ::"r"(rs1), "r"(rs2), "i"(encoding)
        );
    }
    template <long encoding>
    __attribute__((always_inline))
    static inline void __custom_32b_instruction2r_write(long rs1, long& rd)
    {
        asm volatile (
            "\n"
            ".ifc %0,zero\n.set _R0,0\n.endif\n.ifc %0,ra\n.set _R0,1\n.endif\n.ifc %0,sp\n.set _R0,2\n.endif\n.ifc %0,gp\n.set _R0,3\n.endif\n.ifc %0,tp\n.set _R0,4\n.endif\n.ifc %0,t0\n.set _R0,5\n.endif\n.ifc %0,t1\n.set _R0,6\n.endif\n.ifc %0,t2\n.set _R0,7\n.endif\n.ifc %0,s0\n.set _R0,8\n.endif\n.ifc %0,s1\n.set _R0,9\n.endif\n.ifc %0,a0\n.set _R0,10\n.endif\n.ifc %0,a1\n.set _R0,11\n.endif\n.ifc %0,a2\n.set _R0,12\n.endif\n.ifc %0,a3\n.set _R0,13\n.endif\n.ifc %0,a4\n.set _R0,14\n.endif\n.ifc %0,a5\n.set _R0,15\n.endif\n.ifc %0,a6\n.set _R0,16\n.endif\n.ifc %0,a7\n.set _R0,17\n.endif\n.ifc %0,s2\n.set _R0,18\n.endif\n.ifc %0,s3\n.set _R0,19\n.endif\n.ifc %0,s4\n.set _R0,20\n.endif\n.ifc %0,s5\n.set _R0,21\n.endif\n.ifc %0,s6\n.set _R0,22\n.endif\n.ifc %0,s7\n.set _R0,23\n.endif\n.ifc %0,s8\n.set _R0,24\n.endif\n.ifc %0,s9\n.set _R0,25\n.endif\n.ifc %0,s10\n.set _R0,26\n.endif\n.ifc %0,s11\n.set _R0,27\n.endif\n.ifc %0,t3\n.set _R0,28\n.endif\n.ifc %0,t4\n.set _R0,29\n.endif\n.ifc %0,t5\n.set _R0,30\n.endif\n.ifc %0,t6\n.set _R0,31\n.endif\n.ifc %1,zero\n.set _R1,0\n.endif\n.ifc %1,ra\n.set _R1,1\n.endif\n.ifc %1,sp\n.set _R1,2\n.endif\n.ifc %1,gp\n.set _R1,3\n.endif\n.ifc %1,tp\n.set _R1,4\n.endif\n.ifc %1,t0\n.set _R1,5\n.endif\n.ifc %1,t1\n.set _R1,6\n.endif\n.ifc %1,t2\n.set _R1,7\n.endif\n.ifc %1,s0\n.set _R1,8\n.endif\n.ifc %1,s1\n.set _R1,9\n.endif\n.ifc %1,a0\n.set _R1,10\n.endif\n.ifc %1,a1\n.set _R1,11\n.endif\n.ifc %1,a2\n.set _R1,12\n.endif\n.ifc %1,a3\n.set _R1,13\n.endif\n.ifc %1,a4\n.set _R1,14\n.endif\n.ifc %1,a5\n.set _R1,15\n.endif\n.ifc %1,a6\n.set _R1,16\n.endif\n.ifc %1,a7\n.set _R1,17\n.endif\n.ifc %1,s2\n.set _R1,18\n.endif\n.ifc %1,s3\n.set _R1,19\n.endif\n.ifc %1,s4\n.set _R1,20\n.endif\n.ifc %1,s5\n.set _R1,21\n.endif\n.ifc %1,s6\n.set _R1,22\n.endif\n.ifc %1,s7\n.set _R1,23\n.endif\n.ifc %1,s8\n.set _R1,24\n.endif\n.ifc %1,s9\n.set _R1,25\n.endif\n.ifc %1,s10\n.set _R1,26\n.endif\n.ifc %1,s11\n.set _R1,27\n.endif\n.ifc %1,t3\n.set _R1,28\n.endif\n.ifc %1,t4\n.set _R1,29\n.endif\n.ifc %1,t5\n.set _R1,30\n.endif\n.ifc %1,t6\n.set _R1,31\n.endif\n"
            ".align 2\n"
            ".insn 4, %2 | 0xb | (_R0 << 7) | (_R1 << 15)\n"
            ".set _R0,z\n.set _R1,z\n"
            :"=r"(rd)
            :"r"(rs1), "i"(encoding)
        );
    }
    template <long encoding>
    __attribute__((always_inline))
    static inline void __custom_32b_instruction1r(long rs, long& rd)
    {
        asm volatile (
           "\n"
           ".ifc %0,zero\n.set _R0,0\n.endif\n.ifc %0,ra\n.set _R0,1\n.endif\n.ifc %0,sp\n.set _R0,2\n.endif\n.ifc %0,gp\n.set _R0,3\n.endif\n.ifc %0,tp\n.set _R0,4\n.endif\n.ifc %0,t0\n.set _R0,5\n.endif\n.ifc %0,t1\n.set _R0,6\n.endif\n.ifc %0,t2\n.set _R0,7\n.endif\n.ifc %0,s0\n.set _R0,8\n.endif\n.ifc %0,s1\n.set _R0,9\n.endif\n.ifc %0,a0\n.set _R0,10\n.endif\n.ifc %0,a1\n.set _R0,11\n.endif\n.ifc %0,a2\n.set _R0,12\n.endif\n.ifc %0,a3\n.set _R0,13\n.endif\n.ifc %0,a4\n.set _R0,14\n.endif\n.ifc %0,a5\n.set _R0,15\n.endif\n.ifc %0,a6\n.set _R0,16\n.endif\n.ifc %0,a7\n.set _R0,17\n.endif\n.ifc %0,s2\n.set _R0,18\n.endif\n.ifc %0,s3\n.set _R0,19\n.endif\n.ifc %0,s4\n.set _R0,20\n.endif\n.ifc %0,s5\n.set _R0,21\n.endif\n.ifc %0,s6\n.set _R0,22\n.endif\n.ifc %0,s7\n.set _R0,23\n.endif\n.ifc %0,s8\n.set _R0,24\n.endif\n.ifc %0,s9\n.set _R0,25\n.endif\n.ifc %0,s10\n.set _R0,26\n.endif\n.ifc %0,s11\n.set _R0,27\n.endif\n.ifc %0,t3\n.set _R0,28\n.endif\n.ifc %0,t4\n.set _R0,29\n.endif\n.ifc %0,t5\n.set _R0,30\n.endif\n.ifc %0,t6\n.set _R0,31\n.endif\n"
            ".align 2\n"
            ".insn 4, %1 | 0xb | (_R0 << 7)\n"
            ".set _R0,z\n"
            :"=r"(rd)
            :"i"(encoding)
        );
    }
    template <long encoding>
    __attribute__((always_inline))
    static inline void __custom_32b_instruction1r(long rs1)
    {
        asm volatile (
            "\n"
            ".ifc %0,zero\n.set _R0,0\n.endif\n.ifc %0,ra\n.set _R0,1\n.endif\n.ifc %0,sp\n.set _R0,2\n.endif\n.ifc %0,gp\n.set _R0,3\n.endif\n.ifc %0,tp\n.set _R0,4\n.endif\n.ifc %0,t0\n.set _R0,5\n.endif\n.ifc %0,t1\n.set _R0,6\n.endif\n.ifc %0,t2\n.set _R0,7\n.endif\n.ifc %0,s0\n.set _R0,8\n.endif\n.ifc %0,s1\n.set _R0,9\n.endif\n.ifc %0,a0\n.set _R0,10\n.endif\n.ifc %0,a1\n.set _R0,11\n.endif\n.ifc %0,a2\n.set _R0,12\n.endif\n.ifc %0,a3\n.set _R0,13\n.endif\n.ifc %0,a4\n.set _R0,14\n.endif\n.ifc %0,a5\n.set _R0,15\n.endif\n.ifc %0,a6\n.set _R0,16\n.endif\n.ifc %0,a7\n.set _R0,17\n.endif\n.ifc %0,s2\n.set _R0,18\n.endif\n.ifc %0,s3\n.set _R0,19\n.endif\n.ifc %0,s4\n.set _R0,20\n.endif\n.ifc %0,s5\n.set _R0,21\n.endif\n.ifc %0,s6\n.set _R0,22\n.endif\n.ifc %0,s7\n.set _R0,23\n.endif\n.ifc %0,s8\n.set _R0,24\n.endif\n.ifc %0,s9\n.set _R0,25\n.endif\n.ifc %0,s10\n.set _R0,26\n.endif\n.ifc %0,s11\n.set _R0,27\n.endif\n.ifc %0,t3\n.set _R0,28\n.endif\n.ifc %0,t4\n.set _R0,29\n.endif\n.ifc %0,t5\n.set _R0,30\n.endif\n.ifc %0,t6\n.set _R0,31\n.endif\n"
            ".align 2\n"
            ".insn 4, %1 | 0xb | (_R0 << 7)\n"
            ".set _R0,z\n"
            ::"r"(rs1), "i"(encoding)
        );
    }
    template <long encoding>
    __attribute__((always_inline))
    static inline void __custom_32b_instruction1r_pos2(long rs1)
    {
        asm volatile (
            "\n"
            ".ifc %0,zero\n.set _R0,0\n.endif\n.ifc %0,ra\n.set _R0,1\n.endif\n.ifc %0,sp\n.set _R0,2\n.endif\n.ifc %0,gp\n.set _R0,3\n.endif\n.ifc %0,tp\n.set _R0,4\n.endif\n.ifc %0,t0\n.set _R0,5\n.endif\n.ifc %0,t1\n.set _R0,6\n.endif\n.ifc %0,t2\n.set _R0,7\n.endif\n.ifc %0,s0\n.set _R0,8\n.endif\n.ifc %0,s1\n.set _R0,9\n.endif\n.ifc %0,a0\n.set _R0,10\n.endif\n.ifc %0,a1\n.set _R0,11\n.endif\n.ifc %0,a2\n.set _R0,12\n.endif\n.ifc %0,a3\n.set _R0,13\n.endif\n.ifc %0,a4\n.set _R0,14\n.endif\n.ifc %0,a5\n.set _R0,15\n.endif\n.ifc %0,a6\n.set _R0,16\n.endif\n.ifc %0,a7\n.set _R0,17\n.endif\n.ifc %0,s2\n.set _R0,18\n.endif\n.ifc %0,s3\n.set _R0,19\n.endif\n.ifc %0,s4\n.set _R0,20\n.endif\n.ifc %0,s5\n.set _R0,21\n.endif\n.ifc %0,s6\n.set _R0,22\n.endif\n.ifc %0,s7\n.set _R0,23\n.endif\n.ifc %0,s8\n.set _R0,24\n.endif\n.ifc %0,s9\n.set _R0,25\n.endif\n.ifc %0,s10\n.set _R0,26\n.endif\n.ifc %0,s11\n.set _R0,27\n.endif\n.ifc %0,t3\n.set _R0,28\n.endif\n.ifc %0,t4\n.set _R0,29\n.endif\n.ifc %0,t5\n.set _R0,30\n.endif\n.ifc %0,t6\n.set _R0,31\n.endif\n"
            ".align 2\n"
            ".insn 4, %1 | 0xb | (_R0 << 20)\n"
            ".set _R0,z\n"
            ::"r"(rs1), "i"(encoding)
        );
    }
    template <long encoding>
    __attribute__((always_inline))
    static inline void __custom_32b_instruction0r(void)
    {
        asm volatile (
            ".align 2\n"
            ".insn 4, %0 | 0xb\n"
            ::"i"(encoding)
        );
    }
    enum __TX_VEC_VV_INT32_OP_ENUM {
        TX_INT32_ROL = 0,
        TX_INT32_SLL = 1,
        TX_INT32_SRA = 2,
        TX_INT32_SRL = 3,
        TX_INT32_ADD = 4,
        TX_INT32_SUB = 5,
        TX_INT32_MUL = 6,
    };
    template <long vb, long vrs2, long op, long vrs1>
    __attribute__((always_inline))
    static inline void __tx_vec_vv_int32_op(void)
    {
        NPUInstBuilder::__custom_32b_instruction0r<
            (0b100001ul << 26ul) | 
            (vb << 25ul)         |
            (vrs2 << 15ul)       |
            (op << 12ul)       |
            (vrs1 << 7ul)       
        >();
    }

    enum __TX_VEC_VV_FLOAT32_OP_ENUM {
        TX_FLOAT32_ADD = 0,
        TX_FLOAT32_SUB = 1,
        TX_FLOAT32_MUL = 2,
    };
    template <long vb, long vrs2, long op, long vrs1>
    __attribute__((always_inline))
    static inline void __tx_vec_vv_float32_op(void)
    {
        NPUInstBuilder::__custom_32b_instruction0r<
            (0b110000ul << 26ul) | 
            (vb << 25ul)         |
            (vrs2 << 15ul)       |
            (op << 12ul)       |
            (vrs1 << 7ul)       
        >();
    }

    enum __TX_VEC_V_INT32_OP_ENUM {
        TX_VEC_V_INT32_CMP = 0,
        TX_VEC_V_INT32_SLL = 1,
        TX_VEC_V_INT32_SRA = 2,
        TX_VEC_V_INT32_SRL = 3,
        TX_VEC_V_INT32_REDUCE = 4,
    };
    template <long vb, long sft, long op, long vrs1>
    __attribute__((always_inline))
    static inline void __tx_vec_v_int32_op(void)
    {
        NPUInstBuilder::__custom_32b_instruction0r<
            (0b1ul << 31ul)  |
            (vb << 25ul)     |
            (0b1ul << 20ul)  |
            (op << 25ul)     |
            (sft << 12ul)    | 
            (vrs1 << 7ul)  
        >();
    }

    enum __TX_VEC_V_FLOAT32_OP_ENUM {
        TX_VEC_V_FLOAT32_CMP = 0,
        TX_VEC_V_FLOAT32_SFU = 1,
        TX_VEC_V_FLOAT32_REDUCE = 2,
    };
    template <long vb, long op, long imm, long vrs1>
    __attribute__((always_inline))
    static inline void __tx_vec_v_float32_op(void)
    {
        NPUInstBuilder::__custom_32b_instruction0r<
            (0b1ul << 31ul)  |
            (vb << 25ul)     |
            (0b1ul << 24ul)  |
            (op << 18ul)     |
            (imm << 12ul)    | 
            (vrs1 << 7ul)     
        >();
    }

    enum __TX_VEC_V_COMB_SEP_OP_ENUM {
        COMB_2X2B = 0,
        COMB_1X4B = 1,
        SEP_2X2B = 2,
        SEP_1X4B = 3,
    };
    template <long vb, long op, long imm, long vrs1>
    __attribute__((always_inline))
    static inline void __tx_vec_v_op(void)
    {
        NPUInstBuilder::__custom_32b_instruction0r<
            (0b1ul << 31ul)     |
            (vb << 25ul)        |
            (0b1111ul << 20ul)  |
            (op << 13ul)        |
            (imm << 12ul)       | 
            (vrs1 << 7ul)     
        >();
    }

    enum __TX_VEC_V_CVT_ENUM {
        FLOAT32_INT32 = 0,
        INT32_FLOAT32 = 1,
        FLOAT32_FP16  = 2,
        FP16_FLOAT32  = 3,
        FLOAT32_E4M3  = 4,
        E4M3_FLOAT32  = 5,
    };
    template <long vb, long type, long vrs1>
    __attribute__((always_inline))
    static inline void __tx_vec_v_cvt(long& rd)
    {
        NPUInstBuilder::__custom_32b_instruction0r<
            (0b100000ul << 26ul) | 
            (vb << 25ul)         |
            (type << 12ul)       |
            (0b11111ul << 20ul)  |
            (vrs1 << 7ul)       
        >();
    }
}

#define LSU_MASK (0b1ul << 29ul)
__attribute__((always_inline))
static inline void __tx_lsu_dma_load(void* rs1, void* rs2) {
        NPUInstBuilder::__custom_32b_instruction2r<
            LSU_MASK>((long)rs1, (long)rs2);
}
__attribute__((always_inline))
static inline void __tx_lsu_dma_store(void* rs1, void* rs2) {
        NPUInstBuilder::__custom_32b_instruction2r<
            (LSU_MASK | (0b001ul << 12ul))>((long)rs1, (long)rs2);
}

#define CONFIG_MASK (unit << 25ul) | (tcsrid << 15ul)
template <long unit, long tcsrid>
__attribute__((always_inline))
static inline void __tx_unit_tcsr_write(long rs1) {
    NPUInstBuilder::__custom_32b_instruction1r<
        CONFIG_MASK | (0b0ul << 14ul)>(rs1);
}
template <long unit, long tcsrid>
__attribute__((always_inline))
static inline void __tx_unit_tcsr_read(long &rd) {
    NPUInstBuilder::__custom_32b_instruction1r<
        CONFIG_MASK | (0b1ul << 14ul)>(0, rd);
}

#define SCOREBOARD_MASK (0b1ul << 27ul) | (unit << 25ul)
template <long unit>
__attribute__((always_inline))
static inline void __tx_unit_sb_produce(long rs1) {
    NPUInstBuilder::__custom_32b_instruction1r<
        SCOREBOARD_MASK | (0b000ul << 12ul)>(rs1);
}
template <long unit>
__attribute__((always_inline))
static inline void __tx_unit_sb_reset(long rs1) {
    NPUInstBuilder::__custom_32b_instruction1r<
        SCOREBOARD_MASK | (0b001ul << 12ul)>(rs1);
}
template <long unit>
__attribute__((always_inline))
static inline void __tx_unit_sb_wait(long rs1) {
    NPUInstBuilder::__custom_32b_instruction1r<
        SCOREBOARD_MASK | (0b010ul << 12ul)>(rs1);
}
template <long unit>
__attribute__((always_inline))
static inline void __tx_unit_get_any(long& rd) {
    NPUInstBuilder::__custom_32b_instruction1r<
        SCOREBOARD_MASK | (0b100ul << 12ul)>(0, rd);
}
template <long unit>
__attribute__((always_inline))
static inline void __tx_unit_wait_any(long& rd) {
    NPUInstBuilder::__custom_32b_instruction1r<
        SCOREBOARD_MASK | (0b101ul << 12ul)>(0, rd);
}
template <long unit>
__attribute__((always_inline))
static inline void __tx_unit_sb_get(long rs1, long& rd) {
    NPUInstBuilder::__custom_32b_instruction2r_write<
        SCOREBOARD_MASK | (0b100ul << 12ul)>(rs1, rd);
}

#define VEC_LOAD_MASK (0b1ul << 30ul) | (vb << 25ul)
template <long vb>
__attribute__((always_inline))
static inline void __tx_vec_v_load(long rs) {
    NPUInstBuilder::__custom_32b_instruction1r_pos2<
        VEC_LOAD_MASK | (0b000ul << 15ul)>(rs);
}
template <long vb>
__attribute__((always_inline))
static inline void __tx_vec_s_load(long rs) {
    NPUInstBuilder::__custom_32b_instruction1r_pos2<
        VEC_LOAD_MASK | (0b001ul << 15ul)>(rs);
}
template <long vb>
__attribute__((always_inline))
static inline void __tx_vec_const_load(long rs) {
    NPUInstBuilder::__custom_32b_instruction1r_pos2<
        VEC_LOAD_MASK | (0b010ul << 15ul)>(rs);
}
template <long vb, long vrs1>
__attribute__((always_inline))
static inline void __tx_vec_v_gather64(long rs) {
    NPUInstBuilder::__custom_32b_instruction1r_pos2<
        VEC_LOAD_MASK | (0b100ul << 15ul) | (vrs1 << 7ul)>(rs);
}
template <long vb, long cond, long vrs1>
__attribute__((always_inline))
static inline void __tx_vec_v_gather64_mask(long rs) {
    NPUInstBuilder::__custom_32b_instruction1r_pos2<
        VEC_LOAD_MASK | (0b101ul << 15ul) | (cond << 12ul) | (vrs1 << 7ul)>(rs);
}

#define VEC_STORE_MASK (0b1ul << 30ul) | (vrs2 << 15ul)
template <long vrs2>
__attribute__((always_inline))
static inline void __tx_vec_v_store(long rs) {
    NPUInstBuilder::__custom_32b_instruction1r_pos2<
        VEC_STORE_MASK | (0b010ul << 25ul)>(rs);
}
template <long vrs2, long cond>
__attribute__((always_inline))
static inline void __tx_vec_v_store_mask(long rs) {
    NPUInstBuilder::__custom_32b_instruction1r_pos2<
        VEC_STORE_MASK | (0b010ul << 25ul) | (cond << 12ul)>(rs); 
}
template <long vrs2>
__attribute__((always_inline))
static inline void __tx_vec_s_store(long rs) {
    NPUInstBuilder::__custom_32b_instruction1r_pos2<
        VEC_STORE_MASK | (0b010ul << 25ul)>(rs); 
}
template <long vrs1, long vrs2>
__attribute__((always_inline))
static inline void __tx_vec_v_scatter64(long rs) {
    NPUInstBuilder::__custom_32b_instruction1r_pos2<
        VEC_STORE_MASK | (0b100ul << 25ul) | (vrs1 << 7ul)>(rs); 
}
template <long vrs1, long vrs2, long cond>
__attribute__((always_inline))
static inline void __tx_vec_v_scatter64_mask(long rs) {
    NPUInstBuilder::__custom_32b_instruction1r_pos2<
        VEC_STORE_MASK | (0b101ul << 25ul) | (cond << 12ul) | (vrs1 << 7ul)>(rs);
}

#define GENERATE_TX_VEC_VV_INT32_OP(opname, op) \
template <long vb, long vrs2, long vrs1> \
__attribute__((always_inline)) \
static inline void __tx_vec_vv_int32_##opname(void) { \
    NPUInstBuilder::__tx_vec_vv_int32_op< \
        vb, vrs2,  \
        NPUInstBuilder::op, \
        vrs1 \
        >(); \
}
GENERATE_TX_VEC_VV_INT32_OP(rol,     TX_INT32_ROL)
GENERATE_TX_VEC_VV_INT32_OP(sll,     TX_INT32_SLL)
GENERATE_TX_VEC_VV_INT32_OP(sra,     TX_INT32_SRA)
GENERATE_TX_VEC_VV_INT32_OP(srl,     TX_INT32_SRL)
GENERATE_TX_VEC_VV_INT32_OP(add,     TX_INT32_ADD)
GENERATE_TX_VEC_VV_INT32_OP(sub,     TX_INT32_SUB)
GENERATE_TX_VEC_VV_INT32_OP(mul,     TX_INT32_MUL)
template <long vb, long vrs2, long cmp_mask, long vrs1>
__attribute__((always_inline))
static inline void __tx_vec_vv_int32_cmp(void)
{
    NPUInstBuilder::__custom_32b_instruction0r<
        (0b100001ul << 26ul) | 
        (vb << 25ul)         |
        (0b1ul << 20ul)      | 
        (vrs2 << 15ul)       |
        (cmp_mask << 12ul)   |
        (vrs1 << 7ul)       
    >();
}
template <long vb, long vrs2, long l_mask, long vrs1>
__attribute__((always_inline))
static inline void __tx_vec_vv_int32_bitwise(void)
{
    NPUInstBuilder::__custom_32b_instruction0r<
        (0b100001ul << 26ul)          | 
        (vb << 25ul)                  |
        (0b1ul << 21ul)               | 
        ((l_mask >> 3ul) << 20ul)     |
        (vrs2 << 15ul)                |
        ((l_mask & 0b111ul) << 12ul)  |
        (vrs1 << 7ul)       
    >();
}

#define GENERATE_TX_VEC_VV_FLOAT32_OP(opname, op) \
template <long vb, long vrs2, long vrs1> \
__attribute__((always_inline)) \
static inline void __tx_vec_vv_float32_##opname(void) { \
    NPUInstBuilder::__tx_vec_vv_float32_op< \
        vb, vrs2,  \
        NPUInstBuilder::op, \
        vrs1 \
        >(); \
}
GENERATE_TX_VEC_VV_FLOAT32_OP(add,     TX_FLOAT32_ADD)
GENERATE_TX_VEC_VV_FLOAT32_OP(sub,     TX_FLOAT32_SUB)
GENERATE_TX_VEC_VV_FLOAT32_OP(mul,     TX_FLOAT32_MUL)
template <long vb, long vrs2, long cmp_mask, long vrs1>
__attribute__((always_inline))
static inline void __tx_vec_vv_float32_cmp(void)
{
    NPUInstBuilder::__custom_32b_instruction0r<
        (0b110000ul << 26ul) | 
        (vb << 25ul)         |
        (0b1ul << 20ul)      | 
        (vrs2 << 15ul)       |
        (cmp_mask << 12ul)   |
        (vrs1 << 7ul)       
    >();
}

template <long vb, long vrs2, long cond, long vrs1>
__attribute__((always_inline))
static inline void __tx_vec_vv_mu2x(void)
{
    NPUInstBuilder::__custom_32b_instruction0r<
        (0b111111ul << 26ul) | 
        (vb << 25ul)         |
        (vrs2 << 15ul)       |
        (cond << 12ul)       |
        (vrs1 << 7ul)       
    >();
}

template <long l_mask, long cond2, long cond1>
__attribute__((always_inline))
static inline void __tx_vec_cond_bitwise(void)
{
    NPUInstBuilder::__custom_32b_instruction0r<
        (0b1ul << 31ul)      | 
        (l_mask << 13ul)     |
        (cond2 << 10ul)      | 
        (cond1 << 7ul)     
    >();
}

// NOTE: shift -> '<=>' in cmp
//             -> 'reduce op' in reduce
#define GENERATE_TX_VEC_V_INT32(opname, op) \
template <long vb, long shift, long vrs1> \
__attribute__((always_inline)) \
static inline void __tx_vec_v_int32_##opname(void) { \
    NPUInstBuilder::__tx_vec_v_int32_op< \
        vb, \
        shift, \
        NPUInstBuilder::op, \
        vrs1 \
    >(); \
} 
GENERATE_TX_VEC_V_INT32(cmp,     TX_VEC_V_INT32_CMP)
GENERATE_TX_VEC_V_INT32(sll,     TX_VEC_V_INT32_SLL)
GENERATE_TX_VEC_V_INT32(sra,     TX_VEC_V_INT32_SRA)
GENERATE_TX_VEC_V_INT32(srl,     TX_VEC_V_INT32_SRL)
GENERATE_TX_VEC_V_INT32(reduce,  TX_VEC_V_INT32_REDUCE)

#define GENERATE_TX_VEC_V(opname, op) \
template <long vb, long imm, long vrs1> \
__attribute__((always_inline)) \
static inline void __tx_vec_v_##opname(void) { \
    NPUInstBuilder::__tx_vec_v_op< \
        vb, \
        NPUInstBuilder::op, \
        imm, \
        vrs1 \
    >(); \
} 
GENERATE_TX_VEC_V(comb_2x2b,    COMB_2X2B)
GENERATE_TX_VEC_V(comb_1x4b,    COMB_1X4B)
GENERATE_TX_VEC_V(sep_2x2b,     SEP_2X2B)
GENERATE_TX_VEC_V(sep_1x4b,     SEP_1X4B)

// NOTE: imm -> '<=>' in cmp
//           -> 'sfu op' in sfu
//           -> 'reduce op' in reduce
#define GENERATE_TX_VEC_V_FLOAT32(opname, op) \
template <long vb, long imm, long vrs1> \
__attribute__((always_inline)) \
static inline void __tx_vec_v_float32_##opname(void) { \
    NPUInstBuilder::__tx_vec_v_float32_op< \
        vb, \
        NPUInstBuilder::op, \
        imm, \
        vrs1 \
    >(); \
} 
GENERATE_TX_VEC_V_FLOAT32(cmp,        TX_VEC_V_FLOAT32_CMP)
GENERATE_TX_VEC_V_FLOAT32(sfu,        TX_VEC_V_FLOAT32_SFU)
GENERATE_TX_VEC_V_FLOAT32(reduce,     TX_VEC_V_FLOAT32_REDUCE)

#define GENERATE_TX_VEC_V_CVT(typefromname, typetoname, typefrom, typeto) \
template <long vb, long vrs1> \
__attribute__((always_inline)) \
static inline void __tx_vec_v_cvt_##typefromname##_##typetoname(void) { \
    NPUInstBuilder::__tx_vec_v_cvt< \
        vb, \
        NPUInstBuilder::typefrom##_##typeto, \
        vrs1 \
    >(); \
}
GENERATE_TX_VEC_V_CVT(float32, int32, FLOAT32, INT32)
GENERATE_TX_VEC_V_CVT(int32, float32, INT32, FLOAT32)
GENERATE_TX_VEC_V_CVT(float32, fp16, FLOAT32, FP16)
GENERATE_TX_VEC_V_CVT(fp16, float32, FP16, FLOAT32)
GENERATE_TX_VEC_V_CVT(float32, e4m3, FLOAT32, E4M3)
GENERATE_TX_VEC_V_CVT(e4m3, float32, E4M3, FLOAT32)

#define MAT_MASK (0b1ul << 28ul)
__attribute__((always_inline))
static inline void __tx_mat_mma(void* rs1, void* rs2, void* rs3)
{
    NPUInstBuilder::__custom_32b_instruction3r<
    MAT_MASK>
    ((long)rs1, (long)rs2, (long)rs3);
}
__attribute__((always_inline))
static inline void __tx_mat_mm(void* rs1, void* rs2, void* rs3)
{
    NPUInstBuilder::__custom_32b_instruction3r<
    MAT_MASK | (0b1ul << 12ul)>
    ((long)rs1, (long)rs2, (long)rs3);
}

#endif

#endif