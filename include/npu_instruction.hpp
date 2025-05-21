#ifndef _NPU_INSTRUCTION_HPP_
#define _NPU_INSTRUCTION_HPP_

#define NRAM_SIZE 1024 * 256 // 256KB

#ifdef __EMULATED_RVT_EXT__
#include <stdio.h>
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
        char* nram;
        long* tcsr;
        TCFG* tcfg;
        RVTEnv() {
            nram = new char[NRAM_SIZE];
            tcsr = new long[64];
            tcfg = (TCFG*)tcsr;
        }
        ~RVTEnv() {
            delete[] nram;
            delete[] tcsr;
        }
        template <typename T>
        T* get_ptr(void* raw_ptr) {
            long lraw_ptr = (long)raw_ptr;
            ASSERT(lraw_ptr < NRAM_SIZE);
            return &nram[lraw_ptr];
        }
};
extern RVTEnv* __rvt_env;

static inline void __tx_init() {
    __rvt_env = new RVTEnv();
}

static inline void __tx_fence() {
    // Do nothing
}

template <long tcfgrs1, long tcfgrs2, long tcfgrs3>
static inline void __tx_matmul_subfloat_e4m3_e4m3_bf16(void* rs1, void* rs2, void* rs3, long& rd) {
    TCFG* tcfg1 = __rvt_env->tcfg[tcfgrs1];
    TCFG* tcfg2 = __rvt_env->tcfg[tcfgrs2];
    TCFG* tcfg3 = __rvt_env->tcfg[tcfgrs3];
    ASSERT(tcfg1->dim1 == tcfg2->dim0);
    ASSERT(tcfg1->dim0 == tcfg3->dim0);
    ASSERT(tcfg2->dim1 == tcfg3->dim1);
    for(int m =0; m < tcfg3->dim0; m+=1) {
        for(int n = 0 ; n < tcfg3->dim1; n+=1) {
            for(int k = 0; k < tcfg1->dim1; k += 1) {
                long offset1 = tcfg1->get_offset(m, k);
                long offset2 = tcfg2->get_offset(k, n);
                long offset3 = tcfg3->get_offset(m, n);
                char rs1_val = *(__rvt_env->get_ptr<char>(rs1) + offset1);
                char rs2_val = *(__rvt_env->get_ptr<char>(rs2) + offset2);
                long rs3_val = *(__rvt_env->get_ptr<long>(rs3) + offset3);
                long result = rs3_val + rs1_val * rs2_val;
                *(__rvt_env->get_ptr<long>(rs3) + offset3) = result;
            }
        }
    }
}
template <long tcfgrs1, long tcfgrs2, long tcfgrs3>
static inline void __tx_matmul_subfloat_e4m3_e4m3_bf16(void* rs1, void* rs2, void* rs3) {
    int rd = 0;
    __tx_matmul_subfloat_e4m3_e4m3_bf16<0, 0, 0>(rs1, rs2, rs3, rd);
}

static inline void* __emu_get_dram_ptr(long in) {
    return (void*)in;
}
static inline void* __emu_get_nram_ptr(long in) {
    return (void*)(__rvt_env->get_ptr<char>((void*)in));
}

template <typename T>
static inline void __emu_raw_copy(long src, long dst, TCFG* tcfgsrc, TCFG* tcfgdst, void* get_src_ptr(long), void* get_dst_ptr(long)) {
    ASSERT(tcfgsrc->dim0 * tcfgsrc->dim1 * tcfgsrc->dim2 * tcfgsrc->dim3 <= NRAM_SIZE);
    ASSERT(tcfgsrc->dim0 * tcfgsrc->dim1 * tcfgsrc->dim2 * tcfgsrc->dim3 == tcfgdst->dim0 * tcfgdst->dim1 * tcfgdst->dim2 * tcfgdst->dim3);
    while(1) {
        long src0 = 0, src1 = 0, src2 = 0, src3 = 0;
        long dst0 = 0, dst1 = 0, dst2 = 0, dst3 = 0;
        // One Element One Time
        (T*)get_dst_ptr(src + sizeof(T) * tcfgdst->get_offset(dst0, dst1, dst2, dst3)) = (T*)get_src_ptr(src + sizeof(T) * tcfgsrc->get_offset(src0, src1, src2, src3));
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
    TCFG* tcfg1 = __rvt_env->tcfg[tcfgrs1];
    TCFG* tcfg3 = __rvt_env->tcfg[tcfgrs3];
    static_assert(log2w >= 0 && log2w <= 3, "log2w must be between 0 and 3");

}
template <long tcfgrs1, long tcfgrs3, long log2w>
static inline void __tx_async_nram_load(void* rs1, void* rs3, long& rd) {
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

#endif

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
        __custom_64b_instruction3r<
        (basetype << 56ul) | (rs1type << 50ul) | (rs2type << 44ul) | (rs3type << 43ul) | (tcfgrs1 << 32ul) | (tcfgrs2 << 35ul) | (tcfgrs3 << 38ul)>
        ((long)rs1, (long)rs2, (long)rs3, rd);
    }
    template <long basetype, long rs1type, long rs2type, long rs3type, long tcfgrs1, long tcfgrs2, long tcfgrs3>
    __attribute__((always_inline))
    static inline void __tx_matmul(void* rs1, void* rs2, void* rs3)
    {
        __custom_64b_instruction3r<
        (basetype << 56ul) | (rs1type << 50ul) | (rs2type << 44ul) | (rs3type << 43ul) | (tcfgrs1 << 32ul) | (tcfgrs2 << 35ul) | (tcfgrs3 << 38ul)>
        ((long)rs1, (long)rs2, (long)rs3);
    }
}

// public

const long TCSR_DIM0 = 0;const long TCSR_STRIDE0 = 1;
const long TCSR_DIM1 = 2;const long TCSR_STRIDE1 = 3;
const long TCSR_DIM2 = 4;const long TCSR_STRIDE2 = 5;
const long TCSR_DIM3 = 6;const long TCSR_STRIDE3 = 7;

#ifndef __EMULATED_RVT_EXT__
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

static inline void __tx_init() {
    // Do Nothing
}
#endif

#endif