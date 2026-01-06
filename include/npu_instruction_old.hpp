#ifndef _NPU_INSTRUCTION_OLD_HPP_
#define _NPU_INSTRUCTION_OLD_HPP_

#include <stdint.h>

typedef unsigned long ulong;

const long TCSR_DIM0 = 0;
const long TCSR_STRIDE0 = 1;
const long TCSR_DIM1 = 2;
const long TCSR_STRIDE1 = 3;
const long TCSR_DIM2 = 4;
const long TCSR_STRIDE2 = 5;
const long TCSR_DIM3 = 6;
const long TCSR_STRIDE3 = 7;

class TCFG {
  public:
    ulong dim0;
    ulong stride0;
    ulong dim1;
    ulong stride1;
    ulong dim2;
    ulong stride2;
    ulong dim3;
    ulong stride3;
};

namespace NPUInstBuilder {
template <long encoding>
__attribute__((always_inline)) static inline void __custom_64b_instruction3r(long rs1, long rs2,
                                                                             long rs3, long &rd) {
    asm volatile(
        "\n"
        ".ifc %0,zero\n.set _R0,0\n.endif\n.ifc %0,ra\n.set _R0,1\n.endif\n.ifc %0,sp\n.set "
        "_R0,2\n.endif\n.ifc %0,gp\n.set _R0,3\n.endif\n.ifc %0,tp\n.set _R0,4\n.endif\n.ifc "
        "%0,t0\n.set _R0,5\n.endif\n.ifc %0,t1\n.set _R0,6\n.endif\n.ifc %0,t2\n.set "
        "_R0,7\n.endif\n.ifc %0,s0\n.set _R0,8\n.endif\n.ifc %0,s1\n.set _R0,9\n.endif\n.ifc "
        "%0,a0\n.set _R0,10\n.endif\n.ifc %0,a1\n.set _R0,11\n.endif\n.ifc %0,a2\n.set "
        "_R0,12\n.endif\n.ifc %0,a3\n.set _R0,13\n.endif\n.ifc %0,a4\n.set _R0,14\n.endif\n.ifc "
        "%0,a5\n.set _R0,15\n.endif\n.ifc %0,a6\n.set _R0,16\n.endif\n.ifc %0,a7\n.set "
        "_R0,17\n.endif\n.ifc %0,s2\n.set _R0,18\n.endif\n.ifc %0,s3\n.set _R0,19\n.endif\n.ifc "
        "%0,s4\n.set _R0,20\n.endif\n.ifc %0,s5\n.set _R0,21\n.endif\n.ifc %0,s6\n.set "
        "_R0,22\n.endif\n.ifc %0,s7\n.set _R0,23\n.endif\n.ifc %0,s8\n.set _R0,24\n.endif\n.ifc "
        "%0,s9\n.set _R0,25\n.endif\n.ifc %0,s10\n.set _R0,26\n.endif\n.ifc %0,s11\n.set "
        "_R0,27\n.endif\n.ifc %0,t3\n.set _R0,28\n.endif\n.ifc %0,t4\n.set _R0,29\n.endif\n.ifc "
        "%0,t5\n.set _R0,30\n.endif\n.ifc %0,t6\n.set _R0,31\n.endif\n.ifc %1,zero\n.set "
        "_R1,0\n.endif\n.ifc %1,ra\n.set _R1,1\n.endif\n.ifc %1,sp\n.set _R1,2\n.endif\n.ifc "
        "%1,gp\n.set _R1,3\n.endif\n.ifc %1,tp\n.set _R1,4\n.endif\n.ifc %1,t0\n.set "
        "_R1,5\n.endif\n.ifc %1,t1\n.set _R1,6\n.endif\n.ifc %1,t2\n.set _R1,7\n.endif\n.ifc "
        "%1,s0\n.set _R1,8\n.endif\n.ifc %1,s1\n.set _R1,9\n.endif\n.ifc %1,a0\n.set "
        "_R1,10\n.endif\n.ifc %1,a1\n.set _R1,11\n.endif\n.ifc %1,a2\n.set _R1,12\n.endif\n.ifc "
        "%1,a3\n.set _R1,13\n.endif\n.ifc %1,a4\n.set _R1,14\n.endif\n.ifc %1,a5\n.set "
        "_R1,15\n.endif\n.ifc %1,a6\n.set _R1,16\n.endif\n.ifc %1,a7\n.set _R1,17\n.endif\n.ifc "
        "%1,s2\n.set _R1,18\n.endif\n.ifc %1,s3\n.set _R1,19\n.endif\n.ifc %1,s4\n.set "
        "_R1,20\n.endif\n.ifc %1,s5\n.set _R1,21\n.endif\n.ifc %1,s6\n.set _R1,22\n.endif\n.ifc "
        "%1,s7\n.set _R1,23\n.endif\n.ifc %1,s8\n.set _R1,24\n.endif\n.ifc %1,s9\n.set "
        "_R1,25\n.endif\n.ifc %1,s10\n.set _R1,26\n.endif\n.ifc %1,s11\n.set _R1,27\n.endif\n.ifc "
        "%1,t3\n.set _R1,28\n.endif\n.ifc %1,t4\n.set _R1,29\n.endif\n.ifc %1,t5\n.set "
        "_R1,30\n.endif\n.ifc %1,t6\n.set _R1,31\n.endif\n.ifc %2,zero\n.set _R2,0\n.endif\n.ifc "
        "%2,ra\n.set _R2,1\n.endif\n.ifc %2,sp\n.set _R2,2\n.endif\n.ifc %2,gp\n.set "
        "_R2,3\n.endif\n.ifc %2,tp\n.set _R2,4\n.endif\n.ifc %2,t0\n.set _R2,5\n.endif\n.ifc "
        "%2,t1\n.set _R2,6\n.endif\n.ifc %2,t2\n.set _R2,7\n.endif\n.ifc %2,s0\n.set "
        "_R2,8\n.endif\n.ifc %2,s1\n.set _R2,9\n.endif\n.ifc %2,a0\n.set _R2,10\n.endif\n.ifc "
        "%2,a1\n.set _R2,11\n.endif\n.ifc %2,a2\n.set _R2,12\n.endif\n.ifc %2,a3\n.set "
        "_R2,13\n.endif\n.ifc %2,a4\n.set _R2,14\n.endif\n.ifc %2,a5\n.set _R2,15\n.endif\n.ifc "
        "%2,a6\n.set _R2,16\n.endif\n.ifc %2,a7\n.set _R2,17\n.endif\n.ifc %2,s2\n.set "
        "_R2,18\n.endif\n.ifc %2,s3\n.set _R2,19\n.endif\n.ifc %2,s4\n.set _R2,20\n.endif\n.ifc "
        "%2,s5\n.set _R2,21\n.endif\n.ifc %2,s6\n.set _R2,22\n.endif\n.ifc %2,s7\n.set "
        "_R2,23\n.endif\n.ifc %2,s8\n.set _R2,24\n.endif\n.ifc %2,s9\n.set _R2,25\n.endif\n.ifc "
        "%2,s10\n.set _R2,26\n.endif\n.ifc %2,s11\n.set _R2,27\n.endif\n.ifc %2,t3\n.set "
        "_R2,28\n.endif\n.ifc %2,t4\n.set _R2,29\n.endif\n.ifc %2,t5\n.set _R2,30\n.endif\n.ifc "
        "%2,t6\n.set _R2,31\n.endif\n.ifc %3,zero\n.set _R3,0\n.endif\n.ifc %3,ra\n.set "
        "_R3,1\n.endif\n.ifc %3,sp\n.set _R3,2\n.endif\n.ifc %3,gp\n.set _R3,3\n.endif\n.ifc "
        "%3,tp\n.set _R3,4\n.endif\n.ifc %3,t0\n.set _R3,5\n.endif\n.ifc %3,t1\n.set "
        "_R3,6\n.endif\n.ifc %3,t2\n.set _R3,7\n.endif\n.ifc %3,s0\n.set _R3,8\n.endif\n.ifc "
        "%3,s1\n.set _R3,9\n.endif\n.ifc %3,a0\n.set _R3,10\n.endif\n.ifc %3,a1\n.set "
        "_R3,11\n.endif\n.ifc %3,a2\n.set _R3,12\n.endif\n.ifc %3,a3\n.set _R3,13\n.endif\n.ifc "
        "%3,a4\n.set _R3,14\n.endif\n.ifc %3,a5\n.set _R3,15\n.endif\n.ifc %3,a6\n.set "
        "_R3,16\n.endif\n.ifc %3,a7\n.set _R3,17\n.endif\n.ifc %3,s2\n.set _R3,18\n.endif\n.ifc "
        "%3,s3\n.set _R3,19\n.endif\n.ifc %3,s4\n.set _R3,20\n.endif\n.ifc %3,s5\n.set "
        "_R3,21\n.endif\n.ifc %3,s6\n.set _R3,22\n.endif\n.ifc %3,s7\n.set _R3,23\n.endif\n.ifc "
        "%3,s8\n.set _R3,24\n.endif\n.ifc %3,s9\n.set _R3,25\n.endif\n.ifc %3,s10\n.set "
        "_R3,26\n.endif\n.ifc %3,s11\n.set _R3,27\n.endif\n.ifc %3,t3\n.set _R3,28\n.endif\n.ifc "
        "%3,t4\n.set _R3,29\n.endif\n.ifc %3,t5\n.set _R3,30\n.endif\n.ifc %3,t6\n.set "
        "_R3,31\n.endif\n"
        ".align 3\n"
        ".insn 8, %4 | 0x3F | (_R0 << 7) | (_R1 << 15) | (_R2 << 20) | (_R3 << 27)\n"
        ".set _R0,z\n.set _R1,z\n.set _R2,z\n.set _R3,z\n"
        : "+r"(rd)
        : "r"(rs1), "r"(rs2), "r"(rs3), "i"(encoding));
}
template <long encoding>
__attribute__((always_inline)) static inline void __custom_64b_instruction3r(long rs1, long rs2,
                                                                             long rs3) {
    asm volatile(
        "\n"
        ".ifc %0,zero\n.set _R0,0\n.endif\n.ifc %0,ra\n.set _R0,1\n.endif\n.ifc %0,sp\n.set "
        "_R0,2\n.endif\n.ifc %0,gp\n.set _R0,3\n.endif\n.ifc %0,tp\n.set _R0,4\n.endif\n.ifc "
        "%0,t0\n.set _R0,5\n.endif\n.ifc %0,t1\n.set _R0,6\n.endif\n.ifc %0,t2\n.set "
        "_R0,7\n.endif\n.ifc %0,s0\n.set _R0,8\n.endif\n.ifc %0,s1\n.set _R0,9\n.endif\n.ifc "
        "%0,a0\n.set _R0,10\n.endif\n.ifc %0,a1\n.set _R0,11\n.endif\n.ifc %0,a2\n.set "
        "_R0,12\n.endif\n.ifc %0,a3\n.set _R0,13\n.endif\n.ifc %0,a4\n.set _R0,14\n.endif\n.ifc "
        "%0,a5\n.set _R0,15\n.endif\n.ifc %0,a6\n.set _R0,16\n.endif\n.ifc %0,a7\n.set "
        "_R0,17\n.endif\n.ifc %0,s2\n.set _R0,18\n.endif\n.ifc %0,s3\n.set _R0,19\n.endif\n.ifc "
        "%0,s4\n.set _R0,20\n.endif\n.ifc %0,s5\n.set _R0,21\n.endif\n.ifc %0,s6\n.set "
        "_R0,22\n.endif\n.ifc %0,s7\n.set _R0,23\n.endif\n.ifc %0,s8\n.set _R0,24\n.endif\n.ifc "
        "%0,s9\n.set _R0,25\n.endif\n.ifc %0,s10\n.set _R0,26\n.endif\n.ifc %0,s11\n.set "
        "_R0,27\n.endif\n.ifc %0,t3\n.set _R0,28\n.endif\n.ifc %0,t4\n.set _R0,29\n.endif\n.ifc "
        "%0,t5\n.set _R0,30\n.endif\n.ifc %0,t6\n.set _R0,31\n.endif\n.ifc %1,zero\n.set "
        "_R1,0\n.endif\n.ifc %1,ra\n.set _R1,1\n.endif\n.ifc %1,sp\n.set _R1,2\n.endif\n.ifc "
        "%1,gp\n.set _R1,3\n.endif\n.ifc %1,tp\n.set _R1,4\n.endif\n.ifc %1,t0\n.set "
        "_R1,5\n.endif\n.ifc %1,t1\n.set _R1,6\n.endif\n.ifc %1,t2\n.set _R1,7\n.endif\n.ifc "
        "%1,s0\n.set _R1,8\n.endif\n.ifc %1,s1\n.set _R1,9\n.endif\n.ifc %1,a0\n.set "
        "_R1,10\n.endif\n.ifc %1,a1\n.set _R1,11\n.endif\n.ifc %1,a2\n.set _R1,12\n.endif\n.ifc "
        "%1,a3\n.set _R1,13\n.endif\n.ifc %1,a4\n.set _R1,14\n.endif\n.ifc %1,a5\n.set "
        "_R1,15\n.endif\n.ifc %1,a6\n.set _R1,16\n.endif\n.ifc %1,a7\n.set _R1,17\n.endif\n.ifc "
        "%1,s2\n.set _R1,18\n.endif\n.ifc %1,s3\n.set _R1,19\n.endif\n.ifc %1,s4\n.set "
        "_R1,20\n.endif\n.ifc %1,s5\n.set _R1,21\n.endif\n.ifc %1,s6\n.set _R1,22\n.endif\n.ifc "
        "%1,s7\n.set _R1,23\n.endif\n.ifc %1,s8\n.set _R1,24\n.endif\n.ifc %1,s9\n.set "
        "_R1,25\n.endif\n.ifc %1,s10\n.set _R1,26\n.endif\n.ifc %1,s11\n.set _R1,27\n.endif\n.ifc "
        "%1,t3\n.set _R1,28\n.endif\n.ifc %1,t4\n.set _R1,29\n.endif\n.ifc %1,t5\n.set "
        "_R1,30\n.endif\n.ifc %1,t6\n.set _R1,31\n.endif\n.ifc %2,zero\n.set _R2,0\n.endif\n.ifc "
        "%2,ra\n.set _R2,1\n.endif\n.ifc %2,sp\n.set _R2,2\n.endif\n.ifc %2,gp\n.set "
        "_R2,3\n.endif\n.ifc %2,tp\n.set _R2,4\n.endif\n.ifc %2,t0\n.set _R2,5\n.endif\n.ifc "
        "%2,t1\n.set _R2,6\n.endif\n.ifc %2,t2\n.set _R2,7\n.endif\n.ifc %2,s0\n.set "
        "_R2,8\n.endif\n.ifc %2,s1\n.set _R2,9\n.endif\n.ifc %2,a0\n.set _R2,10\n.endif\n.ifc "
        "%2,a1\n.set _R2,11\n.endif\n.ifc %2,a2\n.set _R2,12\n.endif\n.ifc %2,a3\n.set "
        "_R2,13\n.endif\n.ifc %2,a4\n.set _R2,14\n.endif\n.ifc %2,a5\n.set _R2,15\n.endif\n.ifc "
        "%2,a6\n.set _R2,16\n.endif\n.ifc %2,a7\n.set _R2,17\n.endif\n.ifc %2,s2\n.set "
        "_R2,18\n.endif\n.ifc %2,s3\n.set _R2,19\n.endif\n.ifc %2,s4\n.set _R2,20\n.endif\n.ifc "
        "%2,s5\n.set _R2,21\n.endif\n.ifc %2,s6\n.set _R2,22\n.endif\n.ifc %2,s7\n.set "
        "_R2,23\n.endif\n.ifc %2,s8\n.set _R2,24\n.endif\n.ifc %2,s9\n.set _R2,25\n.endif\n.ifc "
        "%2,s10\n.set _R2,26\n.endif\n.ifc %2,s11\n.set _R2,27\n.endif\n.ifc %2,t3\n.set "
        "_R2,28\n.endif\n.ifc %2,t4\n.set _R2,29\n.endif\n.ifc %2,t5\n.set _R2,30\n.endif\n.ifc "
        "%2,t6\n.set _R2,31\n.endif\n"
        ".align 3\n"
        ".insn 8, %3 | 0x3F | (_R0 << 15) | (_R1 << 20) | (_R2 << 27)\n"
        ".set _R0,z\n.set _R1,z\n.set _R2,z\n" ::"r"(rs1),
        "r"(rs2), "r"(rs3), "i"(encoding));
}
template <long encoding>
__attribute__((always_inline)) static inline void __custom_64b_instruction2r(long rs1, long rs3,
                                                                             long &rd) {
    asm volatile(
        "\n"
        ".ifc %0,zero\n.set _R0,0\n.endif\n.ifc %0,ra\n.set _R0,1\n.endif\n.ifc %0,sp\n.set "
        "_R0,2\n.endif\n.ifc %0,gp\n.set _R0,3\n.endif\n.ifc %0,tp\n.set _R0,4\n.endif\n.ifc "
        "%0,t0\n.set _R0,5\n.endif\n.ifc %0,t1\n.set _R0,6\n.endif\n.ifc %0,t2\n.set "
        "_R0,7\n.endif\n.ifc %0,s0\n.set _R0,8\n.endif\n.ifc %0,s1\n.set _R0,9\n.endif\n.ifc "
        "%0,a0\n.set _R0,10\n.endif\n.ifc %0,a1\n.set _R0,11\n.endif\n.ifc %0,a2\n.set "
        "_R0,12\n.endif\n.ifc %0,a3\n.set _R0,13\n.endif\n.ifc %0,a4\n.set _R0,14\n.endif\n.ifc "
        "%0,a5\n.set _R0,15\n.endif\n.ifc %0,a6\n.set _R0,16\n.endif\n.ifc %0,a7\n.set "
        "_R0,17\n.endif\n.ifc %0,s2\n.set _R0,18\n.endif\n.ifc %0,s3\n.set _R0,19\n.endif\n.ifc "
        "%0,s4\n.set _R0,20\n.endif\n.ifc %0,s5\n.set _R0,21\n.endif\n.ifc %0,s6\n.set "
        "_R0,22\n.endif\n.ifc %0,s7\n.set _R0,23\n.endif\n.ifc %0,s8\n.set _R0,24\n.endif\n.ifc "
        "%0,s9\n.set _R0,25\n.endif\n.ifc %0,s10\n.set _R0,26\n.endif\n.ifc %0,s11\n.set "
        "_R0,27\n.endif\n.ifc %0,t3\n.set _R0,28\n.endif\n.ifc %0,t4\n.set _R0,29\n.endif\n.ifc "
        "%0,t5\n.set _R0,30\n.endif\n.ifc %0,t6\n.set _R0,31\n.endif\n.ifc %1,zero\n.set "
        "_R1,0\n.endif\n.ifc %1,ra\n.set _R1,1\n.endif\n.ifc %1,sp\n.set _R1,2\n.endif\n.ifc "
        "%1,gp\n.set _R1,3\n.endif\n.ifc %1,tp\n.set _R1,4\n.endif\n.ifc %1,t0\n.set "
        "_R1,5\n.endif\n.ifc %1,t1\n.set _R1,6\n.endif\n.ifc %1,t2\n.set _R1,7\n.endif\n.ifc "
        "%1,s0\n.set _R1,8\n.endif\n.ifc %1,s1\n.set _R1,9\n.endif\n.ifc %1,a0\n.set "
        "_R1,10\n.endif\n.ifc %1,a1\n.set _R1,11\n.endif\n.ifc %1,a2\n.set _R1,12\n.endif\n.ifc "
        "%1,a3\n.set _R1,13\n.endif\n.ifc %1,a4\n.set _R1,14\n.endif\n.ifc %1,a5\n.set "
        "_R1,15\n.endif\n.ifc %1,a6\n.set _R1,16\n.endif\n.ifc %1,a7\n.set _R1,17\n.endif\n.ifc "
        "%1,s2\n.set _R1,18\n.endif\n.ifc %1,s3\n.set _R1,19\n.endif\n.ifc %1,s4\n.set "
        "_R1,20\n.endif\n.ifc %1,s5\n.set _R1,21\n.endif\n.ifc %1,s6\n.set _R1,22\n.endif\n.ifc "
        "%1,s7\n.set _R1,23\n.endif\n.ifc %1,s8\n.set _R1,24\n.endif\n.ifc %1,s9\n.set "
        "_R1,25\n.endif\n.ifc %1,s10\n.set _R1,26\n.endif\n.ifc %1,s11\n.set _R1,27\n.endif\n.ifc "
        "%1,t3\n.set _R1,28\n.endif\n.ifc %1,t4\n.set _R1,29\n.endif\n.ifc %1,t5\n.set "
        "_R1,30\n.endif\n.ifc %1,t6\n.set _R1,31\n.endif\n.ifc %2,zero\n.set _R2,0\n.endif\n.ifc "
        "%2,ra\n.set _R2,1\n.endif\n.ifc %2,sp\n.set _R2,2\n.endif\n.ifc %2,gp\n.set "
        "_R2,3\n.endif\n.ifc %2,tp\n.set _R2,4\n.endif\n.ifc %2,t0\n.set _R2,5\n.endif\n.ifc "
        "%2,t1\n.set _R2,6\n.endif\n.ifc %2,t2\n.set _R2,7\n.endif\n.ifc %2,s0\n.set "
        "_R2,8\n.endif\n.ifc %2,s1\n.set _R2,9\n.endif\n.ifc %2,a0\n.set _R2,10\n.endif\n.ifc "
        "%2,a1\n.set _R2,11\n.endif\n.ifc %2,a2\n.set _R2,12\n.endif\n.ifc %2,a3\n.set "
        "_R2,13\n.endif\n.ifc %2,a4\n.set _R2,14\n.endif\n.ifc %2,a5\n.set _R2,15\n.endif\n.ifc "
        "%2,a6\n.set _R2,16\n.endif\n.ifc %2,a7\n.set _R2,17\n.endif\n.ifc %2,s2\n.set "
        "_R2,18\n.endif\n.ifc %2,s3\n.set _R2,19\n.endif\n.ifc %2,s4\n.set _R2,20\n.endif\n.ifc "
        "%2,s5\n.set _R2,21\n.endif\n.ifc %2,s6\n.set _R2,22\n.endif\n.ifc %2,s7\n.set "
        "_R2,23\n.endif\n.ifc %2,s8\n.set _R2,24\n.endif\n.ifc %2,s9\n.set _R2,25\n.endif\n.ifc "
        "%2,s10\n.set _R2,26\n.endif\n.ifc %2,s11\n.set _R2,27\n.endif\n.ifc %2,t3\n.set "
        "_R2,28\n.endif\n.ifc %2,t4\n.set _R2,29\n.endif\n.ifc %2,t5\n.set _R2,30\n.endif\n.ifc "
        "%2,t6\n.set _R2,31\n.endif\n"
        ".align 3\n"
        ".insn 8, %3 | 0x3F | (_R0 << 7) | (_R1 << 15) | (_R2 << 27)\n"
        ".set _R0,z\n.set _R1,z\n.set _R2,z\n"
        : "+r"(rd)
        : "r"(rs1), "r"(rs3), "i"(encoding));
}
template <long encoding>
__attribute__((always_inline)) static inline void __custom_64b_instruction2r(long rs1, long rs3) {
    asm volatile(
        "\n"
        ".ifc %0,zero\n.set _R0,0\n.endif\n.ifc %0,ra\n.set _R0,1\n.endif\n.ifc %0,sp\n.set "
        "_R0,2\n.endif\n.ifc %0,gp\n.set _R0,3\n.endif\n.ifc %0,tp\n.set _R0,4\n.endif\n.ifc "
        "%0,t0\n.set _R0,5\n.endif\n.ifc %0,t1\n.set _R0,6\n.endif\n.ifc %0,t2\n.set "
        "_R0,7\n.endif\n.ifc %0,s0\n.set _R0,8\n.endif\n.ifc %0,s1\n.set _R0,9\n.endif\n.ifc "
        "%0,a0\n.set _R0,10\n.endif\n.ifc %0,a1\n.set _R0,11\n.endif\n.ifc %0,a2\n.set "
        "_R0,12\n.endif\n.ifc %0,a3\n.set _R0,13\n.endif\n.ifc %0,a4\n.set _R0,14\n.endif\n.ifc "
        "%0,a5\n.set _R0,15\n.endif\n.ifc %0,a6\n.set _R0,16\n.endif\n.ifc %0,a7\n.set "
        "_R0,17\n.endif\n.ifc %0,s2\n.set _R0,18\n.endif\n.ifc %0,s3\n.set _R0,19\n.endif\n.ifc "
        "%0,s4\n.set _R0,20\n.endif\n.ifc %0,s5\n.set _R0,21\n.endif\n.ifc %0,s6\n.set "
        "_R0,22\n.endif\n.ifc %0,s7\n.set _R0,23\n.endif\n.ifc %0,s8\n.set _R0,24\n.endif\n.ifc "
        "%0,s9\n.set _R0,25\n.endif\n.ifc %0,s10\n.set _R0,26\n.endif\n.ifc %0,s11\n.set "
        "_R0,27\n.endif\n.ifc %0,t3\n.set _R0,28\n.endif\n.ifc %0,t4\n.set _R0,29\n.endif\n.ifc "
        "%0,t5\n.set _R0,30\n.endif\n.ifc %0,t6\n.set _R0,31\n.endif\n.ifc %1,zero\n.set "
        "_R1,0\n.endif\n.ifc %1,ra\n.set _R1,1\n.endif\n.ifc %1,sp\n.set _R1,2\n.endif\n.ifc "
        "%1,gp\n.set _R1,3\n.endif\n.ifc %1,tp\n.set _R1,4\n.endif\n.ifc %1,t0\n.set "
        "_R1,5\n.endif\n.ifc %1,t1\n.set _R1,6\n.endif\n.ifc %1,t2\n.set _R1,7\n.endif\n.ifc "
        "%1,s0\n.set _R1,8\n.endif\n.ifc %1,s1\n.set _R1,9\n.endif\n.ifc %1,a0\n.set "
        "_R1,10\n.endif\n.ifc %1,a1\n.set _R1,11\n.endif\n.ifc %1,a2\n.set _R1,12\n.endif\n.ifc "
        "%1,a3\n.set _R1,13\n.endif\n.ifc %1,a4\n.set _R1,14\n.endif\n.ifc %1,a5\n.set "
        "_R1,15\n.endif\n.ifc %1,a6\n.set _R1,16\n.endif\n.ifc %1,a7\n.set _R1,17\n.endif\n.ifc "
        "%1,s2\n.set _R1,18\n.endif\n.ifc %1,s3\n.set _R1,19\n.endif\n.ifc %1,s4\n.set "
        "_R1,20\n.endif\n.ifc %1,s5\n.set _R1,21\n.endif\n.ifc %1,s6\n.set _R1,22\n.endif\n.ifc "
        "%1,s7\n.set _R1,23\n.endif\n.ifc %1,s8\n.set _R1,24\n.endif\n.ifc %1,s9\n.set "
        "_R1,25\n.endif\n.ifc %1,s10\n.set _R1,26\n.endif\n.ifc %1,s11\n.set _R1,27\n.endif\n.ifc "
        "%1,t3\n.set _R1,28\n.endif\n.ifc %1,t4\n.set _R1,29\n.endif\n.ifc %1,t5\n.set "
        "_R1,30\n.endif\n.ifc %1,t6\n.set _R1,31\n.endif\n"
        ".align 3\n"
        ".insn 8, %2 | 0x3F | (_R0 << 15) | (_R1 << 27)\n"
        ".set _R0,z\n.set _R1,z\n" ::"r"(rs1),
        "r"(rs3), "i"(encoding));
}
template <long encoding>
__attribute__((always_inline)) static inline void __custom_64b_instruction1r(long rs3, long &rd) {
    asm volatile(
        "\n"
        ".ifc %0,zero\n.set _R0,0\n.endif\n.ifc %0,ra\n.set _R0,1\n.endif\n.ifc %0,sp\n.set "
        "_R0,2\n.endif\n.ifc %0,gp\n.set _R0,3\n.endif\n.ifc %0,tp\n.set _R0,4\n.endif\n.ifc "
        "%0,t0\n.set _R0,5\n.endif\n.ifc %0,t1\n.set _R0,6\n.endif\n.ifc %0,t2\n.set "
        "_R0,7\n.endif\n.ifc %0,s0\n.set _R0,8\n.endif\n.ifc %0,s1\n.set _R0,9\n.endif\n.ifc "
        "%0,a0\n.set _R0,10\n.endif\n.ifc %0,a1\n.set _R0,11\n.endif\n.ifc %0,a2\n.set "
        "_R0,12\n.endif\n.ifc %0,a3\n.set _R0,13\n.endif\n.ifc %0,a4\n.set _R0,14\n.endif\n.ifc "
        "%0,a5\n.set _R0,15\n.endif\n.ifc %0,a6\n.set _R0,16\n.endif\n.ifc %0,a7\n.set "
        "_R0,17\n.endif\n.ifc %0,s2\n.set _R0,18\n.endif\n.ifc %0,s3\n.set _R0,19\n.endif\n.ifc "
        "%0,s4\n.set _R0,20\n.endif\n.ifc %0,s5\n.set _R0,21\n.endif\n.ifc %0,s6\n.set "
        "_R0,22\n.endif\n.ifc %0,s7\n.set _R0,23\n.endif\n.ifc %0,s8\n.set _R0,24\n.endif\n.ifc "
        "%0,s9\n.set _R0,25\n.endif\n.ifc %0,s10\n.set _R0,26\n.endif\n.ifc %0,s11\n.set "
        "_R0,27\n.endif\n.ifc %0,t3\n.set _R0,28\n.endif\n.ifc %0,t4\n.set _R0,29\n.endif\n.ifc "
        "%0,t5\n.set _R0,30\n.endif\n.ifc %0,t6\n.set _R0,31\n.endif\n.ifc %1,zero\n.set "
        "_R1,0\n.endif\n.ifc %1,ra\n.set _R1,1\n.endif\n.ifc %1,sp\n.set _R1,2\n.endif\n.ifc "
        "%1,gp\n.set _R1,3\n.endif\n.ifc %1,tp\n.set _R1,4\n.endif\n.ifc %1,t0\n.set "
        "_R1,5\n.endif\n.ifc %1,t1\n.set _R1,6\n.endif\n.ifc %1,t2\n.set _R1,7\n.endif\n.ifc "
        "%1,s0\n.set _R1,8\n.endif\n.ifc %1,s1\n.set _R1,9\n.endif\n.ifc %1,a0\n.set "
        "_R1,10\n.endif\n.ifc %1,a1\n.set _R1,11\n.endif\n.ifc %1,a2\n.set _R1,12\n.endif\n.ifc "
        "%1,a3\n.set _R1,13\n.endif\n.ifc %1,a4\n.set _R1,14\n.endif\n.ifc %1,a5\n.set "
        "_R1,15\n.endif\n.ifc %1,a6\n.set _R1,16\n.endif\n.ifc %1,a7\n.set _R1,17\n.endif\n.ifc "
        "%1,s2\n.set _R1,18\n.endif\n.ifc %1,s3\n.set _R1,19\n.endif\n.ifc %1,s4\n.set "
        "_R1,20\n.endif\n.ifc %1,s5\n.set _R1,21\n.endif\n.ifc %1,s6\n.set _R1,22\n.endif\n.ifc "
        "%1,s7\n.set _R1,23\n.endif\n.ifc %1,s8\n.set _R1,24\n.endif\n.ifc %1,s9\n.set "
        "_R1,25\n.endif\n.ifc %1,s10\n.set _R1,26\n.endif\n.ifc %1,s11\n.set _R1,27\n.endif\n.ifc "
        "%1,t3\n.set _R1,28\n.endif\n.ifc %1,t4\n.set _R1,29\n.endif\n.ifc %1,t5\n.set "
        "_R1,30\n.endif\n.ifc %1,t6\n.set _R1,31\n.endif\n"
        ".align 3\n"
        ".insn 8, %2 | 0x3F | (_R0 << 7) | (_R1 << 27)\n"
        ".set _R0,z\n.set _R1,z\n"
        : "+r"(rd)
        : "r"(rs3), "i"(encoding));
}
template <long encoding>
__attribute__((always_inline)) static inline void __custom_64b_instruction_bar() {
    asm volatile(
        ".align 3\n"
        ".insn 8, %0 | 0x3F | (5 << 7) | (5 << 27)\n"
                 "andi t0,t0,0\n" ::"i"(encoding));
}
template <long encoding>
__attribute__((always_inline)) static inline void __custom_64b_instruction1r(long rs3) {
    asm volatile(
        "\n"
        ".ifc %0,zero\n.set _R0,0\n.endif\n.ifc %0,ra\n.set _R0,1\n.endif\n.ifc %0,sp\n.set "
        "_R0,2\n.endif\n.ifc %0,gp\n.set _R0,3\n.endif\n.ifc %0,tp\n.set _R0,4\n.endif\n.ifc "
        "%0,t0\n.set _R0,5\n.endif\n.ifc %0,t1\n.set _R0,6\n.endif\n.ifc %0,t2\n.set "
        "_R0,7\n.endif\n.ifc %0,s0\n.set _R0,8\n.endif\n.ifc %0,s1\n.set _R0,9\n.endif\n.ifc "
        "%0,a0\n.set _R0,10\n.endif\n.ifc %0,a1\n.set _R0,11\n.endif\n.ifc %0,a2\n.set "
        "_R0,12\n.endif\n.ifc %0,a3\n.set _R0,13\n.endif\n.ifc %0,a4\n.set _R0,14\n.endif\n.ifc "
        "%0,a5\n.set _R0,15\n.endif\n.ifc %0,a6\n.set _R0,16\n.endif\n.ifc %0,a7\n.set "
        "_R0,17\n.endif\n.ifc %0,s2\n.set _R0,18\n.endif\n.ifc %0,s3\n.set _R0,19\n.endif\n.ifc "
        "%0,s4\n.set _R0,20\n.endif\n.ifc %0,s5\n.set _R0,21\n.endif\n.ifc %0,s6\n.set "
        "_R0,22\n.endif\n.ifc %0,s7\n.set _R0,23\n.endif\n.ifc %0,s8\n.set _R0,24\n.endif\n.ifc "
        "%0,s9\n.set _R0,25\n.endif\n.ifc %0,s10\n.set _R0,26\n.endif\n.ifc %0,s11\n.set "
        "_R0,27\n.endif\n.ifc %0,t3\n.set _R0,28\n.endif\n.ifc %0,t4\n.set _R0,29\n.endif\n.ifc "
        "%0,t5\n.set _R0,30\n.endif\n.ifc %0,t6\n.set _R0,31\n.endif\n"
        ".align 3\n"
        ".insn 8, %1 | 0x3F | (_R0 << 27)\n"
        ".set _R0,z\n" ::"r"(rs3),
        "i"(encoding));
}

template <long basetype, long rs1type, long rs2type, long rs3type, long tcfgrs1, long tcfgrs2,
          long tcfgrs3>
__attribute__((always_inline)) static inline void __tx_matmul(void *rs1, void *rs2, void *rs3,
                                                              long &rd) {
    NPUInstBuilder::__custom_64b_instruction3r<
        (basetype << 56ul) | (rs1type << 50ul) | (rs2type << 44ul) | (rs3type << 43ul) |
        (tcfgrs1 << 32ul) | (tcfgrs2 << 35ul) | (tcfgrs3 << 38ul)>((long)rs1, (long)rs2, (long)rs3,
                                                                   rd);
}
template <long basetype, long rs1type, long rs2type, long rs3type, long tcfgrs1, long tcfgrs2,
          long tcfgrs3>
__attribute__((always_inline)) static inline void __tx_matmul(void *rs1, void *rs2, void *rs3) {
    NPUInstBuilder::__custom_64b_instruction3r<
        (basetype << 56ul) | (rs1type << 50ul) | (rs2type << 44ul) | (rs3type << 43ul) |
        (tcfgrs1 << 32ul) | (tcfgrs2 << 35ul) | (tcfgrs3 << 38ul)>((long)rs1, (long)rs2, (long)rs3);
}

enum __TX_ELEM_DTYPE_ENUM {
    TX_SUBBYTE_FLOAT = 0,
    TX_U8 = 0b1000000,
    TX_S8 = 0b1000001,
    TX_U16 = 0b1000010,
    TX_S16 = 0b1000011,
    TX_U32 = 0b1000100,
    TX_S32 = 0b1000101,
    TX_U64 = 0b1000110,
    TX_S64 = 0b1000111,
    TX_FP16 = 0b1001010,
    TX_BF16 = 0b1001011,
    TX_FP32 = 0b1001100,
    TX_TF32 = 0b1001101,
    TX_FP64 = 0b1001110,
    TX_E4M3 = 0b0011100
};

enum __TX_INT_OP_ENUM {
    TX_INT_ADD = 0,
    TX_INT_SUB = 1,
    TX_INT_MUL = 2,
    TX_INT_SRL = 3,
    TX_INT_SRA = 4,
    TX_INT_SLL = 5,
    TX_INT_LUTOP = 16, // 16 | LUT
};
template <long op, long type, long tcfgrs1, long tcfgrs2, long tcfgrs3, long broadcast>
__attribute__((always_inline)) static inline void __tx_int_op_type(void *rs1, void *rs2,
                                                                   void *rs3) {
    NPUInstBuilder::__custom_64b_instruction3r<(0b00001000ul << 56ul) | // func == 0000 1000
                                               (op << 50ul) |           // op
                                               (broadcast << 45ul) |    // broadcast
                                               (((type >> 3) & 0b1111) << 41ul) | // Type-HI
                                               (((type >> 0) & 0b0111) << 12ul) | // Type-LO
                                               (tcfgrs3 << 38ul) | (tcfgrs2 << 35ul) |
                                               (tcfgrs1 << 32ul)>((long)rs1, (long)rs2, (long)rs3);
}

enum __TX_FP_OP_ENUM { TX_FP_ADD = 0, TX_FP_SUB = 1, TX_FP_MUL = 2 };

enum __TX_FP_REDUCE_ENUM { TX_FP_MAX = 0, TX_FP_MIN = 1, TX_FP_SUM = 2 };

enum __TX_INT_REDUCE_ENUM { TX_INT_MAX = 0, TX_INT_MIN = 1, TX_INT_SUM = 2 };

template <long op, long type, long tcfgrs1, long tcfgrs2, long tcfgrs3, long broadcast>
__attribute__((always_inline)) static inline void __tx_fp_op_type(void *rs1, void *rs2, void *rs3) {
    NPUInstBuilder::__custom_64b_instruction3r<(0b00001001ul << 56ul) | // func == 0000 1001
                                               (op << 50ul) |           // op
                                               (broadcast << 45ul) |    // broadcast
                                               (((type >> 3) & 0b1111) << 41ul) | // Type-HI
                                               (((type >> 0) & 0b0111) << 12ul) | // Type-LO
                                               (tcfgrs3 << 38ul) | (tcfgrs2 << 35ul) |
                                               (tcfgrs1 << 32ul)>((long)rs1, (long)rs2, (long)rs3);
}

template <long type, long tcfgrs1, long width, long tcfgrs3, long log2Length>
__attribute__((always_inline)) static inline void __tx_lut_int_type(void *rs1, void *rs2,
                                                                    void *rs3) {
    NPUInstBuilder::__custom_64b_instruction3r<(0b00001010ul << 56ul) | // func == 0000 1010
                                               (log2Length << 50ul) |   // log2Length
                                               (((type >> 3) & 0b1111) << 41ul) | // Type-HI
                                               (((type >> 0) & 0b0111) << 12ul) | // Type-LO
                                               (tcfgrs3 << 38ul) | (width << 35ul) |
                                               (tcfgrs1 << 32ul)>((long)rs1, (long)rs2, (long)rs3);
}

template <long mode, long type, long log2Length, long tcfgrs1, long width, long tcfgrs3>
__attribute__((always_inline)) static inline void __tx_lut_fp_mode_type(long rs1, long rs2,
                                                                        long rs3) {
    NPUInstBuilder::__custom_64b_instruction3r<(0b00001011ul << 56ul) | // func == 0000 1011
                                               (log2Length << 50ul) |   // log2Length
                                               (((type >> 3) & 0b1111) << 41ul) | // Type-HI
                                               (((type >> 0) & 0b0111) << 12ul) | // Type-LO
                                               (tcfgrs3 << 38ul) | (width << 35ul) |
                                               (tcfgrs1 << 32ul) | (mode << 25ul)>(
        (long)rs1, (long)rs2, (long)rs3);
}

template <long op, long type, long tcfgrs1, long tcfgrs3, long dim>
__attribute__((always_inline)) static inline void __tx_int_reduce_op_type(void *rs1, void *rs3) {
    constexpr long encoding = (0b10001100ul << 56ul) |           // func == 1000 1100
                              (op << 50ul) |                     // op
                              (((type >> 3) & 0b1111) << 41ul) | // Type-HI
                              (((type >> 0) & 0b0111) << 12ul) | // Type-LO
                              (tcfgrs3 << 38ul) | (tcfgrs1 << 32ul) | (dim << 25ul);
    NPUInstBuilder::__custom_64b_instruction2r<encoding>((long)rs1, (long)rs3);
}

template <long op, long type, long tcfgrs1, long tcfgrs3, long dim>
__attribute__((always_inline)) static inline void __tx_fp_reduce_op_type(void *rs1, void *rs3) {
    constexpr long encoding = (0b10001101ul << 56ul) |           // func == 1000 1101
                              (op << 50ul) |                     // op
                              (((type >> 3) & 0b1111) << 41ul) | // Type-HI
                              (((type >> 0) & 0b0111) << 12ul) | // Type-LO
                              (tcfgrs3 << 38ul) | (tcfgrs1 << 32ul) | (dim << 25ul);
    NPUInstBuilder::__custom_64b_instruction2r<encoding>((long)rs1, (long)rs3);
}

enum __TX_CONVERT_MODE { TX_CONVERT_ROUND = 0, TX_CONVERT_FLOOR = 1, TX_CONVERT_CEIL = 2 };
template <long rs1type, long rs3type, long mode, long tcfgrs1, long tcfgrs3>
__attribute__((always_inline)) static inline void __tx_convert_rs1type_rs3type_mode(void *rs1,
                                                                                    void *rs3) {
    constexpr long encoding = (0b10001110ul << 56ul) | // func == 1000 1110
                              (rs3type << 45ul) | (((rs1type >> 3) & 0b1111) << 41ul) | // Type-HI
                              (((rs1type >> 0) & 0b0111) << 12ul) |                     // Type-LO
                              (tcfgrs3 << 38ul) | (tcfgrs1 << 32ul) | (mode << 25ul);
    NPUInstBuilder::__custom_64b_instruction2r<encoding>((long)rs1, (long)rs3);
}

enum __TX_SFU_OPCODE {
    TX_SFU_RECIP = 0,
    TX_SFU_RSQRT = 1,
    TX_SFU_SIN = 2,
    TX_SFU_COS = 3,
    TX_SFU_EXP = 4,
    TX_SFU_RELU = 5,
};
template <long op, long type, long tcfgrs1, long tcfgrs3>
__attribute__((always_inline)) static inline void __tx_sfu_op_type(void *rs1, void *rs3) {

    constexpr long encoding = (0b10010000ul << 56ul) |           // func == 1001 0000
                              (op << 50ul) |                     // op
                              (((type >> 3) & 0b1111) << 41ul) | // Type-HI
                              (((type >> 0) & 0b0111) << 12ul) | // Type-LO
                              (tcfgrs3 << 38ul) | (tcfgrs1 << 32ul);
    NPUInstBuilder::__custom_64b_instruction2r<encoding>((long)rs1, (long)rs3);
}
} // namespace NPUInstBuilder

template <long tcfgrs1, long tcfgrs2, long tcfgrs3>
__attribute__((always_inline)) static inline void
__tx_matmul_subfloat_e4m3_e4m3_bf16(void *rs1, void *rs2, void *rs3, long &rd) {
    NPUInstBuilder::__tx_matmul<0x0, 0b011100, 0b011100, 0b1, tcfgrs1, tcfgrs2, tcfgrs3>(rs1, rs2,
                                                                                         rs3, rd);
}
template <long tcfgrs1, long tcfgrs2, long tcfgrs3>
__attribute__((always_inline)) static inline void
__tx_matmul_subfloat_e4m3_e4m3_bf16(void *rs1, void *rs2, void *rs3) {
    NPUInstBuilder::__tx_matmul<0x0, 0b011100, 0b011100, 0b1, tcfgrs1, tcfgrs2, tcfgrs3>(rs1, rs2,
                                                                                         rs3);
}

template <long tcfgrs1, long tcfgrs2, long tcfgrs3>
__attribute__((always_inline)) static inline void
__tx_matmul_subfloat_e4m3_e4m3_32(void *rs1, void *rs2, void *rs3, long &rd) {
    NPUInstBuilder::__tx_matmul<0x0, 0b011100, 0b011100, 0b0, tcfgrs1, tcfgrs2, tcfgrs3>(rs1, rs2,
                                                                                         rs3, rd);
}
template <long tcfgrs1, long tcfgrs2, long tcfgrs3>
__attribute__((always_inline)) static inline void
__tx_matmul_subfloat_e4m3_e4m3_f32(void *rs1, void *rs2, void *rs3) {
    NPUInstBuilder::__tx_matmul<0x0, 0b011100, 0b011100, 0b0, tcfgrs1, tcfgrs2, tcfgrs3>(rs1, rs2,
                                                                                         rs3);
}
template <long tcfgrs1, long tcfgrs2, long tcfgrs3>
__attribute__((always_inline)) static inline void
__tx_matmul_float_f32_f32_f32(void *rs1, void *rs2, void *rs3) {
    NPUInstBuilder::__tx_matmul<0x1, 0b10, 0b10, 0b0, tcfgrs1, tcfgrs2, tcfgrs3>(rs1, rs2, rs3);
}
template <long tcfgrs1, long tcfgrs2, long tcfgrs3>
__attribute__((always_inline)) static inline void
__tx_matmul_float_fp16_fp16_f32(void *rs1, void *rs2, void *rs3) {
    NPUInstBuilder::__tx_matmul<0x1, 0b00, 0b00, 0b0, tcfgrs1, tcfgrs2, tcfgrs3>(rs1, rs2, rs3);
}
template <long tcfgrs1, long tcfgrs2, long tcfgrs3>
__attribute__((always_inline)) static inline void
__tx_matmul_float_bf16_bf16_f32(void *rs1, void *rs2, void *rs3) {
    NPUInstBuilder::__tx_matmul<0x1, 0b01, 0b01, 0b0, tcfgrs1, tcfgrs2, tcfgrs3>(rs1, rs2, rs3);
}
template <long tcfgrs1, long tcfgrs2, long tcfgrs3>
__attribute__((always_inline)) static inline void __tx_matmul_fix_s8_s8_s32(void *rs1, void *rs2,
                                                                            void *rs3, long &rd) {
    NPUInstBuilder::__tx_matmul<0b00000010, 7, 7, 0b0, tcfgrs1, tcfgrs2, tcfgrs3>(rs1, rs2, rs3,
                                                                                  rd);
}
template <long tcfgrs1, long tcfgrs2, long tcfgrs3>
__attribute__((always_inline)) static inline void __tx_matmul_fix_s8_s8_s32(void *rs1, void *rs2,
                                                                            void *rs3) {
    NPUInstBuilder::__tx_matmul<0b00000010, 7, 7, 0b0, tcfgrs1, tcfgrs2, tcfgrs3>(rs1, rs2, rs3);
}

template <long tcfgrs1, long tcfgrs2, long tcfgrs3>
__attribute__((always_inline)) static inline void __tx_conv_float_f32_f32_f32(void *rs1, void *rs2,
                                                                              void *rs3) {
    NPUInstBuilder::__tx_matmul<0b101, 0b10, 0b10, 0b0, tcfgrs1, tcfgrs2, tcfgrs3>(rs1, rs2, rs3);
}

template <long tcfgrs1, long tcfgrs2, long tcfgrs3>
__attribute__((always_inline)) static inline void
__tx_conv_float_fp16_fp16_f32(void *rs1, void *rs2, void *rs3) {
    NPUInstBuilder::__tx_matmul<0b101, 0b00, 0b00, 0b0, tcfgrs1, tcfgrs2, tcfgrs3>(rs1, rs2, rs3);
}

template <long tcfgrs1, long tcfgrs2, long tcfgrs3>
__attribute__((always_inline)) static inline void
__tx_conv_float_fp16_fp16_fp16(void *rs1, void *rs2, void *rs3) {
    NPUInstBuilder::__tx_matmul<0b101, 0b00, 0b00, 0b1, tcfgrs1, tcfgrs2, tcfgrs3>(rs1, rs2, rs3);
}

template <long tcfgrs1, long tcfgrs3>
__attribute__((always_inline)) static inline void __tx_maxpool_fp16_fp16(void *rs1, void *rs3,
                                                                         long &rd) {
    NPUInstBuilder::__custom_64b_instruction2r<((long)0b10000010 << 56ul) | ((long)0b00 << 50ul) |
                                               ((long)0b00 << 44ul) | ((long)0b1 << 43ul) |
                                               (tcfgrs3 << 38ul) | (tcfgrs1 << 32ul)>(
        (long)rs1, (long)rs3, rd);
}
template <long tcfgrs1, long tcfgrs3>
__attribute__((always_inline)) static inline void __tx_maxpool_fp16_fp16(void *rs1, void *rs3) {
    NPUInstBuilder::__custom_64b_instruction2r<((long)0b10000010 << 56ul) | ((long)0b00 << 50ul) |
                                               ((long)0b00 << 44ul) | ((long)0b1 << 43ul) |
                                               (tcfgrs3 << 38ul) | (tcfgrs1 << 32ul)>((long)rs1,
                                                                                      (long)rs3);
}
template <long rs1type, long rs3type, long tcfgrs1, long tcfgrs3>
__attribute__((always_inline)) static inline void __tx_avgpool_fp16_fp16(void *rs1, void *rs3,
                                                                         long &rd) {
    NPUInstBuilder::__custom_64b_instruction2r<((long)0b10000010 << 56ul) | ((long)0b00 << 50ul) |
                                               ((long)0b01 << 44ul) | ((long)0b1 << 43ul) |
                                               (tcfgrs3 << 38ul) | (tcfgrs1 << 32ul)>(
        (long)rs1, (long)rs3, rd);
}
template <long tcfgrs1, long tcfgrs3>
__attribute__((always_inline)) static inline void __tx_avgpool_fp16_fp16(void *rs1, void *rs3) {
    NPUInstBuilder::__custom_64b_instruction2r<((long)0b10000010 << 56ul) | ((long)0b00 << 50ul) |
                                               ((long)0b01 << 44ul) | ((long)0b1 << 43ul) |
                                               (tcfgrs3 << 38ul) | (tcfgrs1 << 32ul)>((long)rs1,
                                                                                      (long)rs3);
}

template <long tcfgrs1, long tcfgrs3, long log2w>
__attribute__((always_inline)) static inline void __tx_async_uram_copy(void *rs1, void *rs3,
                                                                       long &rd) {
    NPUInstBuilder::__custom_64b_instruction2r<((long)0b10000100 << 56ul) | (tcfgrs3 << 38ul) |
                                               (tcfgrs1 << 32ul) | (log2w << 12)>((long)rs1,
                                                                                  (long)rs3, rd);
}

template <long tcfgrs1, long tcfgrs3, long log2w>
__attribute__((always_inline)) static inline void __tx_async_uram_copy(void *rs1, void *rs3) {
    NPUInstBuilder::__custom_64b_instruction2r<((long)0b10000100 << 56ul) | (tcfgrs3 << 38ul) |
                                               (tcfgrs1 << 32ul) | (log2w << 12)>((long)rs1,
                                                                                  (long)rs3);
}

// template <long tcfgrs1, long tcfgrs3, long log2w>
// __attribute__((always_inline))
// static inline void __tx_async_uram_layout_weight(void* rs1, void* rs3, long& rd) {
//     NPUInstBuilder::__custom_64b_instruction2r<
//         ((long)0b100001001 << 55ul) | (tcfgrs3 << 38ul) | (tcfgrs1 << 32ul) | (log2w <<
//         12)>((long)rs1, (long)rs3, rd);
// }

// template <long tcfgrs1, long tcfgrs3, long log2w>
// __attribute__((always_inline))
// static inline void __tx_async_uram_layout_weight(void* rs1, void* rs3) {
//     NPUInstBuilder::__custom_64b_instruction2r<
//         ((long)0b100001001 << 55ul) | (tcfgrs3 << 38ul) | (tcfgrs1 << 32ul) | (log2w <<
//         12)>((long)rs1, (long)rs3);
// }

template <long tcfgrs1, long tcfgrs3, long log2w>
__attribute__((always_inline)) static inline void __tx_async_uram_load(void *rs1, void *rs3,
                                                                       long &rd) {
    NPUInstBuilder::__custom_64b_instruction2r<((long)0b10000110 << 56ul) | (tcfgrs3 << 38ul) |
                                               (tcfgrs1 << 32ul) | (log2w << 12)>((long)rs1,
                                                                                  (long)rs3, rd);
}

template <long tcfgrs1, long tcfgrs3, long log2w>
__attribute__((always_inline)) static inline void __tx_async_uram_load(void *rs1, void *rs3) {
    NPUInstBuilder::__custom_64b_instruction2r<((long)0b10000110 << 56ul) | (tcfgrs3 << 38ul) |
                                               (tcfgrs1 << 32ul) | (log2w << 12)>((long)rs1,
                                                                                  (long)rs3);
}

template <long tcfgrs1, long tcfgrs3, long log2w>
__attribute__((always_inline)) static inline void __tx_async_uram_store(void *rs1, void *rs3,
                                                                        long &rd) {
    NPUInstBuilder::__custom_64b_instruction2r<((long)0b10000101 << 56ul) | (tcfgrs3 << 38ul) |
                                               (tcfgrs1 << 32ul) | (log2w << 12)>((long)rs1,
                                                                                  (long)rs3, rd);
}
template <long tcfgrs1, long tcfgrs3, long log2w>
__attribute__((always_inline)) static inline void __tx_async_uram_store(void *rs1, void *rs3) {
    NPUInstBuilder::__custom_64b_instruction2r<((long)0b10000101 << 56ul) | (tcfgrs3 << 38ul) |
                                               (tcfgrs1 << 32ul) | (log2w << 12)>((long)rs1,
                                                                                  (long)rs3);
}

template <long tcsrid> __attribute__((always_inline)) static inline void __tx_tcsr_write(long rs1) {
    NPUInstBuilder::__custom_64b_instruction2r<((long)0b10000000 << 56ul) | (tcsrid << 32)>(rs1, 0);
}
template <long tcsrid>
__attribute__((always_inline)) static inline void __tx_tcsr_xchg(long rs1, long &rd) {
    NPUInstBuilder::__custom_64b_instruction2r<((long)0b10000000 << 56ul) | (tcsrid << 32)>(rs1, 0,
                                                                                            rd);
}
template <long tcsrid>
__attribute__((always_inline)) static inline void __tx_tcsr_xchg_mask(long rs1, long rs3,
                                                                      long &rd) {
    NPUInstBuilder::__custom_64b_instruction2r<((long)0b10000001 << 56ul) | (tcsrid << 32)>(
        rs1, rs3, rd);
}

__attribute__((always_inline)) static inline void __tx_fence() {
    asm volatile("FENCE iorw, iorw" ::: "memory");
}

#define GENERATE_TX_INT_OP_TYPE(opname, op, typename, type)                                        \
    template <long tcfgrs1, long tcfgrs2, long tcfgrs3, long broadcast>                            \
    __attribute__((always_inline)) static inline void __tx_int_##opname##_##                       \
        typename(void *rs1, void *rs2, void *rs3) {                                                \
        NPUInstBuilder::__tx_int_op_type<NPUInstBuilder::op, NPUInstBuilder::type, tcfgrs1,        \
                                         tcfgrs2, tcfgrs3, broadcast>(rs1, rs2, rs3);              \
    }

#define GENERATE_TX_FP_OP_TYPE(opname, op, typename, type)                                         \
    template <long tcfgrs1, long tcfgrs2, long tcfgrs3, long broadcast>                            \
    __attribute__((always_inline)) static inline void __tx_fp_##opname##_##                        \
        typename(void *rs1, void *rs2, void *rs3) {                                                \
        NPUInstBuilder::__tx_fp_op_type<NPUInstBuilder::op, NPUInstBuilder::type, tcfgrs1,         \
                                        tcfgrs2, tcfgrs3, broadcast>(rs1, rs2, rs3);               \
    }

GENERATE_TX_INT_OP_TYPE(add, TX_INT_ADD, s32, TX_S32)
GENERATE_TX_INT_OP_TYPE(sub, TX_INT_SUB, s32, TX_S32)
GENERATE_TX_INT_OP_TYPE(mul, TX_INT_MUL, s32, TX_S32)
GENERATE_TX_INT_OP_TYPE(srl, TX_INT_SRL, s32, TX_S32)
GENERATE_TX_INT_OP_TYPE(sra, TX_INT_SRA, s32, TX_S32)
GENERATE_TX_INT_OP_TYPE(sll, TX_INT_SLL, s32, TX_S32)

template <long lutop, long tcfgrs1, long tcfgrs2, long tcfgrs3, long broadcast>
__attribute__((always_inline)) static inline void __tx_int_lutop_s32(void *rs1, void *rs2,
                                                                     void *rs3) {
    NPUInstBuilder::__tx_int_op_type<NPUInstBuilder::TX_INT_LUTOP | lutop, NPUInstBuilder::TX_S32,
                                     tcfgrs1, tcfgrs2, tcfgrs3, broadcast>(rs1, rs2, rs3);
}

GENERATE_TX_FP_OP_TYPE(add, TX_FP_ADD, fp32, TX_FP32)
GENERATE_TX_FP_OP_TYPE(sub, TX_FP_SUB, fp32, TX_FP32)
GENERATE_TX_FP_OP_TYPE(mul, TX_FP_MUL, fp32, TX_FP32)

#undef GENERATE_TX_INT_OP_TYPE
#undef GENERATE_TX_FP_OP_TYPE

#define GENERATE_TX_CONVERT(rs1typename, rs1type, rs3typename, rs3type, modename, mode)            \
    template <long tcfgrs1, long tcfgrs3>                                                          \
    __attribute__((always_inline)) static inline void                                              \
        __tx_convert_##rs1typename##_##rs3typename##modename(void *rs1, void *rs3) {               \
        NPUInstBuilder::__tx_convert_rs1type_rs3type_mode<NPUInstBuilder::rs1type,                 \
                                                          NPUInstBuilder::rs3type,                 \
                                                          NPUInstBuilder::mode, tcfgrs1, tcfgrs3>( \
            rs1, rs3);                                                                             \
    }

GENERATE_TX_CONVERT(fp32, TX_FP32, s32, TX_S32, _round, TX_CONVERT_ROUND)
GENERATE_TX_CONVERT(fp32, TX_FP32, s32, TX_S32, _floor, TX_CONVERT_FLOOR)
GENERATE_TX_CONVERT(fp32, TX_FP32, s32, TX_S32, _ceil, TX_CONVERT_CEIL)
GENERATE_TX_CONVERT(fp32, TX_FP32, e4m3, TX_E4M3, _round, TX_CONVERT_ROUND)
GENERATE_TX_CONVERT(fp32, TX_FP32, fp16, TX_FP16, _round, TX_CONVERT_ROUND)
GENERATE_TX_CONVERT(fp32, TX_FP32, bf16, TX_BF16, _round, TX_CONVERT_ROUND)
GENERATE_TX_CONVERT(bf16, TX_BF16, fp32, TX_FP32, _round, TX_CONVERT_ROUND)
GENERATE_TX_CONVERT(s32, TX_S32, fp32, TX_FP32, , TX_CONVERT_ROUND)

#undef GENERATE_TX_CONVERT

#define GENERATE_TX_FP_REDUCE_TYPE(opname, op, typename, type)                                     \
    template <long tcfgrs1, long tcfgrs3, long dim>                                                \
    __attribute__((always_inline)) static inline void __tx_fp_reduce_##opname##_##                 \
        typename(void *rs1, void *rs3) {                                                           \
        NPUInstBuilder::__tx_fp_reduce_op_type<NPUInstBuilder::op, NPUInstBuilder::type, tcfgrs1,  \
                                               tcfgrs3, dim>(rs1, rs3);                            \
    }

GENERATE_TX_FP_REDUCE_TYPE(max, TX_FP_MAX, fp32, TX_FP32)
GENERATE_TX_FP_REDUCE_TYPE(min, TX_FP_MIN, fp32, TX_FP32)
GENERATE_TX_FP_REDUCE_TYPE(add, TX_FP_SUM, fp32, TX_FP32)
#undef GENERATE_TX_FP_REDUCE_TYPE

#define GENERATE_TX_INT_REDUCE_TYPE(opname, op, typename, type)                                    \
    template <long tcfgrs1, long tcfgrs3, long dim>                                                \
    __attribute__((always_inline)) static inline void __tx_int_reduce_##opname##_##                \
        typename(void *rs1, void *rs3) {                                                           \
        NPUInstBuilder::__tx_int_reduce_op_type<NPUInstBuilder::op, NPUInstBuilder::type, tcfgrs1, \
                                                tcfgrs3, dim>(rs1, rs3);                           \
    }
GENERATE_TX_INT_REDUCE_TYPE(max, TX_INT_MAX, s32, TX_S32)
GENERATE_TX_INT_REDUCE_TYPE(min, TX_INT_MIN, s32, TX_S32)
GENERATE_TX_INT_REDUCE_TYPE(add, TX_INT_SUM, s32, TX_S32)
#undef GENERATE_TX_INT_REDUCE_TYPE

#define GENERATE_TX_LUT_INT_TYPE(typename, type)                                                   \
    template <long tcfgrs1, long width, long tcfgrs3, long log2Length>                             \
    __attribute__((always_inline)) static inline void __tx_lut_int_##                              \
        typename(void *rs1, void *rs2, void *rs3) {                                                \
        NPUInstBuilder::__tx_lut_int_type<NPUInstBuilder::type, tcfgrs1, width, tcfgrs3,           \
                                          log2Length>(rs1, rs2, rs3);                              \
    }
GENERATE_TX_LUT_INT_TYPE(s32, TX_S32)
#undef GENERATE_TX_LUT_INT_TYPE

#define GENERATE_TX_SFU_OP_TYPE(opname, op, typename, type)                                        \
    template <long tcfgrs1, long tcfgrs3>                                                          \
    __attribute__((always_inline)) static inline void __tx_sfu_##opname##_##                       \
        typename(void *rs1, void *rs3) {                                                           \
        NPUInstBuilder::__tx_sfu_op_type<NPUInstBuilder::op, NPUInstBuilder::type, tcfgrs1,        \
                                         tcfgrs3>(rs1, rs3);                                       \
    }
GENERATE_TX_SFU_OP_TYPE(recip, TX_SFU_RECIP, fp32, TX_FP32)
GENERATE_TX_SFU_OP_TYPE(rsqrt, TX_SFU_RSQRT, fp32, TX_FP32)
GENERATE_TX_SFU_OP_TYPE(sin, TX_SFU_SIN, fp32, TX_FP32)
GENERATE_TX_SFU_OP_TYPE(cos, TX_SFU_COS, fp32, TX_FP32)
GENERATE_TX_SFU_OP_TYPE(exp, TX_SFU_EXP, fp32, TX_FP32)
GENERATE_TX_SFU_OP_TYPE(relu, TX_SFU_RELU, fp32, TX_FP32)
#undef GENERATE_TX_SFU_OP_TYPE

// New Barrier and Produce / Consume Instructions
__attribute__((always_inline)) static inline void __tx_sb_produce(long rd) {
    NPUInstBuilder::__custom_64b_instruction1r<((long)0b11111111 << 56ul) | ((long)0b0000 << 52ul)>(
        rd);
}
__attribute__((always_inline)) static inline void __tx_sb_bar() {
    NPUInstBuilder::__custom_64b_instruction_bar<((long)0b11111111 << 56ul) |
                                                 ((long)0b0001 << 52ul)>();
}

#include "gen_npu_instruction.h"
static inline void __tx_lsu_sb_bar(void) {
    long some_val;
    tx_lsu_sb_bar(some_val);
    tx_lsu_sb_bar(some_val);
}

#endif