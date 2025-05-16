#ifndef _NPU_INSTRUCTION_HPP_
#define _NPU_INSTRUCTION_HPP_

#define NRAM_SIZE 1024 * 256 // 256KB

#ifdef __EMULATED_RVT_EXT__

struct {
    long dim0;
    long stride0;
    long dim1;
    long stride1;
    long dim2;
    long stride2;
    long dim3;
    long stride3;
} TCFG;

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
};
extern RVTEnv* __rvt_env;
#endif

namespace NPUInstBuilder {
    template <long encoding>
    static inline void __custom_64b_instruction3r(long rs1, long rs2, long rs3, long& rd) 
    {
        asm volatile (
            "\n"
            ".ifc %0,zero\n.set __REG0,0\n.endif\n.ifc %0,ra\n.set __REG0,1\n.endif\n.ifc %0,sp\n.set __REG0,2\n.endif\n.ifc %0,gp\n.set __REG0,3\n.endif\n.ifc %0,tp\n.set __REG0,4\n.endif\n.ifc %0,t0\n.set __REG0,5\n.endif\n.ifc %0,t1\n.set __REG0,6\n.endif\n.ifc %0,t2\n.set __REG0,7\n.endif\n.ifc %0,s0\n.set __REG0,8\n.endif\n.ifc %0,s1\n.set __REG0,9\n.endif\n.ifc %0,a0\n.set __REG0,10\n.endif\n.ifc %0,a1\n.set __REG0,11\n.endif\n.ifc %0,a2\n.set __REG0,12\n.endif\n.ifc %0,a3\n.set __REG0,13\n.endif\n.ifc %0,a4\n.set __REG0,14\n.endif\n.ifc %0,a5\n.set __REG0,15\n.endif\n.ifc %0,a6\n.set __REG0,16\n.endif\n.ifc %0,a7\n.set __REG0,17\n.endif\n.ifc %0,s2\n.set __REG0,18\n.endif\n.ifc %0,s3\n.set __REG0,19\n.endif\n.ifc %0,s4\n.set __REG0,20\n.endif\n.ifc %0,s5\n.set __REG0,21\n.endif\n.ifc %0,s6\n.set __REG0,22\n.endif\n.ifc %0,s7\n.set __REG0,23\n.endif\n.ifc %0,s8\n.set __REG0,24\n.endif\n.ifc %0,s9\n.set __REG0,25\n.endif\n.ifc %0,s10\n.set __REG0,26\n.endif\n.ifc %0,s11\n.set __REG0,27\n.endif\n.ifc %0,t3\n.set __REG0,28\n.endif\n.ifc %0,t4\n.set __REG0,29\n.endif\n.ifc %0,t5\n.set __REG0,30\n.endif\n.ifc %0,t6\n.set __REG0,31\n.endif\n.ifc %1,zero\n.set __REG1,0\n.endif\n.ifc %1,ra\n.set __REG1,1\n.endif\n.ifc %1,sp\n.set __REG1,2\n.endif\n.ifc %1,gp\n.set __REG1,3\n.endif\n.ifc %1,tp\n.set __REG1,4\n.endif\n.ifc %1,t0\n.set __REG1,5\n.endif\n.ifc %1,t1\n.set __REG1,6\n.endif\n.ifc %1,t2\n.set __REG1,7\n.endif\n.ifc %1,s0\n.set __REG1,8\n.endif\n.ifc %1,s1\n.set __REG1,9\n.endif\n.ifc %1,a0\n.set __REG1,10\n.endif\n.ifc %1,a1\n.set __REG1,11\n.endif\n.ifc %1,a2\n.set __REG1,12\n.endif\n.ifc %1,a3\n.set __REG1,13\n.endif\n.ifc %1,a4\n.set __REG1,14\n.endif\n.ifc %1,a5\n.set __REG1,15\n.endif\n.ifc %1,a6\n.set __REG1,16\n.endif\n.ifc %1,a7\n.set __REG1,17\n.endif\n.ifc %1,s2\n.set __REG1,18\n.endif\n.ifc %1,s3\n.set __REG1,19\n.endif\n.ifc %1,s4\n.set __REG1,20\n.endif\n.ifc %1,s5\n.set __REG1,21\n.endif\n.ifc %1,s6\n.set __REG1,22\n.endif\n.ifc %1,s7\n.set __REG1,23\n.endif\n.ifc %1,s8\n.set __REG1,24\n.endif\n.ifc %1,s9\n.set __REG1,25\n.endif\n.ifc %1,s10\n.set __REG1,26\n.endif\n.ifc %1,s11\n.set __REG1,27\n.endif\n.ifc %1,t3\n.set __REG1,28\n.endif\n.ifc %1,t4\n.set __REG1,29\n.endif\n.ifc %1,t5\n.set __REG1,30\n.endif\n.ifc %1,t6\n.set __REG1,31\n.endif\n.ifc %2,zero\n.set __REG2,0\n.endif\n.ifc %2,ra\n.set __REG2,1\n.endif\n.ifc %2,sp\n.set __REG2,2\n.endif\n.ifc %2,gp\n.set __REG2,3\n.endif\n.ifc %2,tp\n.set __REG2,4\n.endif\n.ifc %2,t0\n.set __REG2,5\n.endif\n.ifc %2,t1\n.set __REG2,6\n.endif\n.ifc %2,t2\n.set __REG2,7\n.endif\n.ifc %2,s0\n.set __REG2,8\n.endif\n.ifc %2,s1\n.set __REG2,9\n.endif\n.ifc %2,a0\n.set __REG2,10\n.endif\n.ifc %2,a1\n.set __REG2,11\n.endif\n.ifc %2,a2\n.set __REG2,12\n.endif\n.ifc %2,a3\n.set __REG2,13\n.endif\n.ifc %2,a4\n.set __REG2,14\n.endif\n.ifc %2,a5\n.set __REG2,15\n.endif\n.ifc %2,a6\n.set __REG2,16\n.endif\n.ifc %2,a7\n.set __REG2,17\n.endif\n.ifc %2,s2\n.set __REG2,18\n.endif\n.ifc %2,s3\n.set __REG2,19\n.endif\n.ifc %2,s4\n.set __REG2,20\n.endif\n.ifc %2,s5\n.set __REG2,21\n.endif\n.ifc %2,s6\n.set __REG2,22\n.endif\n.ifc %2,s7\n.set __REG2,23\n.endif\n.ifc %2,s8\n.set __REG2,24\n.endif\n.ifc %2,s9\n.set __REG2,25\n.endif\n.ifc %2,s10\n.set __REG2,26\n.endif\n.ifc %2,s11\n.set __REG2,27\n.endif\n.ifc %2,t3\n.set __REG2,28\n.endif\n.ifc %2,t4\n.set __REG2,29\n.endif\n.ifc %2,t5\n.set __REG2,30\n.endif\n.ifc %2,t6\n.set __REG2,31\n.endif\n.ifc %3,zero\n.set __REG3,0\n.endif\n.ifc %3,ra\n.set __REG3,1\n.endif\n.ifc %3,sp\n.set __REG3,2\n.endif\n.ifc %3,gp\n.set __REG3,3\n.endif\n.ifc %3,tp\n.set __REG3,4\n.endif\n.ifc %3,t0\n.set __REG3,5\n.endif\n.ifc %3,t1\n.set __REG3,6\n.endif\n.ifc %3,t2\n.set __REG3,7\n.endif\n.ifc %3,s0\n.set __REG3,8\n.endif\n.ifc %3,s1\n.set __REG3,9\n.endif\n.ifc %3,a0\n.set __REG3,10\n.endif\n.ifc %3,a1\n.set __REG3,11\n.endif\n.ifc %3,a2\n.set __REG3,12\n.endif\n.ifc %3,a3\n.set __REG3,13\n.endif\n.ifc %3,a4\n.set __REG3,14\n.endif\n.ifc %3,a5\n.set __REG3,15\n.endif\n.ifc %3,a6\n.set __REG3,16\n.endif\n.ifc %3,a7\n.set __REG3,17\n.endif\n.ifc %3,s2\n.set __REG3,18\n.endif\n.ifc %3,s3\n.set __REG3,19\n.endif\n.ifc %3,s4\n.set __REG3,20\n.endif\n.ifc %3,s5\n.set __REG3,21\n.endif\n.ifc %3,s6\n.set __REG3,22\n.endif\n.ifc %3,s7\n.set __REG3,23\n.endif\n.ifc %3,s8\n.set __REG3,24\n.endif\n.ifc %3,s9\n.set __REG3,25\n.endif\n.ifc %3,s10\n.set __REG3,26\n.endif\n.ifc %3,s11\n.set __REG3,27\n.endif\n.ifc %3,t3\n.set __REG3,28\n.endif\n.ifc %3,t4\n.set __REG3,29\n.endif\n.ifc %3,t5\n.set __REG3,30\n.endif\n.ifc %3,t6\n.set __REG3,31\n.endif\n"
            ".insn 8, %4 | 0x3F | (__REG0 << 7) | (__REG1 << 15) | (__REG2 << 20) | (__REG3 << 27)\n"
            ".set __REG0,z\n.set __REG1,z\n.set __REG2,z\n.set __REG3,z\n"
            :"+r"(rd)
            :"r"(rs1), "r"(rs2), "r"(rs3), "i"(encoding)
        );
    }
    template <long encoding>
    static inline void __custom_64b_instruction3r(long rs1, long rs2, long rs3) 
    {
        asm volatile (
            "\n"
            ".ifc %0,zero\n.set __REG0,0\n.endif\n.ifc %0,ra\n.set __REG0,1\n.endif\n.ifc %0,sp\n.set __REG0,2\n.endif\n.ifc %0,gp\n.set __REG0,3\n.endif\n.ifc %0,tp\n.set __REG0,4\n.endif\n.ifc %0,t0\n.set __REG0,5\n.endif\n.ifc %0,t1\n.set __REG0,6\n.endif\n.ifc %0,t2\n.set __REG0,7\n.endif\n.ifc %0,s0\n.set __REG0,8\n.endif\n.ifc %0,s1\n.set __REG0,9\n.endif\n.ifc %0,a0\n.set __REG0,10\n.endif\n.ifc %0,a1\n.set __REG0,11\n.endif\n.ifc %0,a2\n.set __REG0,12\n.endif\n.ifc %0,a3\n.set __REG0,13\n.endif\n.ifc %0,a4\n.set __REG0,14\n.endif\n.ifc %0,a5\n.set __REG0,15\n.endif\n.ifc %0,a6\n.set __REG0,16\n.endif\n.ifc %0,a7\n.set __REG0,17\n.endif\n.ifc %0,s2\n.set __REG0,18\n.endif\n.ifc %0,s3\n.set __REG0,19\n.endif\n.ifc %0,s4\n.set __REG0,20\n.endif\n.ifc %0,s5\n.set __REG0,21\n.endif\n.ifc %0,s6\n.set __REG0,22\n.endif\n.ifc %0,s7\n.set __REG0,23\n.endif\n.ifc %0,s8\n.set __REG0,24\n.endif\n.ifc %0,s9\n.set __REG0,25\n.endif\n.ifc %0,s10\n.set __REG0,26\n.endif\n.ifc %0,s11\n.set __REG0,27\n.endif\n.ifc %0,t3\n.set __REG0,28\n.endif\n.ifc %0,t4\n.set __REG0,29\n.endif\n.ifc %0,t5\n.set __REG0,30\n.endif\n.ifc %0,t6\n.set __REG0,31\n.endif\n.ifc %1,zero\n.set __REG1,0\n.endif\n.ifc %1,ra\n.set __REG1,1\n.endif\n.ifc %1,sp\n.set __REG1,2\n.endif\n.ifc %1,gp\n.set __REG1,3\n.endif\n.ifc %1,tp\n.set __REG1,4\n.endif\n.ifc %1,t0\n.set __REG1,5\n.endif\n.ifc %1,t1\n.set __REG1,6\n.endif\n.ifc %1,t2\n.set __REG1,7\n.endif\n.ifc %1,s0\n.set __REG1,8\n.endif\n.ifc %1,s1\n.set __REG1,9\n.endif\n.ifc %1,a0\n.set __REG1,10\n.endif\n.ifc %1,a1\n.set __REG1,11\n.endif\n.ifc %1,a2\n.set __REG1,12\n.endif\n.ifc %1,a3\n.set __REG1,13\n.endif\n.ifc %1,a4\n.set __REG1,14\n.endif\n.ifc %1,a5\n.set __REG1,15\n.endif\n.ifc %1,a6\n.set __REG1,16\n.endif\n.ifc %1,a7\n.set __REG1,17\n.endif\n.ifc %1,s2\n.set __REG1,18\n.endif\n.ifc %1,s3\n.set __REG1,19\n.endif\n.ifc %1,s4\n.set __REG1,20\n.endif\n.ifc %1,s5\n.set __REG1,21\n.endif\n.ifc %1,s6\n.set __REG1,22\n.endif\n.ifc %1,s7\n.set __REG1,23\n.endif\n.ifc %1,s8\n.set __REG1,24\n.endif\n.ifc %1,s9\n.set __REG1,25\n.endif\n.ifc %1,s10\n.set __REG1,26\n.endif\n.ifc %1,s11\n.set __REG1,27\n.endif\n.ifc %1,t3\n.set __REG1,28\n.endif\n.ifc %1,t4\n.set __REG1,29\n.endif\n.ifc %1,t5\n.set __REG1,30\n.endif\n.ifc %1,t6\n.set __REG1,31\n.endif\n.ifc %2,zero\n.set __REG2,0\n.endif\n.ifc %2,ra\n.set __REG2,1\n.endif\n.ifc %2,sp\n.set __REG2,2\n.endif\n.ifc %2,gp\n.set __REG2,3\n.endif\n.ifc %2,tp\n.set __REG2,4\n.endif\n.ifc %2,t0\n.set __REG2,5\n.endif\n.ifc %2,t1\n.set __REG2,6\n.endif\n.ifc %2,t2\n.set __REG2,7\n.endif\n.ifc %2,s0\n.set __REG2,8\n.endif\n.ifc %2,s1\n.set __REG2,9\n.endif\n.ifc %2,a0\n.set __REG2,10\n.endif\n.ifc %2,a1\n.set __REG2,11\n.endif\n.ifc %2,a2\n.set __REG2,12\n.endif\n.ifc %2,a3\n.set __REG2,13\n.endif\n.ifc %2,a4\n.set __REG2,14\n.endif\n.ifc %2,a5\n.set __REG2,15\n.endif\n.ifc %2,a6\n.set __REG2,16\n.endif\n.ifc %2,a7\n.set __REG2,17\n.endif\n.ifc %2,s2\n.set __REG2,18\n.endif\n.ifc %2,s3\n.set __REG2,19\n.endif\n.ifc %2,s4\n.set __REG2,20\n.endif\n.ifc %2,s5\n.set __REG2,21\n.endif\n.ifc %2,s6\n.set __REG2,22\n.endif\n.ifc %2,s7\n.set __REG2,23\n.endif\n.ifc %2,s8\n.set __REG2,24\n.endif\n.ifc %2,s9\n.set __REG2,25\n.endif\n.ifc %2,s10\n.set __REG2,26\n.endif\n.ifc %2,s11\n.set __REG2,27\n.endif\n.ifc %2,t3\n.set __REG2,28\n.endif\n.ifc %2,t4\n.set __REG2,29\n.endif\n.ifc %2,t5\n.set __REG2,30\n.endif\n.ifc %2,t6\n.set __REG2,31\n.endif\n"
            ".insn 8, %3 | 0x3F | (__REG0 << 15) | (__REG1 << 20) | (__REG2 << 27)\n"
            ".set __REG0,z\n.set __REG1,z\n.set __REG2,z\n.set __REG3,z\n"
            ::"r"(rs1), "r"(rs2), "r"(rs3), "i"(encoding)
        );
    }
    template <long encoding>
    static inline void __custom_64b_instruction2r(long rs1, long rs3, long& rd) 
    {
        asm volatile (
            "\n"
            ".ifc %0,zero\n.set __REG0,0\n.endif\n.ifc %0,ra\n.set __REG0,1\n.endif\n.ifc %0,sp\n.set __REG0,2\n.endif\n.ifc %0,gp\n.set __REG0,3\n.endif\n.ifc %0,tp\n.set __REG0,4\n.endif\n.ifc %0,t0\n.set __REG0,5\n.endif\n.ifc %0,t1\n.set __REG0,6\n.endif\n.ifc %0,t2\n.set __REG0,7\n.endif\n.ifc %0,s0\n.set __REG0,8\n.endif\n.ifc %0,s1\n.set __REG0,9\n.endif\n.ifc %0,a0\n.set __REG0,10\n.endif\n.ifc %0,a1\n.set __REG0,11\n.endif\n.ifc %0,a2\n.set __REG0,12\n.endif\n.ifc %0,a3\n.set __REG0,13\n.endif\n.ifc %0,a4\n.set __REG0,14\n.endif\n.ifc %0,a5\n.set __REG0,15\n.endif\n.ifc %0,a6\n.set __REG0,16\n.endif\n.ifc %0,a7\n.set __REG0,17\n.endif\n.ifc %0,s2\n.set __REG0,18\n.endif\n.ifc %0,s3\n.set __REG0,19\n.endif\n.ifc %0,s4\n.set __REG0,20\n.endif\n.ifc %0,s5\n.set __REG0,21\n.endif\n.ifc %0,s6\n.set __REG0,22\n.endif\n.ifc %0,s7\n.set __REG0,23\n.endif\n.ifc %0,s8\n.set __REG0,24\n.endif\n.ifc %0,s9\n.set __REG0,25\n.endif\n.ifc %0,s10\n.set __REG0,26\n.endif\n.ifc %0,s11\n.set __REG0,27\n.endif\n.ifc %0,t3\n.set __REG0,28\n.endif\n.ifc %0,t4\n.set __REG0,29\n.endif\n.ifc %0,t5\n.set __REG0,30\n.endif\n.ifc %0,t6\n.set __REG0,31\n.endif\n.ifc %1,zero\n.set __REG1,0\n.endif\n.ifc %1,ra\n.set __REG1,1\n.endif\n.ifc %1,sp\n.set __REG1,2\n.endif\n.ifc %1,gp\n.set __REG1,3\n.endif\n.ifc %1,tp\n.set __REG1,4\n.endif\n.ifc %1,t0\n.set __REG1,5\n.endif\n.ifc %1,t1\n.set __REG1,6\n.endif\n.ifc %1,t2\n.set __REG1,7\n.endif\n.ifc %1,s0\n.set __REG1,8\n.endif\n.ifc %1,s1\n.set __REG1,9\n.endif\n.ifc %1,a0\n.set __REG1,10\n.endif\n.ifc %1,a1\n.set __REG1,11\n.endif\n.ifc %1,a2\n.set __REG1,12\n.endif\n.ifc %1,a3\n.set __REG1,13\n.endif\n.ifc %1,a4\n.set __REG1,14\n.endif\n.ifc %1,a5\n.set __REG1,15\n.endif\n.ifc %1,a6\n.set __REG1,16\n.endif\n.ifc %1,a7\n.set __REG1,17\n.endif\n.ifc %1,s2\n.set __REG1,18\n.endif\n.ifc %1,s3\n.set __REG1,19\n.endif\n.ifc %1,s4\n.set __REG1,20\n.endif\n.ifc %1,s5\n.set __REG1,21\n.endif\n.ifc %1,s6\n.set __REG1,22\n.endif\n.ifc %1,s7\n.set __REG1,23\n.endif\n.ifc %1,s8\n.set __REG1,24\n.endif\n.ifc %1,s9\n.set __REG1,25\n.endif\n.ifc %1,s10\n.set __REG1,26\n.endif\n.ifc %1,s11\n.set __REG1,27\n.endif\n.ifc %1,t3\n.set __REG1,28\n.endif\n.ifc %1,t4\n.set __REG1,29\n.endif\n.ifc %1,t5\n.set __REG1,30\n.endif\n.ifc %1,t6\n.set __REG1,31\n.endif\n.ifc %2,zero\n.set __REG2,0\n.endif\n.ifc %2,ra\n.set __REG2,1\n.endif\n.ifc %2,sp\n.set __REG2,2\n.endif\n.ifc %2,gp\n.set __REG2,3\n.endif\n.ifc %2,tp\n.set __REG2,4\n.endif\n.ifc %2,t0\n.set __REG2,5\n.endif\n.ifc %2,t1\n.set __REG2,6\n.endif\n.ifc %2,t2\n.set __REG2,7\n.endif\n.ifc %2,s0\n.set __REG2,8\n.endif\n.ifc %2,s1\n.set __REG2,9\n.endif\n.ifc %2,a0\n.set __REG2,10\n.endif\n.ifc %2,a1\n.set __REG2,11\n.endif\n.ifc %2,a2\n.set __REG2,12\n.endif\n.ifc %2,a3\n.set __REG2,13\n.endif\n.ifc %2,a4\n.set __REG2,14\n.endif\n.ifc %2,a5\n.set __REG2,15\n.endif\n.ifc %2,a6\n.set __REG2,16\n.endif\n.ifc %2,a7\n.set __REG2,17\n.endif\n.ifc %2,s2\n.set __REG2,18\n.endif\n.ifc %2,s3\n.set __REG2,19\n.endif\n.ifc %2,s4\n.set __REG2,20\n.endif\n.ifc %2,s5\n.set __REG2,21\n.endif\n.ifc %2,s6\n.set __REG2,22\n.endif\n.ifc %2,s7\n.set __REG2,23\n.endif\n.ifc %2,s8\n.set __REG2,24\n.endif\n.ifc %2,s9\n.set __REG2,25\n.endif\n.ifc %2,s10\n.set __REG2,26\n.endif\n.ifc %2,s11\n.set __REG2,27\n.endif\n.ifc %2,t3\n.set __REG2,28\n.endif\n.ifc %2,t4\n.set __REG2,29\n.endif\n.ifc %2,t5\n.set __REG2,30\n.endif\n.ifc %2,t6\n.set __REG2,31\n.endif\n"
            ".insn 8, %3 | 0x3F | (__REG0 << 7) | (__REG1 << 15) | (__REG2 << 27)\n"
            ".set __REG0,z\n.set __REG1,z\n.set __REG2,z\n.set __REG3,z\n"
            :"+r"(rd)
            :"r"(rs1), "r"(rs3), "i"(encoding)
        );
    }
    template <long encoding>
    static inline void __custom_64b_instruction2r(long rs1, long rs3) 
    {
        asm volatile (
            "\n"
            ".ifc %0,zero\n.set __REG0,0\n.endif\n.ifc %0,ra\n.set __REG0,1\n.endif\n.ifc %0,sp\n.set __REG0,2\n.endif\n.ifc %0,gp\n.set __REG0,3\n.endif\n.ifc %0,tp\n.set __REG0,4\n.endif\n.ifc %0,t0\n.set __REG0,5\n.endif\n.ifc %0,t1\n.set __REG0,6\n.endif\n.ifc %0,t2\n.set __REG0,7\n.endif\n.ifc %0,s0\n.set __REG0,8\n.endif\n.ifc %0,s1\n.set __REG0,9\n.endif\n.ifc %0,a0\n.set __REG0,10\n.endif\n.ifc %0,a1\n.set __REG0,11\n.endif\n.ifc %0,a2\n.set __REG0,12\n.endif\n.ifc %0,a3\n.set __REG0,13\n.endif\n.ifc %0,a4\n.set __REG0,14\n.endif\n.ifc %0,a5\n.set __REG0,15\n.endif\n.ifc %0,a6\n.set __REG0,16\n.endif\n.ifc %0,a7\n.set __REG0,17\n.endif\n.ifc %0,s2\n.set __REG0,18\n.endif\n.ifc %0,s3\n.set __REG0,19\n.endif\n.ifc %0,s4\n.set __REG0,20\n.endif\n.ifc %0,s5\n.set __REG0,21\n.endif\n.ifc %0,s6\n.set __REG0,22\n.endif\n.ifc %0,s7\n.set __REG0,23\n.endif\n.ifc %0,s8\n.set __REG0,24\n.endif\n.ifc %0,s9\n.set __REG0,25\n.endif\n.ifc %0,s10\n.set __REG0,26\n.endif\n.ifc %0,s11\n.set __REG0,27\n.endif\n.ifc %0,t3\n.set __REG0,28\n.endif\n.ifc %0,t4\n.set __REG0,29\n.endif\n.ifc %0,t5\n.set __REG0,30\n.endif\n.ifc %0,t6\n.set __REG0,31\n.endif\n.ifc %1,zero\n.set __REG1,0\n.endif\n.ifc %1,ra\n.set __REG1,1\n.endif\n.ifc %1,sp\n.set __REG1,2\n.endif\n.ifc %1,gp\n.set __REG1,3\n.endif\n.ifc %1,tp\n.set __REG1,4\n.endif\n.ifc %1,t0\n.set __REG1,5\n.endif\n.ifc %1,t1\n.set __REG1,6\n.endif\n.ifc %1,t2\n.set __REG1,7\n.endif\n.ifc %1,s0\n.set __REG1,8\n.endif\n.ifc %1,s1\n.set __REG1,9\n.endif\n.ifc %1,a0\n.set __REG1,10\n.endif\n.ifc %1,a1\n.set __REG1,11\n.endif\n.ifc %1,a2\n.set __REG1,12\n.endif\n.ifc %1,a3\n.set __REG1,13\n.endif\n.ifc %1,a4\n.set __REG1,14\n.endif\n.ifc %1,a5\n.set __REG1,15\n.endif\n.ifc %1,a6\n.set __REG1,16\n.endif\n.ifc %1,a7\n.set __REG1,17\n.endif\n.ifc %1,s2\n.set __REG1,18\n.endif\n.ifc %1,s3\n.set __REG1,19\n.endif\n.ifc %1,s4\n.set __REG1,20\n.endif\n.ifc %1,s5\n.set __REG1,21\n.endif\n.ifc %1,s6\n.set __REG1,22\n.endif\n.ifc %1,s7\n.set __REG1,23\n.endif\n.ifc %1,s8\n.set __REG1,24\n.endif\n.ifc %1,s9\n.set __REG1,25\n.endif\n.ifc %1,s10\n.set __REG1,26\n.endif\n.ifc %1,s11\n.set __REG1,27\n.endif\n.ifc %1,t3\n.set __REG1,28\n.endif\n.ifc %1,t4\n.set __REG1,29\n.endif\n.ifc %1,t5\n.set __REG1,30\n.endif\n.ifc %1,t6\n.set __REG1,31\n.endif\n"
            ".insn 8, %2 | 0x3F | (__REG0 << 15) | (__REG1 << 27)\n"
            ".set __REG0,z\n.set __REG1,z\n.set __REG2,z\n.set __REG3,z\n"
            ::"r"(rs1), "r"(rs3), "i"(encoding)
        );
    }
    template <long encoding>
    static inline void __custom_64b_instruction1r(long rs3, long& rd) 
    {
        asm volatile (
            "\n"
            ".ifc %0,zero\n.set __REG0,0\n.endif\n.ifc %0,ra\n.set __REG0,1\n.endif\n.ifc %0,sp\n.set __REG0,2\n.endif\n.ifc %0,gp\n.set __REG0,3\n.endif\n.ifc %0,tp\n.set __REG0,4\n.endif\n.ifc %0,t0\n.set __REG0,5\n.endif\n.ifc %0,t1\n.set __REG0,6\n.endif\n.ifc %0,t2\n.set __REG0,7\n.endif\n.ifc %0,s0\n.set __REG0,8\n.endif\n.ifc %0,s1\n.set __REG0,9\n.endif\n.ifc %0,a0\n.set __REG0,10\n.endif\n.ifc %0,a1\n.set __REG0,11\n.endif\n.ifc %0,a2\n.set __REG0,12\n.endif\n.ifc %0,a3\n.set __REG0,13\n.endif\n.ifc %0,a4\n.set __REG0,14\n.endif\n.ifc %0,a5\n.set __REG0,15\n.endif\n.ifc %0,a6\n.set __REG0,16\n.endif\n.ifc %0,a7\n.set __REG0,17\n.endif\n.ifc %0,s2\n.set __REG0,18\n.endif\n.ifc %0,s3\n.set __REG0,19\n.endif\n.ifc %0,s4\n.set __REG0,20\n.endif\n.ifc %0,s5\n.set __REG0,21\n.endif\n.ifc %0,s6\n.set __REG0,22\n.endif\n.ifc %0,s7\n.set __REG0,23\n.endif\n.ifc %0,s8\n.set __REG0,24\n.endif\n.ifc %0,s9\n.set __REG0,25\n.endif\n.ifc %0,s10\n.set __REG0,26\n.endif\n.ifc %0,s11\n.set __REG0,27\n.endif\n.ifc %0,t3\n.set __REG0,28\n.endif\n.ifc %0,t4\n.set __REG0,29\n.endif\n.ifc %0,t5\n.set __REG0,30\n.endif\n.ifc %0,t6\n.set __REG0,31\n.endif\n.ifc %1,zero\n.set __REG1,0\n.endif\n.ifc %1,ra\n.set __REG1,1\n.endif\n.ifc %1,sp\n.set __REG1,2\n.endif\n.ifc %1,gp\n.set __REG1,3\n.endif\n.ifc %1,tp\n.set __REG1,4\n.endif\n.ifc %1,t0\n.set __REG1,5\n.endif\n.ifc %1,t1\n.set __REG1,6\n.endif\n.ifc %1,t2\n.set __REG1,7\n.endif\n.ifc %1,s0\n.set __REG1,8\n.endif\n.ifc %1,s1\n.set __REG1,9\n.endif\n.ifc %1,a0\n.set __REG1,10\n.endif\n.ifc %1,a1\n.set __REG1,11\n.endif\n.ifc %1,a2\n.set __REG1,12\n.endif\n.ifc %1,a3\n.set __REG1,13\n.endif\n.ifc %1,a4\n.set __REG1,14\n.endif\n.ifc %1,a5\n.set __REG1,15\n.endif\n.ifc %1,a6\n.set __REG1,16\n.endif\n.ifc %1,a7\n.set __REG1,17\n.endif\n.ifc %1,s2\n.set __REG1,18\n.endif\n.ifc %1,s3\n.set __REG1,19\n.endif\n.ifc %1,s4\n.set __REG1,20\n.endif\n.ifc %1,s5\n.set __REG1,21\n.endif\n.ifc %1,s6\n.set __REG1,22\n.endif\n.ifc %1,s7\n.set __REG1,23\n.endif\n.ifc %1,s8\n.set __REG1,24\n.endif\n.ifc %1,s9\n.set __REG1,25\n.endif\n.ifc %1,s10\n.set __REG1,26\n.endif\n.ifc %1,s11\n.set __REG1,27\n.endif\n.ifc %1,t3\n.set __REG1,28\n.endif\n.ifc %1,t4\n.set __REG1,29\n.endif\n.ifc %1,t5\n.set __REG1,30\n.endif\n.ifc %1,t6\n.set __REG1,31\n.endif\n"
            ".insn 8, %2 | 0x3F | (__REG0 << 7) | (__REG1 << 27)\n"
            ".set __REG0,z\n.set __REG1,z\n.set __REG2,z\n.set __REG3,z\n"
            :"+r"(rd)
            :"r"(rs3), "i"(encoding)
        );
    }
    template <long encoding>
    static inline void __custom_64b_instruction1r(long rs3) 
    {
        asm volatile (
            "\n"
            ".ifc %0,zero\n.set __REG0,0\n.endif\n.ifc %0,ra\n.set __REG0,1\n.endif\n.ifc %0,sp\n.set __REG0,2\n.endif\n.ifc %0,gp\n.set __REG0,3\n.endif\n.ifc %0,tp\n.set __REG0,4\n.endif\n.ifc %0,t0\n.set __REG0,5\n.endif\n.ifc %0,t1\n.set __REG0,6\n.endif\n.ifc %0,t2\n.set __REG0,7\n.endif\n.ifc %0,s0\n.set __REG0,8\n.endif\n.ifc %0,s1\n.set __REG0,9\n.endif\n.ifc %0,a0\n.set __REG0,10\n.endif\n.ifc %0,a1\n.set __REG0,11\n.endif\n.ifc %0,a2\n.set __REG0,12\n.endif\n.ifc %0,a3\n.set __REG0,13\n.endif\n.ifc %0,a4\n.set __REG0,14\n.endif\n.ifc %0,a5\n.set __REG0,15\n.endif\n.ifc %0,a6\n.set __REG0,16\n.endif\n.ifc %0,a7\n.set __REG0,17\n.endif\n.ifc %0,s2\n.set __REG0,18\n.endif\n.ifc %0,s3\n.set __REG0,19\n.endif\n.ifc %0,s4\n.set __REG0,20\n.endif\n.ifc %0,s5\n.set __REG0,21\n.endif\n.ifc %0,s6\n.set __REG0,22\n.endif\n.ifc %0,s7\n.set __REG0,23\n.endif\n.ifc %0,s8\n.set __REG0,24\n.endif\n.ifc %0,s9\n.set __REG0,25\n.endif\n.ifc %0,s10\n.set __REG0,26\n.endif\n.ifc %0,s11\n.set __REG0,27\n.endif\n.ifc %0,t3\n.set __REG0,28\n.endif\n.ifc %0,t4\n.set __REG0,29\n.endif\n.ifc %0,t5\n.set __REG0,30\n.endif\n.ifc %0,t6\n.set __REG0,31\n.endif\n"
            ".insn 8, %1 | 0x3F | (__REG0 << 27)\n"
            ".set __REG0,z\n.set __REG1,z\n.set __REG2,z\n.set __REG3,z\n"
            :"r"(rs3), "i"(encoding)
        );
    }

#ifndef __EMULATED_RVT_EXT__
    template <long basetype, long rs1type, long rs2type, long rs3type, long tcfgrs1, long tcfgrs2, long tcfgrs3>
    static inline void __tx_matmul(long rs1, long rs2, long rs3, long& rd)
    {
        __custom_64b_instruction3r<
        (basetype << 56ul) | (rs1type << 50ul) | (rs2type << 44ul) | (rs3type << 43ul) | (tcfgrs1 << 32ul) | (tcfgrs2 << 35ul) | (tcfgrs3 << 38ul)>
        (rs1, rs2, rs3, rd);
    }
    template <long basetype, long rs1type, long rs2type, long rs3type, long tcfgrs1, long tcfgrs2, long tcfgrs3>
    static inline void __tx_matmul(long rs1, long rs2, long rs3)
    {
        __custom_64b_instruction3r<
        (basetype << 56ul) | (rs1type << 50ul) | (rs2type << 44ul) | (rs3type << 43ul) | (tcfgrs1 << 32ul) | (tcfgrs2 << 35ul) | (tcfgrs3 << 38ul)>
        (rs1, rs2, rs3);
    }
#else
#include "npu_instruction_emulated.hpp"
#endif
}

// public
template <long tcfgrs1, long tcfgrs2, long tcfgrs3>
static inline void __tx_matmul_subfloat_e4m3_e4m3_bf16(long rs1, long rs2, long rs3, long& rd) {
    NPUInstBuilder::__tx_matmul<0x0, 0b011100, 0b011100, 0b1, tcfgrs1, tcfgrs2, tcfgrs3>(rs1, rs2, rs3, rd);
}
template <long tcfgrs1, long tcfgrs2, long tcfgrs3>
static inline void __tx_matmul_subfloat_e4m3_e4m3_bf16(long rs1, long rs2, long rs3)
{
    return NPUInstBuilder::__tx_matmul<0x0, 0b011100, 0b011100, 0b1, tcfgrs1, tcfgrs2, tcfgrs3>(rs1, rs2, rs3);
}

#endif