#ifndef _NPU_INSTRUCTION_HPP_
#define _NPU_INSTRUCTION_HPP_

#define _inst_builder_t template <long encoding> \
__attribute__((always_inline)) static inline void 

namespace BaseNPUInstBuilder {
    _inst_builder_t __custom_32b_instruction_mat(long rd, long rs1, long rs2)
    {
        asm volatile (
            "\n"
            ".ifc %0,zero\n.set _R0,0\n.endif\n.ifc %0,ra\n.set _R0,1\n.endif\n.ifc %0,sp\n.set _R0,2\n.endif\n.ifc %0,gp\n.set _R0,3\n.endif\n.ifc %0,tp\n.set _R0,4\n.endif\n.ifc %0,t0\n.set _R0,5\n.endif\n.ifc %0,t1\n.set _R0,6\n.endif\n.ifc %0,t2\n.set _R0,7\n.endif\n.ifc %0,s0\n.set _R0,8\n.endif\n.ifc %0,s1\n.set _R0,9\n.endif\n.ifc %0,a0\n.set _R0,10\n.endif\n.ifc %0,a1\n.set _R0,11\n.endif\n.ifc %0,a2\n.set _R0,12\n.endif\n.ifc %0,a3\n.set _R0,13\n.endif\n.ifc %0,a4\n.set _R0,14\n.endif\n.ifc %0,a5\n.set _R0,15\n.endif\n.ifc %0,a6\n.set _R0,16\n.endif\n.ifc %0,a7\n.set _R0,17\n.endif\n.ifc %0,s2\n.set _R0,18\n.endif\n.ifc %0,s3\n.set _R0,19\n.endif\n.ifc %0,s4\n.set _R0,20\n.endif\n.ifc %0,s5\n.set _R0,21\n.endif\n.ifc %0,s6\n.set _R0,22\n.endif\n.ifc %0,s7\n.set _R0,23\n.endif\n.ifc %0,s8\n.set _R0,24\n.endif\n.ifc %0,s9\n.set _R0,25\n.endif\n.ifc %0,s10\n.set _R0,26\n.endif\n.ifc %0,s11\n.set _R0,27\n.endif\n.ifc %0,t3\n.set _R0,28\n.endif\n.ifc %0,t4\n.set _R0,29\n.endif\n.ifc %0,t5\n.set _R0,30\n.endif\n.ifc %0,t6\n.set _R0,31\n.endif\n.ifc %1,zero\n.set _R1,0\n.endif\n.ifc %1,ra\n.set _R1,1\n.endif\n.ifc %1,sp\n.set _R1,2\n.endif\n.ifc %1,gp\n.set _R1,3\n.endif\n.ifc %1,tp\n.set _R1,4\n.endif\n.ifc %1,t0\n.set _R1,5\n.endif\n.ifc %1,t1\n.set _R1,6\n.endif\n.ifc %1,t2\n.set _R1,7\n.endif\n.ifc %1,s0\n.set _R1,8\n.endif\n.ifc %1,s1\n.set _R1,9\n.endif\n.ifc %1,a0\n.set _R1,10\n.endif\n.ifc %1,a1\n.set _R1,11\n.endif\n.ifc %1,a2\n.set _R1,12\n.endif\n.ifc %1,a3\n.set _R1,13\n.endif\n.ifc %1,a4\n.set _R1,14\n.endif\n.ifc %1,a5\n.set _R1,15\n.endif\n.ifc %1,a6\n.set _R1,16\n.endif\n.ifc %1,a7\n.set _R1,17\n.endif\n.ifc %1,s2\n.set _R1,18\n.endif\n.ifc %1,s3\n.set _R1,19\n.endif\n.ifc %1,s4\n.set _R1,20\n.endif\n.ifc %1,s5\n.set _R1,21\n.endif\n.ifc %1,s6\n.set _R1,22\n.endif\n.ifc %1,s7\n.set _R1,23\n.endif\n.ifc %1,s8\n.set _R1,24\n.endif\n.ifc %1,s9\n.set _R1,25\n.endif\n.ifc %1,s10\n.set _R1,26\n.endif\n.ifc %1,s11\n.set _R1,27\n.endif\n.ifc %1,t3\n.set _R1,28\n.endif\n.ifc %1,t4\n.set _R1,29\n.endif\n.ifc %1,t5\n.set _R1,30\n.endif\n.ifc %1,t6\n.set _R1,31\n.endif\n.ifc %2,zero\n.set _R2,0\n.endif\n.ifc %2,ra\n.set _R2,1\n.endif\n.ifc %2,sp\n.set _R2,2\n.endif\n.ifc %2,gp\n.set _R2,3\n.endif\n.ifc %2,tp\n.set _R2,4\n.endif\n.ifc %2,t0\n.set _R2,5\n.endif\n.ifc %2,t1\n.set _R2,6\n.endif\n.ifc %2,t2\n.set _R2,7\n.endif\n.ifc %2,s0\n.set _R2,8\n.endif\n.ifc %2,s1\n.set _R2,9\n.endif\n.ifc %2,a0\n.set _R2,10\n.endif\n.ifc %2,a1\n.set _R2,11\n.endif\n.ifc %2,a2\n.set _R2,12\n.endif\n.ifc %2,a3\n.set _R2,13\n.endif\n.ifc %2,a4\n.set _R2,14\n.endif\n.ifc %2,a5\n.set _R2,15\n.endif\n.ifc %2,a6\n.set _R2,16\n.endif\n.ifc %2,a7\n.set _R2,17\n.endif\n.ifc %2,s2\n.set _R2,18\n.endif\n.ifc %2,s3\n.set _R2,19\n.endif\n.ifc %2,s4\n.set _R2,20\n.endif\n.ifc %2,s5\n.set _R2,21\n.endif\n.ifc %2,s6\n.set _R2,22\n.endif\n.ifc %2,s7\n.set _R2,23\n.endif\n.ifc %2,s8\n.set _R2,24\n.endif\n.ifc %2,s9\n.set _R2,25\n.endif\n.ifc %2,s10\n.set _R2,26\n.endif\n.ifc %2,s11\n.set _R2,27\n.endif\n.ifc %2,t3\n.set _R2,28\n.endif\n.ifc %2,t4\n.set _R2,29\n.endif\n.ifc %2,t5\n.set _R2,30\n.endif\n.ifc %2,t6\n.set _R2,31\n.endif\n"
            ".align 2\n"
            ".insn 4, %3 | (_R0 << 7) | (_R1 << 15) | (_R2 << 20)\n"
            ".set _R0,z\n.set _R1,z\n.set _R2,z\n"
            ::"r"(rd), "r"(rs1), "r"(rs2), "i"(encoding)
        );
    }
    _inst_builder_t __custom_32b_instruction_lsu(long rd, long rs1)
    {
        asm volatile (
            "\n"
            ".ifc %0,zero\n.set _R0,0\n.endif\n.ifc %0,ra\n.set _R0,1\n.endif\n.ifc %0,sp\n.set _R0,2\n.endif\n.ifc %0,gp\n.set _R0,3\n.endif\n.ifc %0,tp\n.set _R0,4\n.endif\n.ifc %0,t0\n.set _R0,5\n.endif\n.ifc %0,t1\n.set _R0,6\n.endif\n.ifc %0,t2\n.set _R0,7\n.endif\n.ifc %0,s0\n.set _R0,8\n.endif\n.ifc %0,s1\n.set _R0,9\n.endif\n.ifc %0,a0\n.set _R0,10\n.endif\n.ifc %0,a1\n.set _R0,11\n.endif\n.ifc %0,a2\n.set _R0,12\n.endif\n.ifc %0,a3\n.set _R0,13\n.endif\n.ifc %0,a4\n.set _R0,14\n.endif\n.ifc %0,a5\n.set _R0,15\n.endif\n.ifc %0,a6\n.set _R0,16\n.endif\n.ifc %0,a7\n.set _R0,17\n.endif\n.ifc %0,s2\n.set _R0,18\n.endif\n.ifc %0,s3\n.set _R0,19\n.endif\n.ifc %0,s4\n.set _R0,20\n.endif\n.ifc %0,s5\n.set _R0,21\n.endif\n.ifc %0,s6\n.set _R0,22\n.endif\n.ifc %0,s7\n.set _R0,23\n.endif\n.ifc %0,s8\n.set _R0,24\n.endif\n.ifc %0,s9\n.set _R0,25\n.endif\n.ifc %0,s10\n.set _R0,26\n.endif\n.ifc %0,s11\n.set _R0,27\n.endif\n.ifc %0,t3\n.set _R0,28\n.endif\n.ifc %0,t4\n.set _R0,29\n.endif\n.ifc %0,t5\n.set _R0,30\n.endif\n.ifc %0,t6\n.set _R0,31\n.endif\n.ifc %1,zero\n.set _R1,0\n.endif\n.ifc %1,ra\n.set _R1,1\n.endif\n.ifc %1,sp\n.set _R1,2\n.endif\n.ifc %1,gp\n.set _R1,3\n.endif\n.ifc %1,tp\n.set _R1,4\n.endif\n.ifc %1,t0\n.set _R1,5\n.endif\n.ifc %1,t1\n.set _R1,6\n.endif\n.ifc %1,t2\n.set _R1,7\n.endif\n.ifc %1,s0\n.set _R1,8\n.endif\n.ifc %1,s1\n.set _R1,9\n.endif\n.ifc %1,a0\n.set _R1,10\n.endif\n.ifc %1,a1\n.set _R1,11\n.endif\n.ifc %1,a2\n.set _R1,12\n.endif\n.ifc %1,a3\n.set _R1,13\n.endif\n.ifc %1,a4\n.set _R1,14\n.endif\n.ifc %1,a5\n.set _R1,15\n.endif\n.ifc %1,a6\n.set _R1,16\n.endif\n.ifc %1,a7\n.set _R1,17\n.endif\n.ifc %1,s2\n.set _R1,18\n.endif\n.ifc %1,s3\n.set _R1,19\n.endif\n.ifc %1,s4\n.set _R1,20\n.endif\n.ifc %1,s5\n.set _R1,21\n.endif\n.ifc %1,s6\n.set _R1,22\n.endif\n.ifc %1,s7\n.set _R1,23\n.endif\n.ifc %1,s8\n.set _R1,24\n.endif\n.ifc %1,s9\n.set _R1,25\n.endif\n.ifc %1,s10\n.set _R1,26\n.endif\n.ifc %1,s11\n.set _R1,27\n.endif\n.ifc %1,t3\n.set _R1,28\n.endif\n.ifc %1,t4\n.set _R1,29\n.endif\n.ifc %1,t5\n.set _R1,30\n.endif\n.ifc %1,t6\n.set _R1,31\n.endif\n"
            ".align 2\n"
            ".insn 4, %2 | (_R0 << 7) | (_R1 << 15)\n"
            ".set _R0,z\n.set _R1,z\n"
            ::"r"(rd), "r"(rs1), "i"(encoding)
        );
    }
    _inst_builder_t __custom_32b_instruction_sb_get(long& rd, long rs1)
    {
        asm volatile (
            "\n"
            ".ifc %0,zero\n.set _R0,0\n.endif\n.ifc %0,ra\n.set _R0,1\n.endif\n.ifc %0,sp\n.set _R0,2\n.endif\n.ifc %0,gp\n.set _R0,3\n.endif\n.ifc %0,tp\n.set _R0,4\n.endif\n.ifc %0,t0\n.set _R0,5\n.endif\n.ifc %0,t1\n.set _R0,6\n.endif\n.ifc %0,t2\n.set _R0,7\n.endif\n.ifc %0,s0\n.set _R0,8\n.endif\n.ifc %0,s1\n.set _R0,9\n.endif\n.ifc %0,a0\n.set _R0,10\n.endif\n.ifc %0,a1\n.set _R0,11\n.endif\n.ifc %0,a2\n.set _R0,12\n.endif\n.ifc %0,a3\n.set _R0,13\n.endif\n.ifc %0,a4\n.set _R0,14\n.endif\n.ifc %0,a5\n.set _R0,15\n.endif\n.ifc %0,a6\n.set _R0,16\n.endif\n.ifc %0,a7\n.set _R0,17\n.endif\n.ifc %0,s2\n.set _R0,18\n.endif\n.ifc %0,s3\n.set _R0,19\n.endif\n.ifc %0,s4\n.set _R0,20\n.endif\n.ifc %0,s5\n.set _R0,21\n.endif\n.ifc %0,s6\n.set _R0,22\n.endif\n.ifc %0,s7\n.set _R0,23\n.endif\n.ifc %0,s8\n.set _R0,24\n.endif\n.ifc %0,s9\n.set _R0,25\n.endif\n.ifc %0,s10\n.set _R0,26\n.endif\n.ifc %0,s11\n.set _R0,27\n.endif\n.ifc %0,t3\n.set _R0,28\n.endif\n.ifc %0,t4\n.set _R0,29\n.endif\n.ifc %0,t5\n.set _R0,30\n.endif\n.ifc %0,t6\n.set _R0,31\n.endif\n.ifc %1,zero\n.set _R1,0\n.endif\n.ifc %1,ra\n.set _R1,1\n.endif\n.ifc %1,sp\n.set _R1,2\n.endif\n.ifc %1,gp\n.set _R1,3\n.endif\n.ifc %1,tp\n.set _R1,4\n.endif\n.ifc %1,t0\n.set _R1,5\n.endif\n.ifc %1,t1\n.set _R1,6\n.endif\n.ifc %1,t2\n.set _R1,7\n.endif\n.ifc %1,s0\n.set _R1,8\n.endif\n.ifc %1,s1\n.set _R1,9\n.endif\n.ifc %1,a0\n.set _R1,10\n.endif\n.ifc %1,a1\n.set _R1,11\n.endif\n.ifc %1,a2\n.set _R1,12\n.endif\n.ifc %1,a3\n.set _R1,13\n.endif\n.ifc %1,a4\n.set _R1,14\n.endif\n.ifc %1,a5\n.set _R1,15\n.endif\n.ifc %1,a6\n.set _R1,16\n.endif\n.ifc %1,a7\n.set _R1,17\n.endif\n.ifc %1,s2\n.set _R1,18\n.endif\n.ifc %1,s3\n.set _R1,19\n.endif\n.ifc %1,s4\n.set _R1,20\n.endif\n.ifc %1,s5\n.set _R1,21\n.endif\n.ifc %1,s6\n.set _R1,22\n.endif\n.ifc %1,s7\n.set _R1,23\n.endif\n.ifc %1,s8\n.set _R1,24\n.endif\n.ifc %1,s9\n.set _R1,25\n.endif\n.ifc %1,s10\n.set _R1,26\n.endif\n.ifc %1,s11\n.set _R1,27\n.endif\n.ifc %1,t3\n.set _R1,28\n.endif\n.ifc %1,t4\n.set _R1,29\n.endif\n.ifc %1,t5\n.set _R1,30\n.endif\n.ifc %1,t6\n.set _R1,31\n.endif\n"
            ".align 2\n"
            ".insn 4, %2 | (_R0 << 7) | (_R1 << 15)\n"
            ".set _R0,z\n.set _R1,z\n"
            :"=r"(rd)
            :"r"(rs1), "i"(encoding)
        );
    }
    _inst_builder_t __custom_32b_instruction_tcsr_sb_read(long& rd)
    {
        asm volatile (
           "\n"
           ".ifc %0,zero\n.set _R0,0\n.endif\n.ifc %0,ra\n.set _R0,1\n.endif\n.ifc %0,sp\n.set _R0,2\n.endif\n.ifc %0,gp\n.set _R0,3\n.endif\n.ifc %0,tp\n.set _R0,4\n.endif\n.ifc %0,t0\n.set _R0,5\n.endif\n.ifc %0,t1\n.set _R0,6\n.endif\n.ifc %0,t2\n.set _R0,7\n.endif\n.ifc %0,s0\n.set _R0,8\n.endif\n.ifc %0,s1\n.set _R0,9\n.endif\n.ifc %0,a0\n.set _R0,10\n.endif\n.ifc %0,a1\n.set _R0,11\n.endif\n.ifc %0,a2\n.set _R0,12\n.endif\n.ifc %0,a3\n.set _R0,13\n.endif\n.ifc %0,a4\n.set _R0,14\n.endif\n.ifc %0,a5\n.set _R0,15\n.endif\n.ifc %0,a6\n.set _R0,16\n.endif\n.ifc %0,a7\n.set _R0,17\n.endif\n.ifc %0,s2\n.set _R0,18\n.endif\n.ifc %0,s3\n.set _R0,19\n.endif\n.ifc %0,s4\n.set _R0,20\n.endif\n.ifc %0,s5\n.set _R0,21\n.endif\n.ifc %0,s6\n.set _R0,22\n.endif\n.ifc %0,s7\n.set _R0,23\n.endif\n.ifc %0,s8\n.set _R0,24\n.endif\n.ifc %0,s9\n.set _R0,25\n.endif\n.ifc %0,s10\n.set _R0,26\n.endif\n.ifc %0,s11\n.set _R0,27\n.endif\n.ifc %0,t3\n.set _R0,28\n.endif\n.ifc %0,t4\n.set _R0,29\n.endif\n.ifc %0,t5\n.set _R0,30\n.endif\n.ifc %0,t6\n.set _R0,31\n.endif\n"
            ".align 2\n"
            ".insn 4, %1 | (_R0 << 7)\n"
            ".set _R0,z\n"
            :"=r"(rd)
            :"i"(encoding)
        );
    }
    _inst_builder_t __custom_32b_instruction_tcsr_sb_write(long rd)
    {
        asm volatile (
            "\n"
            ".ifc %0,zero\n.set _R0,0\n.endif\n.ifc %0,ra\n.set _R0,1\n.endif\n.ifc %0,sp\n.set _R0,2\n.endif\n.ifc %0,gp\n.set _R0,3\n.endif\n.ifc %0,tp\n.set _R0,4\n.endif\n.ifc %0,t0\n.set _R0,5\n.endif\n.ifc %0,t1\n.set _R0,6\n.endif\n.ifc %0,t2\n.set _R0,7\n.endif\n.ifc %0,s0\n.set _R0,8\n.endif\n.ifc %0,s1\n.set _R0,9\n.endif\n.ifc %0,a0\n.set _R0,10\n.endif\n.ifc %0,a1\n.set _R0,11\n.endif\n.ifc %0,a2\n.set _R0,12\n.endif\n.ifc %0,a3\n.set _R0,13\n.endif\n.ifc %0,a4\n.set _R0,14\n.endif\n.ifc %0,a5\n.set _R0,15\n.endif\n.ifc %0,a6\n.set _R0,16\n.endif\n.ifc %0,a7\n.set _R0,17\n.endif\n.ifc %0,s2\n.set _R0,18\n.endif\n.ifc %0,s3\n.set _R0,19\n.endif\n.ifc %0,s4\n.set _R0,20\n.endif\n.ifc %0,s5\n.set _R0,21\n.endif\n.ifc %0,s6\n.set _R0,22\n.endif\n.ifc %0,s7\n.set _R0,23\n.endif\n.ifc %0,s8\n.set _R0,24\n.endif\n.ifc %0,s9\n.set _R0,25\n.endif\n.ifc %0,s10\n.set _R0,26\n.endif\n.ifc %0,s11\n.set _R0,27\n.endif\n.ifc %0,t3\n.set _R0,28\n.endif\n.ifc %0,t4\n.set _R0,29\n.endif\n.ifc %0,t5\n.set _R0,30\n.endif\n.ifc %0,t6\n.set _R0,31\n.endif\n"
            ".align 2\n"
            ".insn 4, %1 | (_R0 << 7)\n"
            ".set _R0,z\n"
            ::"r"(rd), "i"(encoding)
        );
    }
    _inst_builder_t __custom_32b_instruction_vec_ls(long rs2)
    {
        asm volatile (
            "\n"
            ".ifc %0,zero\n.set _R0,0\n.endif\n.ifc %0,ra\n.set _R0,1\n.endif\n.ifc %0,sp\n.set _R0,2\n.endif\n.ifc %0,gp\n.set _R0,3\n.endif\n.ifc %0,tp\n.set _R0,4\n.endif\n.ifc %0,t0\n.set _R0,5\n.endif\n.ifc %0,t1\n.set _R0,6\n.endif\n.ifc %0,t2\n.set _R0,7\n.endif\n.ifc %0,s0\n.set _R0,8\n.endif\n.ifc %0,s1\n.set _R0,9\n.endif\n.ifc %0,a0\n.set _R0,10\n.endif\n.ifc %0,a1\n.set _R0,11\n.endif\n.ifc %0,a2\n.set _R0,12\n.endif\n.ifc %0,a3\n.set _R0,13\n.endif\n.ifc %0,a4\n.set _R0,14\n.endif\n.ifc %0,a5\n.set _R0,15\n.endif\n.ifc %0,a6\n.set _R0,16\n.endif\n.ifc %0,a7\n.set _R0,17\n.endif\n.ifc %0,s2\n.set _R0,18\n.endif\n.ifc %0,s3\n.set _R0,19\n.endif\n.ifc %0,s4\n.set _R0,20\n.endif\n.ifc %0,s5\n.set _R0,21\n.endif\n.ifc %0,s6\n.set _R0,22\n.endif\n.ifc %0,s7\n.set _R0,23\n.endif\n.ifc %0,s8\n.set _R0,24\n.endif\n.ifc %0,s9\n.set _R0,25\n.endif\n.ifc %0,s10\n.set _R0,26\n.endif\n.ifc %0,s11\n.set _R0,27\n.endif\n.ifc %0,t3\n.set _R0,28\n.endif\n.ifc %0,t4\n.set _R0,29\n.endif\n.ifc %0,t5\n.set _R0,30\n.endif\n.ifc %0,t6\n.set _R0,31\n.endif\n"
            ".align 2\n"
            ".insn 4, %1 | (_R0 << 20)\n"
            ".set _R0,z\n"
            ::"r"(rs2), "i"(encoding)
        );
    }
    _inst_builder_t __custom_32b_instruction_void(void)
    {
        asm volatile (
            ".align 2\n"
            ".insn 4, %0\n"
            ::"i"(encoding)
        );
    }
}

#undef _inst_builder_t

#include "gen_npu_instruction.h"


#endif
