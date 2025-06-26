#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <cstddef>
#include <stddef.h>
#include <stdint.h>


void configTCRWithMatrixDims(int M, int K, int N) {

  /* config M src matrix: dim0
  Instruction format(tx.tcsr.xchg.nmask):
    63 56	    41 32	   31  27       19 15	 11 7     6 0
    10000000 	 tcsr_id   rs3      	 rs1     rd      0111111

  Example: (tcsr_id = 0/16,  rd = 0, rs1 = 5, rs3 = 0)
    10000000
    00000000
    00000000
    00000000
    00000000
    00000010
    10000000
    00111111
  */

  __asm__ volatile("mv x5, %0\n\t" : : "r"(M) : "x5");
  asm volatile(".byte 0x3f, 0x80, 0x02, 0x00, 0x00, 0x00, 0x00, 0x80");
  asm volatile(".byte 0x3f, 0x80, 0x02, 0x00, 0x10, 0x00, 0x00, 0x80");

  /*Example: (tcsr_id = 2/8,  rd = 0, rs1 = 5, rs3 = 0)
    10000000
    00000000
    00000000
    00000010
    00000000
    00000010
    10000000
    00111111
  */
  __asm__ volatile("mv x5, %0\n\t" : : "r"(K) : "x5");
  asm volatile(".byte 0x3f, 0x80, 0x02, 0x00, 0x02, 0x00, 0x00, 0x80");
  asm volatile(".byte 0x3f, 0x80, 0x02, 0x00, 0x08, 0x00, 0x00, 0x80");

  /*Example: (tcsr_id = 10/18,  rd = 0, rs1 = 5, rs3 = 0)
    10000000
    00000000
    00000000
    00001010
    00000000
    00000010
    10000000
    00111111
  */
  __asm__ volatile("mv x5, %0\n\t" : : "r"(N) : "x5");
  asm volatile(".byte 0x3f, 0x80, 0x02, 0x00, 0x0a, 0x00, 0x00, 0x80");
  asm volatile(".byte 0x3f, 0x80, 0x02, 0x00, 0x12, 0x00, 0x00, 0x80");

}

__device__ void matmul_fp32_fp32_fp32(void *input0, void *input1, void *dst, int m, int n,
                                        int k) {
  configTCRWithMatrixDims(m, k, n);
   /*config addr */
  __asm__ volatile("mv x5, %0\n\t" : : "r"(input0) : "x5");
  __asm__ volatile("mv x6, %0\n\t" : : "r"(input1) : "x6");
  __asm__ volatile("mv x7, %0\n\t" : : "r"(dst) : "x7");

  /*
  Instruction format(tx.matmul.subfloat.<rs1type>.<rs2type>.<rs3type>):
  63 56	  55 50     49 44	     43    40 38	 37 35    34 32 31 27  24 20  19 15    6 0
  00000001 rs1type	rs2type rs3type tcfgrs3  tcfgrs2  tcfgrs1	rs3    rs2    rs1   0111111

  Example: (rs1 = 5,  rs2 = 6, rs3 = 7)
  00000001
  00010000
  01000000
  10001000
  00111000
  01100010
  10000000
  00111111
  */
  asm volatile(".byte 0x3f, 0x80, 0x62, 0x38, 0x88, 0x40, 0x10, 0x01");
}
