# See LICENSE for license details.

#include "encoding.h"

#if __riscv_xlen == 64
# define LREG ld
# define SREG sd
# define REGBYTES 8
#else
# define LREG lw
# define SREG sw
# define REGBYTES 4
#endif

  .section ".text.init"
  .globl _start
_start:
  fence.i
  .insn 0xfc200073
  // Set mtvec to _trap_handler
  la a0, _trap_handler
  csrw mtvec, a0

  // Set bootaddr to 0x10000 for reset incomming  
  li a0, 0x1000  // program reset vector
  li a1, 0x0     // busy flag
  sw a1, 0(a0)   // set boot address to bootrom

  // Enable mie interrupts
  li a0, 8               // MIE or MSIP bit
  csrw mie, a0           // set only MSIP in MIE CSR
  csrs mstatus, a0       // set MIE in mstatus CSR

  // li  x1, 0
  // li  x2, 0
  // li  x3, 0
  // li  x4, 0
  // li  x5, 0
  // li  x6, 0
  // li  x7, 0
  // li  x8, 0
  // li  x9, 0
  // li  x10,0
  // li  x11,0
  // li  x12,0
  // li  x13,0
  // li  x14,0
  // li  x15,0
  // li  x16,0
  // li  x17,0
  // li  x18,0
  // li  x19,0
  // li  x20,0
  // li  x21,0
  // li  x22,0
  // li  x23,0
  // li  x24,0
  // li  x25,0
  // li  x26,0
  // li  x27,0
  // li  x28,0
  // li  x29,0
  // li  x30,0
  // li  x31,0

  // enable FPU and accelerator if present
  li t0, MSTATUS_FS | MSTATUS_XS
  csrs mstatus, t0

  // make sure XLEN agrees with compilation choice
  li t0, 1
  slli t0, t0, 31
#if __riscv_xlen == 64
  bgez t0, 1f
#else
  bltz t0, 1f
#endif
2:
  li a0, 1
  sw a0, tohost, t0
  j 2b
1:

#ifdef __riscv_flen
  # initialize FPU if we have one
  // la t0, 1f
  // csrw mtvec, t0
  fssr    x0
  // fmv.s.x f0, x0
  // fmv.s.x f1, x0
  // fmv.s.x f2, x0
  // fmv.s.x f3, x0
  // fmv.s.x f4, x0
  // fmv.s.x f5, x0
  // fmv.s.x f6, x0
  // fmv.s.x f7, x0
  // fmv.s.x f8, x0
  // fmv.s.x f9, x0
  // fmv.s.x f10,x0
  // fmv.s.x f11,x0
  // fmv.s.x f12,x0
  // fmv.s.x f13,x0
  // fmv.s.x f14,x0
  // fmv.s.x f15,x0
  // fmv.s.x f16,x0
  // fmv.s.x f17,x0
  // fmv.s.x f18,x0
  // fmv.s.x f19,x0
  // fmv.s.x f20,x0
  // fmv.s.x f21,x0
  // fmv.s.x f22,x0
  // fmv.s.x f23,x0
  // fmv.s.x f24,x0
  // fmv.s.x f25,x0
  // fmv.s.x f26,x0
  // fmv.s.x f27,x0
  // fmv.s.x f28,x0
  // fmv.s.x f29,x0
  // fmv.s.x f30,x0
  // fmv.s.x f31,x0
1:
#endif

  // DONOT initialize trap vector
  // la t0, trap_entry
  // csrw mtvec, t0

  // initialize global pointer
.option push
.option norelax
  la gp, __global_pointer$
.option pop

  la  tp, _end + 63
  and tp, tp, -64

  // get core id
  csrr a0, mhartid
  // for now, assume only 1 core
  li a1, 1
1:bgeu a0, a1, 1b

  // give each core 128KB of stack + TLS
#define STKSHIFT 17
  add sp, a0, 1
  sll sp, sp, STKSHIFT
  add sp, sp, tp
  sll a2, a0, STKSHIFT
  add tp, tp, a2

  li t0, 0x80020000
  ld t1, 0(t0)
  addi t1, t1, 1
  sd t1, 0(t0)
  sd sp, 16(t0)
  sd tp, 24(t0)
  sd gp, 32(t0)

  j _init

  .section ".text"
  .globl function_call
// a0: kernel_entry, a1: args count, a2: args pointer
function_call:
    // Reserve stack for ra
    addi sp, sp, -192
    sd	ra, 184(sp)
    // t0: kernel_entry, t1: args count, t2: args pointer
    mv	t0, a0
    mv	t1, a1
    mv	t2, a2
    // Prepare stack for arguments > 8
    li	t3, 8
    // li  s0, 0
    ble	t1, t3, 2f

    // If args count > 8, allocate space for additional arguments
    // addi t3, t1, -7 // Calculate additional args count, align to 16bytes
    // slli t3, t3, 3 // Multiply by 8 (size of each argument)
    // andi s0, t3, -16
    // sub sp, sp, s0 // Allocate space on stack
    // Copy arguments to stack
    addi t4, t2, 8*8 // t4: pointer to first argument
    slli t3, t1, 3 // t3: size of arguments in bytes
    add t3, t2, t3 // t3: pointer to last argument
    mv t5, sp // t5: pointer to stack space
1:
    ld t6, 0(t4) // Load argument
    addi t4, t4, 8 // Move to next argument
    sd t6, 0(t5) // Store argument on stack
    addi t5, t5, 8 // Move stack pointer
    ble t4, t3, 1b // Loop until all arguments are copied

    // if we have 0 arguments, we jump to (t3 + 4 * 13)
    // if we have 8 arguments, we jump to (t3 + 4 * 5)
    // so we should jump to (t3 + 4 * (13 - t1))
    // auipc t3, 0 // Current PC
    // addi t4, t1, -13
    // slli t4, t4, 2 // Multiply by 4 (size of each argument) 4 * (t1 - 13)
    // sub t3, t3, t4 // Adjust PC to jump to the right place
    // jr t3
2:
    // load other arguments into registers
    ld a7, 56(t2) // Load 8th argument
    ld a6, 48(t2) // Load 7th argument
    ld a5, 40(t2) // Load 6th argument
    ld a4, 32(t2) // Load 5th argument
    ld a3, 24(t2) // Load 4th argument
    ld a2, 16(t2) // Load 3th argument
    ld a1,  8(t2) // Load 2th argument
    ld a0,  0(t2) // Load 1th argument
    // Call the kernel function
    jalr t0

    // add sp, sp, s0 // restore space on stack

    // Restore ra and stack
    ld	ra, 184(sp)
    addi sp, sp, 192
    ret

.section ".tohost","aw",@progbits
.align 6
.globl tohost
tohost: .dword 0
.align 6
.globl fromhost
fromhost: .dword 0
