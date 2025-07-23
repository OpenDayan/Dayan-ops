#include "trap.h"

  .section ".text.trap"
  .globl _trap_handler
_trap_handler:
  # Alloc trapframe == sp - TF_SIZE
  # Save Basic Information of sp, t0-t2
  sd sp, TF_REG2 - TF_SIZE(sp)
  sd t0, TF_REG5 - TF_SIZE(sp)
  sd t1, TF_REG6 - TF_SIZE(sp)
  sd t2, TF_REG7 - TF_SIZE(sp)

  # Check if interrupt
  csrr t0, mcause
  li t1, 0x8000000000000000
  and t2, t0, t1
  li t1,  0x10000
  csrw mtvec, t1
  beqz t2, 1f
  csrw mepc, t1
  mret

1:
  # Set new Stack Space
  addi sp, sp, -TF_SIZE
  # Store all regs
  csrr t1, mepc
  sd t0, TF_MCAUSE(sp)
  csrr t2, mstatus
  sd t1, TF_MEPC(sp)
  sd t2, TF_MSTATUS(sp)
  sd x0, TF_REG0(sp)
  sd x1, TF_REG1(sp)
  sd x3, TF_REG3(sp)
  sd x4, TF_REG4(sp)
  sd x8, TF_REG8(sp)
  sd x9, TF_REG9(sp)
  sd x10, TF_REG10(sp)
  sd x11, TF_REG11(sp)
  sd x12, TF_REG12(sp)
  sd x13, TF_REG13(sp)
  sd x14, TF_REG14(sp)
  sd x15, TF_REG15(sp)
  sd x16, TF_REG16(sp)
  sd x17, TF_REG17(sp)
  sd x18, TF_REG18(sp)
  sd x19, TF_REG19(sp)
  sd x20, TF_REG20(sp)
  sd x21, TF_REG21(sp)
  sd x22, TF_REG22(sp)
  sd x23, TF_REG23(sp)
  sd x24, TF_REG24(sp)
  sd x25, TF_REG25(sp)
  sd x26, TF_REG26(sp)
  sd x27, TF_REG27(sp)
  sd x28, TF_REG28(sp)
  sd x29, TF_REG29(sp)
  la t0,  trap_handler
  sd x30, TF_REG30(sp)
  csrw mepc, t0
  sd x31, TF_REG31(sp)

  # Jump to trap_handler
  mv a0, sp
  mret
