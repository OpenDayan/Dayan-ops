// Auto-generated instruction assembler
// Generated from JSON instruction definitions

#pragma once

#include <cstdint>

// Instruction implementations

template <long tcsr_id>
__attribute__((always_inline)) static inline void tx_mat_tcsr_write(long rd) {
    constexpr long encoding = (0b00000000000000000000000000001011) | ((tcsr_id & 1023) << 15);
    BaseNPUInstBuilder::__custom_32b_instruction_tcsr_sb_write<encoding>(rd);
}

template <long tcsr_id>
__attribute__((always_inline)) static inline void tx_mat_tcsr_read(long &rd) {
    constexpr long encoding = (0b00000000000000000100000000001011) | ((tcsr_id & 1023) << 15);
    BaseNPUInstBuilder::__custom_32b_instruction_tcsr_sb_read<encoding>(rd);
}

template <long tcsr_id>
__attribute__((always_inline)) static inline void tx_vec_tcsr_write(long rd) {
    constexpr long encoding = (0b00000010000000000000000000001011) | ((tcsr_id & 1023) << 15);
    BaseNPUInstBuilder::__custom_32b_instruction_tcsr_sb_write<encoding>(rd);
}

template <long tcsr_id>
__attribute__((always_inline)) static inline void tx_vec_tcsr_read(long &rd) {
    constexpr long encoding = (0b00000010000000000100000000001011) | ((tcsr_id & 1023) << 15);
    BaseNPUInstBuilder::__custom_32b_instruction_tcsr_sb_read<encoding>(rd);
}

template <long tcsr_id>
__attribute__((always_inline)) static inline void tx_lsu_tcsr_write(long rd) {
    constexpr long encoding = (0b00000100000000000000000000001011) | ((tcsr_id & 1023) << 15);
    BaseNPUInstBuilder::__custom_32b_instruction_tcsr_sb_write<encoding>(rd);
}

template <long tcsr_id>
__attribute__((always_inline)) static inline void tx_lsu_tcsr_read(long &rd) {
    constexpr long encoding = (0b00000100000000000100000000001011) | ((tcsr_id & 1023) << 15);
    BaseNPUInstBuilder::__custom_32b_instruction_tcsr_sb_read<encoding>(rd);
}

__attribute__((always_inline)) static inline void tx_mat_sb_produce(long rd) {
    constexpr long encoding = 0b00001000000000000000000000001011;
    BaseNPUInstBuilder::__custom_32b_instruction_tcsr_sb_write<encoding>(rd);
}

__attribute__((always_inline)) static inline void tx_mat_sb_bar(long &rd) {
    constexpr long encoding = 0b00001000000000000101000000001011;
    BaseNPUInstBuilder::__custom_32b_instruction_tcsr_sb_read<encoding>(rd);
}

__attribute__((always_inline)) static inline void tx_vec_sb_produce(long rd) {
    constexpr long encoding = 0b00001010000000000000000000001011;
    BaseNPUInstBuilder::__custom_32b_instruction_tcsr_sb_write<encoding>(rd);
}

__attribute__((always_inline)) static inline void tx_vec_sb_bar(long &rd) {
    constexpr long encoding = 0b00001010000000000101000000001011;
    BaseNPUInstBuilder::__custom_32b_instruction_tcsr_sb_read<encoding>(rd);
}

__attribute__((always_inline)) static inline void tx_lsu_sb_produce(long rd) {
    constexpr long encoding = 0b00001100000000000000000000001011;
    BaseNPUInstBuilder::__custom_32b_instruction_tcsr_sb_write<encoding>(rd);
}

__attribute__((always_inline)) static inline void tx_lsu_sb_bar(long &rd) {
    constexpr long encoding = 0b00001100000000000101000000001011;
    BaseNPUInstBuilder::__custom_32b_instruction_tcsr_sb_read<encoding>(rd);
}

__attribute__((always_inline)) static inline void tx_sb_get_any(long &rd) {
    constexpr long encoding = 0b00001110000000000100000000001011;
    BaseNPUInstBuilder::__custom_32b_instruction_tcsr_sb_read<encoding>(rd);
}

template <long vb, long sel>
__attribute__((always_inline)) static inline void tx_vec_v_load(long rs2) {
    constexpr long encoding =
        (0b01000000000000000000000000001011) | ((vb & 1) << 25) | ((sel & 3) << 9);
    BaseNPUInstBuilder::__custom_32b_instruction_vec_ls<encoding>(rs2);
}

template <long vb, long sel>
__attribute__((always_inline)) static inline void tx_vec_s_load(long rs2) {
    constexpr long encoding =
        (0b01000000000000001000000000001011) | ((vb & 1) << 25) | ((sel & 3) << 9);
    BaseNPUInstBuilder::__custom_32b_instruction_vec_ls<encoding>(rs2);
}

template <long vb> __attribute__((always_inline)) static inline void tx_vec_const_load(long rs2) {
    constexpr long encoding = (0b01000000000000001000000000001011) | ((vb & 1) << 25);
    BaseNPUInstBuilder::__custom_32b_instruction_vec_ls<encoding>(rs2);
}

template <long vrs2, long sel>
__attribute__((always_inline)) static inline void tx_vec_v_store(long rs2) {
    constexpr long encoding =
        (0b01000100000000000000000000001011) | ((vrs2 & 31) << 15) | ((sel & 3) << 9);
    BaseNPUInstBuilder::__custom_32b_instruction_vec_ls<encoding>(rs2);
}

template <long vrs2, long sel>
__attribute__((always_inline)) static inline void tx_vec_s_store(long rs2) {
    constexpr long encoding =
        (0b01000100000000000000000100001011) | ((vrs2 & 31) << 15) | ((sel & 3) << 9);
    BaseNPUInstBuilder::__custom_32b_instruction_vec_ls<encoding>(rs2);
}

template <long vb, long vrs1, long vrs2>
__attribute__((always_inline)) static inline void tx_vec_vv_int32_x() {
    constexpr long encoding = (0b10000100000000000000000000001011) | ((vb & 1) << 25) |
                              ((vrs1 & 31) << 15) | ((vrs2 & 31) << 7);
    BaseNPUInstBuilder::__custom_32b_instruction_void<encoding>();
}

template <long vb, long vrs1, long vrs2, long lut>
__attribute__((always_inline)) static inline void tx_vec_vv_int32_bitwise() {
    constexpr long encoding = (0b10000100001000000000000000001011) | ((vb & 1) << 25) |
                              ((vrs1 & 31) << 15) | ((vrs2 & 31) << 7) | ((lut & 7) << 12) |
                              ((lut & 1) << 20);
    BaseNPUInstBuilder::__custom_32b_instruction_void<encoding>();
}

template <long group>
__attribute__((always_inline)) static inline void tx_lsu_dma_u2u(long rd, long rs1) {
    constexpr long encoding = (0b00100000000000000000000000001011) | ((group & 7) << 20);
    BaseNPUInstBuilder::__custom_32b_instruction_lsu<encoding>(rd, rs1);
}

template <long group>
__attribute__((always_inline)) static inline void tx_lsu_dma_u2d(long rd, long rs1) {
    constexpr long encoding = (0b00100000000000000001000000001011) | ((group & 7) << 20);
    BaseNPUInstBuilder::__custom_32b_instruction_lsu<encoding>(rd, rs1);
}

template <long group>
__attribute__((always_inline)) static inline void tx_lsu_dma_d2u(long rd, long rs1) {
    constexpr long encoding = (0b00100000000000000010000000001011) | ((group & 7) << 20);
    BaseNPUInstBuilder::__custom_32b_instruction_lsu<encoding>(rd, rs1);
}

template <long group>
__attribute__((always_inline)) static inline void tx_lsu_dma_d2d(long rd, long rs1) {
    constexpr long encoding = (0b00100000000000000011000000001011) | ((group & 7) << 20);
    BaseNPUInstBuilder::__custom_32b_instruction_lsu<encoding>(rd, rs1);
}

template <long group>
__attribute__((always_inline)) static inline void tx_mat_mma(long rd, long rs1, long rs2) {
    constexpr long encoding = (0b00010000000000000000000000001011) | ((group & 7) << 25);
    BaseNPUInstBuilder::__custom_32b_instruction_mat<encoding>(rd, rs1, rs2);
}

template <long group>
__attribute__((always_inline)) static inline void tx_mat_mm(long rd, long rs1, long rs2) {
    constexpr long encoding = (0b00010000000000000001000000001011) | ((group & 7) << 25);
    BaseNPUInstBuilder::__custom_32b_instruction_mat<encoding>(rd, rs1, rs2);
}
