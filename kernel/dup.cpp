#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <stdio.h>
#include <string.h>

const long aTCSR = 0;
const long bTCSR = 1;
const long cTCSR = 2;

const uint64_t uram0 = 0x1000000;
const uint64_t uram1 = 0x2000000;
const uint64_t uram2 = 0x3000000;

#define aTile 1024
const uint64_t uram_a = 0x0;
const uint64_t uram_res = 0x20000; //128KB

static inline void configTCSRa_cont(int a_bs, int a_h, int a_n, int a_m){
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM0>(a_m);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM1>(a_n);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE1>(a_m);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM2>(a_h);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE2>(a_m * a_n);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM3>(a_bs);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE3>(a_m * a_n * a_h);
}

static inline void configTCSRc_cont(int res_bs, int res_h, int res_n, int res_m){
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM0>(res_m);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM1>(res_n);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE1>(res_m);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM2>(res_h);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE2>(res_m * res_n);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM3>(res_bs);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE3>(res_m * res_n * res_h);    
}

static inline void configTCSRa_no_cont(int a_bs, int a_h, int a_n, int a_m,
                                    int a_nb3, int a_nb2, int a_nb1, int a_nb0){
    // 这个定义与test中不同，需要在外部除以元素长度如sizeof(float)
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM0>(a_m);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE0>(a_nb0);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM1>(a_n);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE1>(a_nb1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM2>(a_h);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE2>(a_nb2);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM3>(a_bs);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE3>(a_nb3);
}
static inline void configTCSRc_no_cont(int res_bs, int res_h, int res_n, int res_m,
                                    int res_nb3, int res_nb2, int res_nb1, int res_nb0){
    // 这个定义与test中不同，需要在外部除以元素长度如sizeof(uint16_t)                                        
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM0>(res_m);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE0>(res_nb0);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM1>(res_n);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE1>(res_nb1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM2>(res_h);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE2>(res_nb2);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM3>(res_bs);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE3>(res_nb3);
}

static inline void configTCSR_Tile_cont(int elem_per_tile, int last_elem_tile_num){
    // a存每组长度，b存最后一组长度
    // printf("configTCSR_Tile_cont elem_per_tile = %d, last_elem_tile_num = %d\n", elem_per_tile, last_elem_tile_num);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM0>(elem_per_tile);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM1>(1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE1>(elem_per_tile);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE2>(elem_per_tile);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM3>(1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE3>(elem_per_tile);

    __tx_tcsr_write<bTCSR * 8 + TCSR_DIM0>(last_elem_tile_num);
    __tx_tcsr_write<bTCSR * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<bTCSR * 8 + TCSR_DIM1>(1);
    __tx_tcsr_write<bTCSR * 8 + TCSR_STRIDE1>(last_elem_tile_num);
    __tx_tcsr_write<bTCSR * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<bTCSR * 8 + TCSR_STRIDE2>(last_elem_tile_num);
    __tx_tcsr_write<bTCSR * 8 + TCSR_DIM3>(1);
    __tx_tcsr_write<bTCSR * 8 + TCSR_STRIDE3>(last_elem_tile_num);

    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM0>(elem_per_tile);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM1>(1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE1>(elem_per_tile);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE2>(elem_per_tile);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM3>(1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE3>(elem_per_tile);
}
__device__ void dup_cont_f32_cont_f32(void *a, void *res,
                                    int a_bs, int a_h, int a_n, int a_m,
                                    int res_bs, int res_h, int res_n, int res_m){
    // 要求 (a_bs*a_h*a_n*a_m) == (res_bs*res_h*res_n*res_m)
    // 分块处理
    // 假设连续时(bs, h, n, m)和(1, 1, 1, bs*h*n*m)等价
    int total_elem_num = a_bs*a_h*a_n*a_m;
    int elem_per_tile = (total_elem_num > aTile) ? aTile : total_elem_num;
    int last_elem_tile_num = (total_elem_num % elem_per_tile == 0) ? elem_per_tile : (total_elem_num % elem_per_tile);
    configTCSR_Tile_cont(elem_per_tile, last_elem_tile_num);
    int num_elem_copied;
    for(num_elem_copied = 0; num_elem_copied + elem_per_tile <= total_elem_num; num_elem_copied += elem_per_tile){
        void *a_ptr = (char*)a + num_elem_copied * sizeof(float);
        void *res_ptr = (char*)res + num_elem_copied * sizeof(float);
        __tx_async_uram_load<aTCSR, aTCSR, 2>(a_ptr, (void*)uram_a);
        __tx_fence();
        __tx_async_uram_store<aTCSR, aTCSR, 2>((void*)uram_a, res_ptr);
        __tx_fence();
    }
    if(total_elem_num - num_elem_copied > 0){
        void *a_ptr = (char*)a + num_elem_copied * sizeof(float);
        void *res_ptr = (char*)res + num_elem_copied * sizeof(float);
        __tx_async_uram_load<bTCSR, bTCSR, 2>(a_ptr, (void*)uram_a);
        __tx_fence();
        __tx_async_uram_store<bTCSR, bTCSR, 2>((void*)uram_a, res_ptr);
        __tx_fence();
    }
}

__device__ void dup_cont_f32_cont_bf16(void *a, void *res,
                                    int a_bs, int a_h, int a_n, int a_m,
                                    int res_bs, int res_h, int res_n, int res_m){
    // 要求 (a_bs*a_h*a_n*a_m) == (res_bs*res_h*res_n*res_m)

    // 一次性处理
    // configTCSRa_cont(a_bs, a_h, a_n, a_m);
    // configTCSRc_cont(res_bs, res_h, res_n, res_m);
    // __tx_async_uram_load<aTCSR, aTCSR, 2>(a, (void*)uram0);
    // __tx_fence();
    // __tx_convert_fp32_bf16_round<aTCSR, aTCSR>((void *)uram0, (void *)uram1); // 连续和不连续的不一样
    // __tx_fence();
    // __tx_async_uram_store<cTCSR, cTCSR, 1>((void*)uram1, res);
    // __tx_fence();

    // 分块处理，LCS软流水
    int total_elem_num = a_bs*a_h*a_n*a_m;
    int elem_per_tile = (total_elem_num > aTile) ? aTile : total_elem_num;
    int total_op_num = (total_elem_num + elem_per_tile - 1) / elem_per_tile;
    int last_elem_tile_num = (total_elem_num % elem_per_tile == 0) ? elem_per_tile : (total_elem_num % elem_per_tile);
    configTCSR_Tile_cont(elem_per_tile, last_elem_tile_num);
    // L(1)
    __tx_async_uram_load<aTCSR, aTCSR, 2>(a, (void*)uram_a);
    __tx_fence();
    // C(1)
    __tx_convert_fp32_bf16_round<aTCSR, aTCSR>((void *)uram_a, (void *)uram_res);
    if(total_op_num > 1){
        if(total_op_num > 2){
            // L(2)
            __tx_async_uram_load<aTCSR, aTCSR, 2>((void*)((char*)a + elem_per_tile * sizeof(float)), (void*)(uram_a + elem_per_tile * sizeof(float)));
            for(int i = 2; i < total_op_num - 1; i++){
                // store(i-2), convert(i-1), load(i), i = 2,...,total_op_num - 2
                // S(1), C(2), L(3), ... , S(N-3), C(N-2), L(N-1)
                __tx_fence();
                __tx_async_uram_store<aTCSR, aTCSR, 1>((void*)(uram_res + (i % 2) * elem_per_tile * sizeof(float)), 
                                                (void*)((char*)res + (i - 2) * elem_per_tile * sizeof(uint16_t)));
                __tx_convert_fp32_bf16_round<aTCSR, aTCSR>((void *)(uram_a + ((i - 1) % 2) * elem_per_tile * sizeof(float)), 
                                                (void *)(uram_res + ((i - 1) % 2) * elem_per_tile * sizeof(float)));
                __tx_async_uram_load<aTCSR, aTCSR, 2>((void*)((char*)a + i * elem_per_tile * sizeof(float)), 
                                                (void*)(uram_a + (i % 2) * elem_per_tile * sizeof(float)));
            }
            // 最后一次的长度不一定一样，单独load
            __tx_fence();
            // S(N-2)
            __tx_async_uram_store<aTCSR, aTCSR, 1>((void*)uram_res + ((total_op_num - 1) % 2) * elem_per_tile * sizeof(float), 
                                            (void*)((char*)res + (total_op_num - 3) * elem_per_tile * sizeof(uint16_t)));// 这个-3不行吧
            // C(N-1)     
            __tx_convert_fp32_bf16_round<aTCSR, aTCSR>((void *)uram_a + ((total_op_num) % 2) * elem_per_tile * sizeof(float), 
                                            (void *)uram_res + ((total_op_num) % 2) * elem_per_tile * sizeof(float)); 
        }     
        // L(N)               
        __tx_async_uram_load<bTCSR, bTCSR, 2>((void*)((char*)a + (total_op_num - 1) * elem_per_tile * sizeof(float)), 
                                        (void*)uram_a + ((total_op_num - 1) % 2) * elem_per_tile * sizeof(float));
        __tx_fence();
        // S(N-1)
        __tx_async_uram_store<aTCSR, aTCSR, 1>((void*)uram_res + (total_op_num % 2) * elem_per_tile * sizeof(float), 
                                        (void*)((char*)res + (total_op_num - 2) * elem_per_tile * sizeof(uint16_t)));
        // C(N)
        __tx_convert_fp32_bf16_round<bTCSR, bTCSR>((void *)uram_a + ((total_op_num - 1) % 2) * elem_per_tile * sizeof(float), 
                                        (void *)uram_res + ((total_op_num - 1) % 2) * elem_per_tile * sizeof(float));                                     
        __tx_fence();
    }
    // S(N)
    __tx_async_uram_store<bTCSR, bTCSR, 1>((void*)uram_res + ((total_op_num - 1) % 2) * elem_per_tile * sizeof(float), 
                                        (void*)((char*)res + (total_op_num - 1) * elem_per_tile * sizeof(uint16_t)));
    __tx_fence();
}

__device__ void dup_cont_f32_no_cont_bf16(void *a, void *res,
                                    int a_bs, int a_h, int a_n, int a_m,
                                    int res_bs, int res_h, int res_n, int res_m,
                                    int a_nb3, int a_nb2, int a_nb1, int a_nb0, 
                                    int res_nb3, int res_nb2, int res_nb1, int res_nb0){
    // 要求 (a_bs*a_h*a_n*a_m) == (res_bs*res_h*res_n*res_m)
    configTCSRa_cont(a_bs, a_h, a_n, a_m);
    configTCSRc_no_cont(res_bs, res_h, res_n, res_m, 
                (res_nb3/sizeof(uint16_t)), (res_nb2/sizeof(uint16_t)), (res_nb1/sizeof(uint16_t)), (res_nb0/sizeof(uint16_t)));
    __tx_async_uram_load<aTCSR, aTCSR, 2>(a, (void*)uram0);
    __tx_fence();
    __tx_convert_fp32_bf16_round<aTCSR, cTCSR>((void *)uram0, (void *)uram1); // 连续和不连续的不一样
    __tx_fence();
    __tx_async_uram_store<cTCSR, cTCSR, 1>((void*)uram1, res);
    __tx_fence();
}

__device__ void dup_no_cont_f32_no_cont_bf16(void *a, void *res,
                                    int a_bs, int a_h, int a_n, int a_m,
                                    int res_bs, int res_h, int res_n, int res_m,
                                    int a_nb3, int a_nb2, int a_nb1, int a_nb0, 
                                    int res_nb3, int res_nb2, int res_nb1, int res_nb0){
    // 要求 (a_bs*a_h*a_n*a_m) == (res_bs*res_h*res_n*res_m)
    configTCSRa_no_cont(a_bs, a_h, a_n, a_m, 
                (a_nb3/sizeof(float)), (a_nb2/sizeof(float)), (a_nb1/sizeof(float)), (a_nb0/sizeof(float)));
    configTCSRc_no_cont(res_bs, res_h, res_n, res_m, 
                (res_nb3/sizeof(uint16_t)), (res_nb2/sizeof(uint16_t)), (res_nb1/sizeof(uint16_t)), (res_nb0/sizeof(uint16_t)));
    __tx_async_uram_load<aTCSR, aTCSR, 2>(a, (void*)uram0);
    __tx_fence();
    __tx_convert_fp32_bf16_round<aTCSR, cTCSR>((void *)uram0, (void *)uram1); // 连续和不连续的不一样
    __tx_fence();
    __tx_async_uram_store<cTCSR, cTCSR, 1>((void*)uram1, res);
    __tx_fence();
}