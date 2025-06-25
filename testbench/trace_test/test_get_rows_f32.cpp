#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <cassert>

#include "common.hpp"

void get_rows_bf16(std::vector<uint16_t>& a, std::vector<int>& b, std::vector<float>& res,
                   TensorParams& psrc0, TensorParams& psrc1, TensorParams& pdst) {
    // source tensor
    int bs0 = psrc0.ne[3];
    int h0 = psrc0.ne[2];
    int n0 = psrc0.ne[1];
    int m0 = psrc0.ne[0];

    // index tensor
    int bs1 = psrc1.ne[3];
    int h1 = psrc1.ne[2];
    int n1 = psrc1.ne[1];
    int m1 = psrc1.ne[0];

    assert((bs0 == h1) && (h0 == n1));
    assert(bs1 == 1);

    // sr0 shape: (h1, n1, x, m0)  (x is the number range of src0[0][j][k][p])
    // sr1 shape: (1, h1, n1, m1)
    // dst shape: (h1, n1, m1, m0)

    for (int j = 0; j < h1; ++j)
        for (int k = 0; k < n1; ++k)
            for (int p = 0; p < m1; ++p) {
                // row_idx = b[0][j][k][p]
                int row_idx = b[j*n1*m1 + k*m1 + p];
                if (row_idx < 0 || row_idx > n0) {
                    std::cerr << "[ERROR] invalid row_idx value: " << row_idx << std::endl;
                    return;
                }
                // res[j][k][p][:] = a[j][k][row_idx][:]
                for (int l = 0; l < m0; l++) {
                    uint32_t val_u32 = (uint32_t)a[j*h0*n0*m0 + k*n0*m0 + row_idx*m0 + l];
                    val_u32 <<= 16;
                    float val_f32;
                    std::memcpy(&val_f32, &val_u32, sizeof(float));
                    res[j*h1*n1*m0 + k*n1*m1 + p*m0 + l] = val_f32;
                }

            }
}

void get_rows_f32(std::vector<float>& a, std::vector<int>& b, std::vector<float>& res,
                   TensorParams& psrc0, TensorParams& psrc1, TensorParams& pdst) {
        // source tensor
    int bs0 = psrc0.ne[3];
    int h0 = psrc0.ne[2];
    int n0 = psrc0.ne[1];
    int m0 = psrc0.ne[0];

    // index tensor
    int bs1 = psrc1.ne[3];
    int h1 = psrc1.ne[2];
    int n1 = psrc1.ne[1];
    int m1 = psrc1.ne[0];

    assert((bs0 == h1) && (h0 == n1));
    assert(bs1 == 1);

    // assert ((bs0 == h1) && (h0 == n1))

    // sr0 shape: (h1, n1, x, m0)  (x is the number range of src0[0][j][k][p])
    // sr1 shape: (1, h1, n1, m1)
    // dst shape: (h1, n1, m1, m0)

    for (int j = 0; j < h1; ++j)
        for (int k = 0; k < n1; ++k)
            for (int p = 0; p < m1; ++p) {
                // row_idx = b[0][j][k][p]
                int row_idx = b[j*n1*m1 + k*m1 + p];
                if (row_idx < 0 || row_idx > n0) {
                    std::cerr << "[ERROR] invalid row_idx value: " << row_idx << std::endl;
                    return;
                }
                // res[j][k][p][:] = a[j][k][row_idx][:]
                for (int l = 0; l < m0; l++) {
                    res[j*h1*n1*m0 + k*n1*m1 + p*m0 + l] = a[j*h0*n0*m0 + k*n0*m0 + row_idx*m0 + l];
                }

            }
}

int test_get_rows(std::string src0_trace, std::string src1_trace, std::string dst_trace,
                  TensorParams& psrc0, TensorParams& psrc1, TensorParams& pdst) {
    try {

        std::vector<int> data_b = readInt32FromFile(src1_trace);
        std::vector<float> data_ref = readF32FromFile(dst_trace);
        std::vector<float> result_data;

        if (psrc0.type == "bf16") {
            std::vector<uint16_t> data_a = readUInt16FromFile(src0_trace);
            if (data_a.empty()) {
                std::cout << "[WARNING] src0 data is empty!" << std::endl;
                return 0;
            }

            result_data.resize(data_a.size());

            // FIXME: RUN KERNEL HERE
            //
            // --------------------------------------------
            //
            // FIXME: CPU reference
            get_rows_bf16(data_a, data_b, result_data, psrc0, psrc1, pdst);

        } else if (psrc0.type == "f32") {
            std::vector<float> data_a = readF32FromFile(src0_trace);
            if (data_a.empty()) {
                std::cout << "[WARNING] src0 data is empty!" << std::endl;
                return 0;
            }

            result_data.resize(data_a.size());

            // FIXME: RUN KERNEL HERE
            //
            // --------------------------------------------
            //
            // FIXME: CPU reference
            get_rows_f32(data_a, data_b, result_data, psrc0, psrc1, pdst);
        } else {
            std::cerr << "[ERROR] psrc0.dtype=" << psrc0.type << " is not supported\n" << std::endl;
            return 1;
        }

        // std::cout << "\nCompare with ref..." << std::endl;
        const float epsilon = 1e-6f;
        bool all_match = true;

        int bs = pdst.ne[3];
        int h = pdst.ne[2];
        int n = pdst.ne[1];
        int m = pdst.ne[0];
        for (int i = 0; i < bs; ++i) {
            for (int j = 0; j < h; ++j) {
                for (int k = 0; k < n; ++k) {
                    for (int p = 0; p < m; ++p) {
                        float result = result_data[i*h*n*m + j*n*m + k*m + p];
                        float ref = data_ref[i*h*n*m + j*n*m + k*m + p];
                        if (std::abs(result - ref) > epsilon) {
                            std::cerr << std::fixed << std::setprecision(8);
                            std::cerr << "\n !!! Compare FAILED !!!" << std::endl;
                            std::cerr << "  - result[" << i << "][" << j << "][" << k << "][" << p << "]:    "
                                    << result << std::endl;
                            std::cerr << "  - ref[" << i << "][" << j << "][" << k << "][" << p << "]:    "
                                    << ref << std::endl;
                            std::cerr << "  - result - ref:        " << std::abs(result - ref) << std::endl;
                            all_match = false;
                            goto Done;
                        }
                    }
                }
            }
        }
        Done:
        if (all_match) {
            std::cout << " *** Compare PASS ***" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;

}


int main(int argc, char * argv[]) {
    TraceParamsParser parser;
    TensorParams psrc0, psrc1, pdst;
    std::cout << ">>>>>>>>>>>>> Start Get_Rows test:" << std::endl;
    for(int i = 0; i < 6; ++i) {
        std::string a = "trace/get_rows/" + std::to_string(i);
        std::string dst_trace = "trace/get_rows/" + std::to_string(i) + "-f32-dst.trace";
        std::string src0_trace = "trace/get_rows/" + std::to_string(i);
        if (i % 3 == 0) { // embeddings
            a += "-bf16.txt";
            src0_trace += "-bf16-src0.trace";
        } else { // last layer
            a += "-f32.txt";
            src0_trace += "-f32-src0.trace";
        }
        std::string src1_trace = "trace/get_rows/" + std::to_string(i) + "-i32-src1.trace";
        parser.parse(a, &pdst, &psrc0, &psrc1, NULL, NULL, NULL);
        if (!pdst.is_contiguous || !psrc0.is_contiguous || !psrc1.is_contiguous) {
            std::cerr << "[ERROR] get_rows tests only support contiguous tensor now!" << std::endl;
            return 1;
        }
        std::cout << "Testing Get_Rows with trace." << i << " ------------------ ";
        test_get_rows(src0_trace, src1_trace, dst_trace, psrc0, psrc1, pdst);
    }
    std::cout << "<<<<<<<<<<<<< Get_Rows test end" << std::endl;
}