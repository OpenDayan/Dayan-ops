#include "common.hpp"
#include <iostream>
#include <string>
#include <cstring>

void permute_f32_2_1(std::vector<float>& a, std::vector<float>& res, TensorParams& pa, TensorParams& pres, PermuteParams& ppermute) {
    int bs = pa.ne[3];
    int n = pa.ne[2];
    int h = pa.ne[1];

    for (int i = 0; i < bs; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < h; ++k) {
                // Copy a[i][j][k][:] to res[i][k][j][:]
                const char * a_ptr = (char *)a.data() + k*pa.nb[1] + j*pa.nb[2] + i*pa.nb[3];
                size_t id = k*n*h + j*h;
                std::memcpy((char *)res.data() + id, a_ptr, h);
            }
        }
    }
}

int test_permute_f32(std::string src0_trace, std::string dst_trace, TensorParams& psrc0, TensorParams& pdst,
                     PermuteParams& ppermute) {
    try {
        std::vector<float> data_a = readF32FromFile(src0_trace);
        std::vector<float> data_ref = readF32FromFile(dst_trace);
        size_t len_src0 = psrc0.ne[0] * psrc0.ne[1] * psrc0.ne[2] * psrc0.ne[3];

        if (data_a.empty()) {
            std::cout << "[WARNING] src0 data is empty!" << std::endl;
            return 0;
        }

        std::vector<float> result_data;
        result_data.resize(len_src0);

        // FIXME: RUN KERNEL HERE
        //
        // --------------------------------------------
        // kernel_permute_f32_2_1data_a, result_data, ...);
        // --------------------------------------------
        // FIXME: CPU reference
        permute_f32_2_1(data_a, result_data, psrc0, pdst, ppermute);

    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}


int main(int argc, char * argv[]) {
    if(argc < 3) {
        std::cout << "Usage: ./test <TRACE DIR> <TRACE ID>" << std::endl;
        return 0;
    }

    std::string trace_dir(argv[1]);
    int trace_id = std::stoi(argv[2]);

    TraceParamsParser parser;
    TensorParams psrc0, pdst;
    PermuteParams ppermute;
    std::string a = trace_dir + "/" + std::to_string(trace_id) + "-params.txt";
    std::string dst_trace = trace_dir + "/" + std::to_string(trace_id) + "-dst.trace";
    std::string src0_trace = trace_dir + "/" + std::to_string(trace_id) + "-src0.trace";
    parser.parse(a, &pdst, &psrc0, NULL, NULL, NULL, NULL, &ppermute);
    if (psrc0.type != "f32" || !psrc0.is_contiguous) {
        std::cerr << "[ERROR] Permute tests only support permute from contiguous FP32" << std::endl;
        return 1;
    }
    if (ppermute.axis0 != 0 || ppermute.axis3 != 3 || ppermute.axis1 == 1 || ppermute.axis2 == 2) {
        std::cerr << "[ERROR] Permute tests only support (b, n, h, w) -> (b, h, n, w) permute" << std::endl;
        return 1;
    }
    std::cout << "Testing Permute/Transpose with trace." << trace_id << " ------------------ ";

    // TODO: is_transpose
    return test_permute_f32(src0_trace, dst_trace, psrc0, pdst, ppermute);
}