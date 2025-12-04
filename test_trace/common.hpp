#ifndef _COMMON_H_
#define _COMMON_H_

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

std::vector<float> readF32FromFile(const std::string &filename);
std::vector<uint8_t> readUInt8FromFile(const std::string &filename);
std::vector<uint16_t> readUInt16FromFile(const std::string &filename);
std::vector<int> readInt32FromFile(const std::string &filename);

enum mrope { NORMAL, NEOX, VISION };

struct RopeParams {
    int n_dims;
    int mode;
    int n_ctx_orig;
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    std::vector<int> sections;
};

struct NormParams {
    float eps;
};

struct SoftmaxParams {
    float scale;
    float max_bias;
};

struct PermuteParams {
    int axis0;
    int axis1;
    int axis2;
    int axis3;
};

struct TensorParams {
    std::vector<int> ne;
    std::vector<int> nb;
    bool is_contiguous;
    std::string type;
    uint64_t len;
};

class TraceParamsParser {
  public:
    void parse(const std::string &filename, TensorParams *pdst, TensorParams *psrc0,
               TensorParams *psrc1, NormParams *prms, RopeParams *prope, SoftmaxParams *psoft,
               PermuteParams *ppermute);

  private:
    void parse_tensor_kv(TensorParams *tensor, const std::string &key, const std::string &value);
    void parse_rope_kv(RopeParams *rope, const std::string &key, const std::string &value);
    void parse_norm_kv(NormParams *rms, const std::string &key, const std::string &value);
    void parse_softmax_kv(SoftmaxParams *softmax, const std::string &key, const std::string &value);
    void parse_permute_kv(PermuteParams *permute, const std::string &key, const std::string &value);
    std::vector<int> parse_vector(const std::string &str);
    float read_hex_float(const std::string &str);
};

#endif