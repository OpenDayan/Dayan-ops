#include "common.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <cstring>

// 从二进制文件中读取所有 float 数据
std::vector<float> readF32FromFile(const std::string& filename) {
    // 以二进制模式打开文件，并定位到文件末尾
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Open file failed: " + filename);
    }

    // 获取文件大小
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // 检查文件大小是否是 float 大小的整数倍
    if (size % sizeof(float) != 0) {
        throw std::runtime_error("Invalid trace file, file size cannot be divided by 4: " + filename);
    }

    std::vector<float> buffer(size / sizeof(float));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Error in reading file: " + filename);
    }

    return buffer;
}

std::vector<uint8_t> readUInt8FromFile(const std::string& filename) {
    // 以二进制模式打开文件，并定位到文件末尾
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Open file failed: " + filename);
    }

    // 获取文件大小
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // 检查文件大小是否是 uint8_t 大小的整数倍
    if (size % sizeof(uint8_t) != 0) {
        throw std::runtime_error("Invalid trace file, file size cannot be divided by 1: " + filename);
    }

    std::vector<uint8_t> buffer(size / sizeof(uint8_t));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Error in reading file: " + filename);
    }

    return buffer;
}

std::vector<uint16_t> readUInt16FromFile(const std::string& filename) {
    // 以二进制模式打开文件，并定位到文件末尾
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Open file failed: " + filename);
    }

    // 获取文件大小
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // 检查文件大小是否是 uint16_t 大小的整数倍
    if (size % sizeof(uint16_t) != 0) {
        throw std::runtime_error("Invalid trace file, file size cannot be divided by 2: " + filename);
    }

    std::vector<uint16_t> buffer(size / sizeof(uint16_t));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Error in reading file: " + filename);
    }

    return buffer;
}

std::vector<int> readInt32FromFile(const std::string& filename) {
    // 以二进制模式打开文件，并定位到文件末尾
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Open file failed: " + filename);
    }

    // 获取文件大小
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // 检查文件大小是否是 int 大小的整数倍
    if (size % sizeof(int) != 0) {
        throw std::runtime_error("Invalid trace file, file size cannot be divided by 4: " + filename);
    }

    std::vector<int> buffer(size / sizeof(int));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Error in reading file: " + filename);
    }

    return buffer;
}

static void trim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

void TraceParamsParser::parse(const std::string& filename, TensorParams * pdst, TensorParams * psrc0,
            TensorParams * psrc1, NormParams * pnorm, RopeParams * prope, SoftmaxParams * psoft) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    file.close();

    std::stringstream processed_stream(content);
    std::string line;

    // 使用状态机确定当前正在解析哪个部分
    enum class Section { NONE, SRC0, SRC1, DST, ROPE, NORM, SOFTMAX};
    Section current_section = Section::NONE;

    while (std::getline(processed_stream, line)) {
        trim(line);
        if (line.empty() || line.find("====") != std::string::npos) {
            // 检查并切换区块
            if (line.find("src0") != std::string::npos) current_section = Section::SRC0;
            else if (line.find("src1") != std::string::npos) current_section = Section::SRC1;
            else if (line.find("dst") != std::string::npos) current_section = Section::DST;
            else if (line.find("RoPE params") != std::string::npos) current_section = Section::ROPE;
            else if (line.find("Norm params") != std::string::npos) current_section = Section::NORM;
            else if (line.find("Softmax params") != std::string::npos) current_section = Section::SOFTMAX;
            continue;
        }

        // 查找分隔符 ('=' 或 ':')
        size_t separator_pos = line.find('=');
        if (separator_pos == std::string::npos) continue;

        std::string key = line.substr(0, separator_pos);
        std::string value = line.substr(separator_pos + 1);
        trim(key);
        trim(value);

        // 根据当前区块，将键值对分配给对应的解析函数
        switch (current_section) {
            case Section::SRC0:
                parse_tensor_kv(psrc0, key, value);
                break;
            case Section::SRC1:
                parse_tensor_kv(psrc1, key, value);
                break;
            case Section::DST:
                parse_tensor_kv(pdst, key, value);
                break;
            case Section::ROPE:
                parse_rope_kv(prope, key, value);
                break;
            case Section::NORM:
                parse_norm_kv(pnorm, key, value);
                break;
            case Section::SOFTMAX:
                parse_softmax_kv(psoft, key, value);
                break;
            case Section::NONE:
                // 忽略在任何区块之外的参数 (除了 'name')
                break;
        }
    }
}

// 解析 TensorParams 的键值对
void TraceParamsParser::parse_tensor_kv(TensorParams * tensor, const std::string& key, const std::string& value) {
    if (tensor == NULL) return;
    if (key == "ne") tensor->ne = parse_vector(value);
    else if (key == "nb") tensor->nb = parse_vector(value);
    else if (key == "dtype") tensor->type = value;
    else if (key == "is_contiguous") tensor->is_contiguous = std::stoi(value);
    else if (key.find("size") != std::string::npos) tensor->len = std::stoll(value);
}

// 解析 RopeParams 的键值对
void TraceParamsParser::parse_rope_kv(RopeParams * rope, const std::string& key, const std::string& value) {
    if (rope == NULL) return;
    if (key == "n_dims") rope->n_dims = std::stoi(value);
    else if (key == "mode") rope->mode = std::stoi(value);
    else if (key == "n_ctx_orig") rope->n_ctx_orig = std::stoll(value);
    else if (key == "freq_base") rope->freq_base = read_hex_float(value);
    else if (key == "freq_scale") rope->freq_scale = read_hex_float(value);
    else if (key == "ext_factor") rope->ext_factor = read_hex_float(value);
    else if (key == "attn_factor") rope->attn_factor = read_hex_float(value);
    else if (key == "beta_fast") rope->beta_fast = read_hex_float(value);
    else if (key == "beta_slow") rope->beta_slow = read_hex_float(value);
    else if (key == "sections") rope->sections = parse_vector(value);
}

// 解析 RMSParams的键值对
void TraceParamsParser::parse_norm_kv(NormParams * rms, const std::string& key, const std::string& value) {
    if(rms == nullptr) return;
    if(key == "eps") rms->eps = read_hex_float(value);
}

void TraceParamsParser::parse_softmax_kv(SoftmaxParams * softmax, const std::string& key, const std::string& value) {
    if (softmax == NULL) return;
    if (key == "scale") softmax->scale = read_hex_float(value);
    else if (key == "max_bias") softmax->max_bias = read_hex_float(value);
}

std::vector<int> TraceParamsParser::parse_vector(const std::string& str) {
    std::vector<int> result;
    std::string numbers_str = str;

    // 移除方括号
    size_t first = numbers_str.find('[');
    size_t last = numbers_str.rfind(']');
    if (first != std::string::npos && last != std::string::npos) {
        numbers_str = numbers_str.substr(first + 1, last - first - 1);
    }

    // 将逗号替换为空格
    std::replace(numbers_str.begin(), numbers_str.end(), ',', ' ');

    std::stringstream ss(numbers_str);
    int num;
    while (ss >> num) {
        result.push_back(num);
    }
    return result;
}

float TraceParamsParser::read_hex_float(const std::string& str) {
    size_t start = str.find("(0x");
    size_t end = str.find(')', start);
    std::string hexStr = str.substr(start + 3, end - start - 3);
    uint32_t f_u32 = static_cast<uint32_t>(std::stoul(hexStr, nullptr, 16));
    float res = 0.0;
    std::memcpy(&res, &f_u32, sizeof(float));
    // printf("[INFO] catch float %f (0x%x)\n", res, f_u32);
    return res;
}