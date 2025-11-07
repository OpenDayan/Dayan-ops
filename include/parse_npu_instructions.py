import json
import re

class InstructionProcessor:
    def __init__(self, json_data):
        self.json_data = json_data
        self.expanded_insts = {}
        
    def expand_insts(self):
        """展开insts表，补全缺失字段并处理micros引用"""
        for inst_name, inst_data in self.json_data["insts"].items():
            expanded_inst = {
                "base_type": inst_data.get("base_type", "void"),
                "fixed_bits": inst_data.get("fixed_bits", "0b0"),
                "var_bits": inst_data.get("var_bits", [])
            }
            
            # 处理var_bits中的字符串引用和字典引用
            var_bits = expanded_inst["var_bits"]
            if isinstance(var_bits, str):
                # 字符串引用，从micros中查找
                if var_bits in self.json_data["micros"]:
                    expanded_inst["var_bits"] = self.json_data["micros"][var_bits]
                else:
                    expanded_inst["var_bits"] = []
            elif isinstance(var_bits, dict):
                # 单个字典，转换为列表
                expanded_inst["var_bits"] = [var_bits]
            
            self.expanded_insts[inst_name] = expanded_inst
    
    def parse_fixed_bits(self, fixed_bits_str):
        """解析fixed_bits字符串为整数"""
        # 移除空格和0b前缀
        clean_bits = fixed_bits_str.replace(" ", "").replace("0b", "")
        # 确保是32位
        if len(clean_bits) < 32:
            clean_bits = clean_bits.zfill(32)
        return "0b" + clean_bits
    
    def generate_encoding_template_params(self, var_bits):
        """生成编码模板参数"""
        if not var_bits:
            return ""
        
        params = []
        for var_item in var_bits:
            if isinstance(var_item, dict):
                for param_name, bit_ranges in var_item.items():
                    params.append(f"long {param_name}")
            # 如果是列表的列表（嵌套结构），递归处理
            elif isinstance(var_item, list):
                # 这种情况在您的JSON中不应该出现，但为了安全处理
                for sub_item in var_item:
                    if isinstance(sub_item, dict):
                        for param_name, bit_ranges in sub_item.items():
                            params.append(f"long {param_name}")
        
        return ", ".join(params)
    
    def generate_encoding_calculation(self, fixed_bits, var_bits):
        """生成编码计算代码"""
        encoding_var = f"({fixed_bits})"  # 初始化为固定位
        
        if not var_bits:
            return encoding_var
        
        # 为每个参数收集所有位域
        param_bit_fields = {}
        for var_item in var_bits:
            if isinstance(var_item, dict):
                for param_name, bit_ranges in var_item.items():
                    if param_name not in param_bit_fields:
                        param_bit_fields[param_name] = []
                    if isinstance(bit_ranges, list):
                        for bit_range in bit_ranges:
                            if isinstance(bit_range, list) and len(bit_range) == 2:
                                param_bit_fields[param_name].append(bit_range)
        
        # 为每个参数生成编码表达式
        for param_name, bit_ranges in param_bit_fields.items():
            if len(bit_ranges) == 1:
                # 单个位域，直接处理
                start, length = bit_ranges[0]
                mask = (1 << length) - 1
                encoding_var = f"{encoding_var} | (({param_name} & {mask}) << {start})"
            else:
                # 多个位域，需要分别处理每个位域
                bit_field_exprs = []
                for start, length in bit_ranges:
                    mask = (1 << length) - 1
                    # 计算参数中需要提取的位
                    shift_amount = 0
                    # 这里需要根据实际情况调整，假设我们总是从参数的最低有效位提取
                    bit_field_exprs.append(f"(({param_name} & {mask}) << {start})")
                
                # 合并所有位域表达式
                combined_expr = " | ".join(bit_field_exprs)
                encoding_var = f"{encoding_var} | {combined_expr}"
        
        return encoding_var
    
    def get_function_params(self, base_type):
        """获取函数参数列表"""
        if base_type in self.json_data["micros"]:
            micro_def = self.json_data["micros"][base_type]
            if "params" in micro_def:
                params = []
                for param_dict in micro_def["params"]:
                    for param_name, param_type in param_dict.items():
                        if(param_type == "r"):
                            params.append(f"long {param_name}")
                        elif(param_type == "w"):
                            params.append(f"long& {param_name}")
                        else:
                            print('Error: No Support Parms Type: ', param_type)
                return ", ".join(params)
        return ""
    
    def get_inst_builder(self, base_type):
        """获取指令构建器名称"""
        if base_type in self.json_data["micros"]:
            micro_def = self.json_data["micros"][base_type]
            if "inst_builder" in micro_def and micro_def["inst_builder"]:
                return micro_def["inst_builder"][0]
        return "__custom_32b_instruction_void"
    
    def extract_param_names(self, function_params_str):
        """从参数字符串中提取参数名"""
        if not function_params_str:
            return []
        params = [param.strip().split()[-1] for param in function_params_str.split(',')]
        return params
    
    def generate_instruction_function(self, inst_name, inst_data):
        """为单个指令生成C++函数"""
        base_type = inst_data["base_type"]
        fixed_bits_value = self.parse_fixed_bits(inst_data["fixed_bits"])
        var_bits = inst_data["var_bits"]
        
        # 生成模板参数
        template_params = self.generate_encoding_template_params(var_bits)
        
        # 生成函数参数
        function_params = self.get_function_params(base_type)
        
        # 生成编码计算
        encoding_calc = self.generate_encoding_calculation(fixed_bits_value, var_bits)
        
        # 获取指令构建器
        inst_builder = self.get_inst_builder(base_type)
        
        # 提取参数名用于函数调用
        param_names = self.extract_param_names(function_params)
        param_call_str = ", ".join(param_names) if param_names else ""
        
        # 生成函数代码
        if template_params:
            function_code = f"""
template <{template_params}>
__attribute__((always_inline))
static inline void {inst_name}({function_params}) {{
    constexpr long encoding = {encoding_calc};
    {inst_builder}<encoding>({param_call_str});
}}
"""
        else:
            function_code = f"""
__attribute__((always_inline))
static inline void {inst_name}({function_params}) {{
    constexpr long encoding = {fixed_bits_value};
    {inst_builder}<encoding>({param_call_str});
}}
"""
        
        return function_code
    
    def generate_all_functions(self):
        """生成所有指令函数"""
        self.expand_insts()
        
        header = """// Auto-generated instruction assembler
// Generated from JSON instruction definitions

#pragma once

#include <cstdint>
"""
        
#         # 为每个base_type生成前置声明
#         base_types = set()
#         for inst_data in self.expanded_insts.values():
#             base_types.add(inst_data["base_type"])
        
#         for base_type in sorted(base_types):
#             inst_builder = self.get_inst_builder(base_type)
#             function_params = self.get_function_params(base_type)
#             param_names = self.extract_param_names(function_params)
            
#             header += f"""
# template <long encoding>
# __attribute__((always_inline))
# static inline void {inst_builder}({function_params});
# """
        
        header += "\n// Instruction implementations\n"
        
        # 生成所有指令函数
        for inst_name, inst_data in self.expanded_insts.items():
            try:
                header += self.generate_instruction_function(inst_name, inst_data)
            except Exception as e:
                print(f"Error generating function for {inst_name}: {e}")
                print(f"Inst data: {inst_data}")
                raise
        
        return header

def main():
    # 读取JSON文件
    try:
        with open('npu_instruction.json', 'r') as f:
            json_data = json.load(f)
    except FileNotFoundError:
        # 如果文件不存在，使用您提供的示例数据
        json_data = {
            "insts": {
                "tx_mat_tcsr_write": {
                    "base_type": "tcsr_sb_write",
                    "fixed_bits": "0b00000 00 0000000000 000 00000 0001011",
                    "var_bits": "TCSR"
                },
                # ... 其他指令定义
            },
            "micros": {
                "TCSR": [
                    {"tcsr_id": [[15, 10]]}
                ],
                # ... 其他micros定义
            }
        }
        print("Using built-in example data")
    
    # 处理并生成代码
    processor = InstructionProcessor(json_data)
    cpp_code = processor.generate_all_functions()
    
    # 输出到文件
    with open('gen_npu_instruction.h', 'w') as f:
        f.write(cpp_code)
    
    print("C++ instruction assembler generated successfully!")

if __name__ == "__main__":
    main()