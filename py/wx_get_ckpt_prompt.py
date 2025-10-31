import os
import re
import random
import yaml
import json
from pathlib import Path
import folder_paths  # 添加此导入

class WxLoopTextPrompt:
    """
    循环读取文本文件的每一行（带显示功能）
    """
    OUTPUT_NODE = True  # 添加 OUTPUT_NODE 标志以支持前端显示
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "prompts.txt"}),
                "start_line": ("INT", {"default": 1, "min": 1, "max": 999999}),
                "next_line": ("INT", {"default": 1, "min": 1, "max": 999999}),
                "end_line": ("INT", {"default": 999999, "min": 1, "max": 999999}),
                "mode": (["正向", "固定", "反向", "随机"], {"default": "正向"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("文本内容", "当前行号")
    FUNCTION = "get_text_line"
    CATEGORY = "WX/提示词"
    
    @classmethod
    def IS_CHANGED(cls, file_path, start_line, next_line, end_line, mode, unique_id=None):
        """
        检测输入是否发生变化
        """
        # 对于正向、反向和随机模式，我们希望每次都改变以实现循环功能
        if mode in ["正向", "反向", "随机"]:
            return float("nan")  # 强制每次都执行
        
        # 对于其他模式，基于所有参数变化来判断
        file_info = ""
        try:
            if os.path.exists(file_path):
                stat = os.stat(file_path)
                file_info = f"{stat.st_mtime}_{stat.st_size}"
        except:
            file_info = "error"
        print("IS_CHANGED检查: ", file_info)
        print(f"IS_CHANGED检查: {file_path}, {start_line}, {next_line}, {end_line}, {mode}")
        return f"{file_path}_{start_line}_{next_line}_{end_line}_{mode}"    
    
    def _get_state_file_path(self, unique_id):
        """获取状态文件路径"""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        state_dir = os.path.join(base_dir, "states")
        os.makedirs(state_dir, exist_ok=True)
        return os.path.join(state_dir, f"loop_text_{unique_id}.json")
    
    def _load_state(self, unique_id):
        """从文件加载状态"""
        state_file = self._get_state_file_path(unique_id)
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载状态文件时出错: {e}")
        return {}
    
    def _save_state(self, unique_id, state):
        """保存状态到文件"""
        state_file = self._get_state_file_path(unique_id)
        try:
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存状态文件时出错: {e}")
    
    def get_text_line(self, file_path, start_line, next_line, end_line, mode, unique_id):
        # 确保start_line不大于end_line
        if start_line > end_line:
            start_line, end_line = end_line, start_line
            
        # 读取文件内容
        lines = []
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
            except Exception as e:
                print(f"读取文件时出错: {e}")
                lines = []
        else:
            print(f"文件不存在: {file_path}")
            lines = []
            
        # 获取文件总行数
        file_total_lines = len(lines)
        
        # 自动修正参数
        original_end_line = end_line
        original_next_line = next_line
        
        # 修正end_line：如果超过文件总行数，则更新为文件总行数
        if end_line > file_total_lines and file_total_lines > 0:
            end_line = file_total_lines
            print(f"自动修正end_line: {original_end_line} -> {end_line}")
        
        # 修正next_line值
        if next_line < start_line:
            next_line = start_line
            print(f"自动修正next_line: {original_next_line} -> {next_line} (下限)")
        elif next_line > end_line:
            next_line = end_line
            print(f"自动修正next_line: {original_next_line} -> {next_line} (上限)")
        
        # 如果没有内容，返回空字符串和默认值
        if not lines:
            status_info = f"状态: 文件无内容 文件总行数: 0"
            return {
                "ui": {
                    "text": [status_info],
                    "next_line": [start_line]
                },
                "result": ("", 0)
            }
            
        start_idx = max(0, start_line - 1)
        end_idx = min(file_total_lines, end_line)
        
        # 如果范围无效，返回空字符串和相关信息
        if start_idx >= end_idx:
            status_info = f"状态: 范围无效 文件总行数: {file_total_lines}"
            return {
                "ui": {
                    "text": [status_info],
                    "next_line": [start_line]
                },
                "result": ("", 0)
            }
            
        # 获取有效范围内的行
        valid_lines = lines[start_idx:end_idx]
        valid_line_count = len(valid_lines)
        
        # 从文件加载状态
        state = self._load_state(unique_id)
        saved_inputs = state.get("inputs", {})
        last_processed_line = state.get("last_processed_line", 0)
        expected_next_line = state.get("expected_next_line", 0)  # 期望的下一行
        
        # 构建当前输入参数
        current_inputs = {
            "file_path": file_path,
            "start_line": start_line,
            "next_line": next_line,
            "end_line": end_line,
            "mode": mode
        }
        
        # 确定当前应该处理的行索引
        if current_inputs != saved_inputs:
            # 参数发生变化
            print(f"参数变化检测: saved_inputs={saved_inputs}, current_inputs={current_inputs}")
            
            # 检查是否是前端自动更新next_line导致的参数变化
            if (expected_next_line > 0 and 
                next_line == expected_next_line and 
                mode in ["正向", "反向"] and
                {k: v for k, v in current_inputs.items() if k != "next_line"} == 
                {k: v for k, v in saved_inputs.items() if k != "next_line"}):
                # 这是前端自动更新next_line，应该使用上次处理的行号来计算下一行
                print(f"检测到前端自动更新next_line: {next_line}")
                last_relative_idx = last_processed_line - start_line
                if 0 <= last_relative_idx < valid_line_count:
                    if mode == "正向":
                        current_idx = (last_relative_idx + 1) % valid_line_count
                    else:  # 反向
                        current_idx = (last_relative_idx - 1) % valid_line_count
                else:
                    current_idx = 0
            else:
                # 用户手动修改了参数，使用next_line参数确定当前行
                relative_idx = next_line - start_line
                if 0 <= relative_idx < valid_line_count:
                    current_idx = relative_idx
                else:
                    current_idx = 0  # 默认从第一行开始
                print(f"用户手动修改参数，使用指定行: next_line={next_line}, current_idx={current_idx}")
        else:
            # 参数未变化，根据模式确定当前行
            if mode == "固定":
                # 固定模式，使用next_line参数确定行
                relative_idx = next_line - start_line
                if 0 <= relative_idx < valid_line_count:
                    current_idx = relative_idx
                else:
                    current_idx = 0
                print(f"固定模式，使用next_line确定行: current_idx={current_idx}")
            elif mode == "随机":
                # 随机模式，每次都随机选择
                current_idx = random.randint(0, valid_line_count - 1)
                print(f"随机模式，current_idx={current_idx}")
            else:
                # 正向或反向模式，基于上次处理的行号计算下一行
                if last_processed_line > 0:
                    # 有上次处理的行记录，根据模式确定下一行
                    last_relative_idx = last_processed_line - start_line
                    if 0 <= last_relative_idx < valid_line_count:
                        if mode == "正向":
                            current_idx = (last_relative_idx + 1) % valid_line_count
                        else:  # 反向
                            current_idx = (last_relative_idx - 1) % valid_line_count
                    else:
                        current_idx = 0
                else:
                    # 没有上次处理记录，使用next_line参数
                    relative_idx = next_line - start_line
                    if 0 <= relative_idx < valid_line_count:
                        current_idx = relative_idx
                    else:
                        current_idx = 0
                print(f"{'正向' if mode == '正向' else '反向'}模式，current_idx={current_idx}")
        
        # 确保索引在有效范围内
        current_idx = max(0, min(current_idx, valid_line_count - 1))
        
        # 计算实际行号
        actual_line_number = start_idx + current_idx + 1
        
        # 计算期望的下一行（用于检测前端自动更新）
        if mode == "正向":
            expected_next_line = (actual_line_number - start_line) % valid_line_count + start_line + 1
        elif mode == "反向":
            expected_next_line = (actual_line_number - start_line - 2) % valid_line_count + start_line + 1
        else:
            expected_next_line = actual_line_number  # 固定和随机模式不预设下一行
        
        # 保存状态（包括输入参数、最后处理的行号和期望的下一行）
        self._save_state(unique_id, {
            "inputs": current_inputs,
            "last_processed_line": actual_line_number,
            "expected_next_line": expected_next_line
        })
        print(f"保存状态: last_processed_line={actual_line_number}, expected_next_line={expected_next_line}")
        
        # 获取当前行文本
        if 0 <= current_idx < valid_line_count:
            # 创建状态信息字符串
            status_info = f"状态: 正常运行 范围: {start_line}-{end_line}\n当前索引: {current_idx+1}/{valid_line_count}"
            
            # 计算下一个行号用于前端回填
            if mode == "正向":
                next_line = actual_line_number + 1 if actual_line_number < end_line else start_line
            elif mode == "反向":
                next_line = actual_line_number - 1 if actual_line_number > start_line else end_line
            else:
                next_line = actual_line_number  # 固定和随机模式保持当前行号
            
            # 返回文本内容、当前行号、总行数和状态信息
            print(f"输出第{actual_line_number}行内容: {valid_lines[current_idx][:50]}...")
            # 修复UI返回格式，确保所有值都是列表
            return {
                "ui": {
                    "text": [status_info],
                    "next_line": [next_line]  # 确保是列表格式
                },
                "result": (valid_lines[current_idx], actual_line_number, file_total_lines)
            }
        else:
            status_info = f"状态: 索引超出范围 文件总行数: {file_total_lines}"
            return {
                "ui": {
                    "text": [status_info],
                    "next_line": [start_line]
                },
                "result": ("", 0)
            }

class WxGetCkptPrompt:
    def __init__(self):
        # 初始化时检查并创建配置文件
        self.ensure_config_exists()
    
    def ensure_config_exists(self):
        """
        检查配置文件是否存在，如果不存在则创建默认配置文件
        """
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models_prompt_config.yaml")
        if not os.path.exists(config_path):
            default_config = {
                "default.safetensors": {
                    "trigger_word": "",
                    "prompt": "((extremely detailed CG)),((8k_wallpaper)),(masterpiece),best quality,high resolution illustration,hyperdetailed,highres,((Overexposure)),bare shoulders,(Upper body),head tilt,seiza,seductive smile,1girl,long hair,beautiful_face,Highly detailed and beautiful eyes,(an extremely delicate and beautiful),(Beautiful and detailed facial depiction),Mature women Chinese antique clothing,White and blue Taoist robe,earrings,necklace,Winter snow,",
                    "negative_prompt":"(((simple background))),monochrome,lowres,bad anatomy,bad hands,text,error,missing fingers,extra digit,fewer digits,cropped,worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,username,blurry,lowres,bad anatomy,bad hands,text,error,extra digit,fewer digits,cropped,worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,username,blurry,ugly,pregnant,vore,duplicate,morbid,mut ilated,tran nsexual,hermaphrodite,long neck,mutated hands,poorly drawn hands,poorly drawn face,mutation,deformed,blurry,bad anatomy,bad proportions,malformed limbs,extra limbs,cloned face,disfigured,gross proportions,(((missing arms))),(((missing legs))),(((extra arms))),(((extra legs))),pubic hair,plump,bad legs,error legs,username,blurry,bad feet,",
                    "cfg": 8.5,
                    "steps": 60,
                    "sampler_name": "euler_ancestral",
                    "scheduler": "Karras",
                    "clip_skip": 2,
                    "vae_name": "",
                    "seed": 0,
                    "denoise": "",
                }
            }
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, allow_unicode=True, indent=2)
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        返回一个包含所有输入字段配置的字典。
        """
        return {
            "required": {
                "config_path": ("STRING", {"default": "models_prompt_config.yaml"}),
            },
            "optional": {
                "pipe": ("PIPE_LINE",),
                "ckpt_name": ("STRING", {"default": ""}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    @classmethod
    def IS_CHANGED(cls, config_path, ckpt_name="", pipe=None, **kwargs):
        """
        简化但有效的变化检测
        """
        # 从不同来源获取模型名称
        model_name = ckpt_name
        if not model_name and pipe and 'loader_settings' in pipe:
            model_name = pipe['loader_settings'].get('ckpt_name', None)
        
        # 获取配置文件的完整路径
        full_config_path = cls._get_config_path(config_path)
        
        # 获取配置文件的状态信息
        file_info = ""
        if os.path.exists(full_config_path):
            try:
                stat = os.stat(full_config_path)
                file_info = f"{stat.st_mtime}_{stat.st_size}"
            except:
                file_info = "error"
        
        # 返回模型名和配置文件信息
        return (model_name or "NO_CHECKPOINT") + "|" + file_info
    
    RETURN_TYPES = ('STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'INT', 'INT', 'FLOAT', 'STRING', 'INT', 'FLOAT')
    RETURN_NAMES = ('触发词', '正向提示词', '负向提示词', 'VAE名称', '采样器', '调度器', 'CLIP停止层', '步数', 'CFG', '模型版本', '种子', '降噪')
    FUNCTION = "get_model_prompt"
    CATEGORY = "WX/提示词"

    @classmethod
    def get_model_prompt(cls, config_path, ckpt_name="", pipe=None, unique_id=None, extra_pnginfo=None):
        # 从不同来源获取模型名称
        model_name = cls._get_model_name(ckpt_name, pipe)
        
        # 如果没有获取到模型名称
        if not model_name:
            print("警告: 无法获取模型名称，返回默认值")
            return (
                "",           # 触发词
                "",           # 正向提示词
                "",           # 负向提示词
                "euler_ancestral",  # 采样器
                "Karras",     # 调度器
                2,            # CLIP停止层
                30,           # 步数
                8.5,          # CFG
                "",           # 模型版本
                0,            # 种子
                0.0,          # 降噪
            )
        
        print(f"最终使用的模型名称: {model_name}")
        
        # 清理模型名称
        base_name = os.path.basename(model_name)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # 处理配置文件路径
        full_config_path = cls._get_config_path(config_path)
        print(f"配置文件路径: {full_config_path}")
        
        # 初始化默认值
        defaults = {
            "trigger_word": "",
            "prompt": "",
            "negative_prompt": "",
            "vae_name": "",
            "sampler_name": "euler_ancestral",
            "scheduler": "Karras",
            "clip_skip": -2,
            "steps": 30,
            "cfg": 8.5,
            "sd_version": "",
            "seed": 0,
            "denoise": 1.0,
        }
        
        # 加载配置文件
        config_data = cls._load_config(full_config_path, base_name, name_without_ext, model_name)
        
        result = cls._merge_config_with_defaults(config_data, defaults)

        
        print(f"触发词: {result['trigger_word']}")
        print(f"提示词: {result['prompt']}")
        print(f"负面提示词: {result['negative_prompt']}")
        print(f"VAE: {result['vae_name']}")
        print(f"采样器: {result['sampler_name']}")
        print(f"调度器: {result['scheduler']}")
        print(f"CLIP停止层: {result['clip_skip']}")
        print(f"步数: {result['steps']}")
        print(f"CFG: {result['cfg']}")
        print(f"模型版本: {result['sd_version']}")
        print(f"种子: {result['seed']}")
        print(f"降噪: {result['denoise']}")
                    
        return (
            result["trigger_word"],
            result["prompt"],
            result["negative_prompt"],
            result["vae_name"],
            result["sampler_name"],
            result["scheduler"],
            result["clip_skip"],
            result["steps"],
            result["cfg"],
            result["sd_version"],
            result["seed"],
            result["denoise"]
        )
    
    @staticmethod
    def _merge_config_with_defaults(config, defaults):
        """
        合并配置文件和默认值
        - 如果配置文件中的值存在且不为空，则使用配置文件的值
        - 否则使用默认值
        """
        merged = {}
        
        # 获取所有可能的键
        all_keys = set(config.keys()) | set(defaults.keys())
        
        for key in all_keys:
            # 检查配置文件中的值是否存在且不为空
            config_value = config.get(key)
            if config_value is not None and config_value != "":
                merged[key] = config_value
            else:
                # 使用默认值
                merged[key] = defaults.get(key)
        
        return merged
    
    @staticmethod
    def _get_model_name(ckpt_name, pipe):
        """
        从不同来源获取模型名称
        """
        # 优先级1: 直接传入的 ckpt_name 参数
        if ckpt_name and ckpt_name.strip():
            print(f"从 ckpt_name 参数获取模型名称: {ckpt_name.strip()}")
            return ckpt_name.strip()
        
        # 优先级2: 从 pipe 中获取
        if pipe and 'loader_settings' in pipe:
            loader_settings = pipe['loader_settings']
            # 检查多个可能的字段
            for key in ['ckpt_name', 'model_name', 'checkpoint']:
                if key in loader_settings and loader_settings[key]:
                    print(f"从 pipe.{key} 获取模型名称: {loader_settings[key]}")
                    return loader_settings[key]
        
        return None
    
    @staticmethod
    def _get_config_path(config_path):
        """
        处理配置文件路径 - 只支持 yaml 扩展名
        """
        base_dir = os.path.dirname(os.path.dirname(__file__))
        
        # 确保使用 yaml 扩展名
        full_path = os.path.join(base_dir, config_path)
        if not full_path.endswith('.yaml'):
            # 移除可能的其他扩展名并添加 .yaml
            full_path = os.path.splitext(full_path)[0] + '.yaml'
        return full_path
    
    @staticmethod
    def _load_config(full_config_path, base_name, name_without_ext, model_name):
        """
        加载配置文件并返回匹配的配置项 (仅支持YAML格式)
        """
        if not os.path.exists(full_config_path):
            print(f"配置文件不存在: {full_config_path}")
            return {}
        
        try:
            with open(full_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            print(f"成功加载配置文件，包含 {len(config)} 个模型配置")
            
            # 匹配优先级：
            # 1. 完整文件名匹配（带扩展名）
            # 2. 无扩展名匹配
            # 3. 原始路径匹配

            matching_attempts = [
                (base_name, "完整文件名"),
                (name_without_ext, "无扩展名"),
                (model_name, "原始路径"),
                (os.path.splitext(model_name)[0], "仅模型名称")  # 新增匹配方式
            ]
            
            for attempt_key, attempt_desc in matching_attempts:
                if attempt_key in config:
                    print(f"通过{attempt_desc}匹配到模型配置: {attempt_key}")
                    print(f"配置内容: {config[attempt_key]}")
                    return config[attempt_key]
            
            print(f"未找到匹配的模型配置")
            if model_name:
                print(f"  搜索的模型名称: {model_name}")
            return {}
            
        except Exception as e:
            print(f"读取配置文件时出错: {e}")
            return {}

class WxGetLoraPrompt:
    """
    获取Lora模型提示词
    """
    def __init__(self):
        pass
        # 初始化时检查并创建配置文件
        # self.ensure_config_exists()
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_text": ("STRING", {"default": "", "multiline": True, "tooltip": "可以是提示词或Lora名称"}),
                "config_path": ("STRING", {"default": "loras_prompt_config.yaml"}),
            },
        }
    @classmethod
    def IS_CHANGED(cls, input_text, config_path):
        """
        检测输入或配置文件是否发生变化
        """
        # 获取配置文件的完整路径
        full_config_path = cls._get_config_path(config_path)
        
        # 获取配置文件的状态信息
        file_info = ""
        if os.path.exists(full_config_path):
            try:
                stat = os.stat(full_config_path)
                file_info = f"{stat.st_mtime}_{stat.st_size}"
            except:
                file_info = "error"
        
        # 返回输入文本和配置文件信息
        return input_text + "|" + file_info
    
    RETURN_TYPES = ('STRING', 'STRING', 'STRING', 'STRING', 'FLOAT', 'FLOAT')
    RETURN_NAMES = ('模型触发词', '含提示词的文本', '正向提示词例子', '反向提示词例子', '推荐噪声值', '随机噪声值')
    FUNCTION = "process_lora_prompt"
    CATEGORY = "WX/提示词"
          
    @classmethod
    def process_lora_prompt(cls, input_text, config_path):
        is_lora_name = False
        lora_name_no_ext = ""  # 初始化变量
        # 输入是提示词，通过正则表达式匹配lora调用代码
        lora_pattern = r'<lora:([^:>]+)(?::[^>]*)?>'
        lora_matches = re.findall(lora_pattern, input_text)
        if not lora_matches:
            # 输入是单个lora名称，需要处理路径和后缀名
            if input_text.strip():
                # 先获取基本文件名（去除路径）
                base_name = os.path.basename(input_text.strip())
                # 再去除后缀名
                lora_name_no_ext = os.path.splitext(base_name)[0]
            else:
                lora_name_no_ext = ""
            lora_matches = [lora_name_no_ext] if lora_name_no_ext else []
            is_lora_name = True
        print(f"输入被视为Lora名称，处理后名称: {lora_name_no_ext}")
        
        # 获取配置文件路径
        full_config_path = cls._get_config_path(config_path)
        
        # 初始化返回值
        triggers = []
        positive_examples = []
        negative_examples = []
        recommended_denoise = 0.5
        random_denoise = 0.5
        
        if lora_matches:
            # 去重但保持顺序
            unique_loras = []
            for lora in lora_matches:
                if lora not in unique_loras:
                    unique_loras.append(lora)
            
            # 读取yaml配置文件
            try:
                with open(full_config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"读取配置文件时出错: {e}")
                config = {}
            
            # 处理每个唯一的lora
            for lora_name in unique_loras:
                # 查找匹配的配置
                config_data = None
                if lora_name in config:
                    config_data = config[lora_name]
                else:
                    # 尝试模糊匹配
                    for key in config.keys():
                        if lora_name.lower() in key.lower() or key.lower() in lora_name.lower():
                            config_data = config[key]
                            break
                
                # 如果找到配置，则提取相关信息
                if config_data:
                    # 提取触发词
                    trigger_word = config_data.get("trigger_word", "")
                    if trigger_word:
                        triggers.append(trigger_word)
                    
                    # 提取正向示例
                    positive_example = config_data.get("prompt", "")
                    if positive_example:
                        positive_examples.append(positive_example)
                    
                    # 提取反向示例
                    negative_example = config_data.get("negative_prompt", "")
                    if negative_example:
                        negative_examples.append(negative_example)
                    
                    # 处理推荐噪声值（使用最后一个有配置的lora的值）
                    denoise_value = config_data.get("denoise", "")
                    if denoise_value != "":
                        recommended_denoise = str(denoise_value)
                    
                    # 处理随机噪声值（使用最后一个有配置的lora的值）
                    min_denoise = config_data.get("min_denoise", 0.3)
                    max_denoise = config_data.get("max_denoise", 0.7)
                    random_denoise = round(random.uniform(min_denoise, max_denoise), 2)
        
        # 组合结果
        triggers_str = ", ".join(triggers) if triggers else ""
        positive_examples_str = "\n".join(positive_examples) if positive_examples else ""
        negative_examples_str = "\n".join(negative_examples) if negative_examples else ""
        
        # 在提示词前插入所有触发词并添加逗号
        text_with_triggers = input_text
        if is_lora_name :
            text_with_triggers = f"<lora:{lora_name_no_ext}:{recommended_denoise}> {triggers_str}"
        else:
            text_with_triggers = f"{triggers_str}, {input_text}"
        
        return (
            triggers_str,               # 触发词
            text_with_triggers,         # 含提示词的文本
            positive_examples_str,      # 正向提示词例子
            negative_examples_str,      # 反向提示词例子
            recommended_denoise,        # 推荐噪声值
            random_denoise              # 随机噪声值
        )
    
    @staticmethod
    def _get_config_path(config_path):
        """
        处理配置文件路径 - 只支持 yaml 扩展名
        """
        base_dir = os.path.dirname(os.path.dirname(__file__))
        
        # 确保使用 yaml 扩展名
        full_path = os.path.join(base_dir, config_path)
        if not full_path.endswith('.yaml'):
            # 移除可能的其他扩展名并添加 .yaml
            full_path = os.path.splitext(full_path)[0] + '.yaml'
        return full_path
    
    @staticmethod
    def _load_config(full_config_path, lora_name):
        """
        加载配置文件并返回匹配的配置项
        """
        if not os.path.exists(full_config_path):
            print(f"配置文件不存在: {full_config_path}")
            return {}
        
        try:
            with open(full_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 查找匹配的lora配置
            if lora_name in config:
                return config[lora_name]
            else:
                # 如果没有精确匹配，尝试模糊匹配
                for key in config.keys():
                    if lora_name.lower() in key.lower() or key.lower() in lora_name.lower():
                        return config[key]
            
            return {}
            
        except Exception as e:
            print(f"读取配置文件时出错: {e}")
            return {}

class WxGetLoraFromTxt:
    """
    从文本文件中根据Lora名称查找匹配的Lora调用代码和提示词，调用代码在前方，提示词在后方
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": ("STRING", {"default": "", "multiline": False, "tooltip": "Lora名称关键字"}),
                "config_path": ("STRING", {"default": "prompt.txt", "multiline": False}),
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, lora_name, config_path):
        """
        检测输入是否发生变化
        """
        file_info = ""
        try:
            if os.path.exists(config_path):
                stat = os.stat(config_path)
                file_info = f"{stat.st_mtime}_{stat.st_size}"
        except:
            file_info = "error"
            
        return f"{lora_name}_{config_path}_{file_info}"
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("Lora调用字符串", "触发词或提示词")
    FUNCTION = "get_lora_config"
    CATEGORY = "WX/提示词"
    
    def get_lora_config(self, lora_name, config_path):
        """
        根据Lora名称从文本文件中查找匹配的配置
        
        处理流程：
        1. 读取配置文件
        2. 逐行查找包含指定Lora名称的行（使用 :lora_name: 模式确保精确匹配）
        3. 解析匹配行的内容，提取Lora调用字符串和提示词
        """
        
        # 检查输入参数
        if not lora_name.strip():
            return ("", "")
        
        # 读取配置文件
        lines = []
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
            except Exception as e:
                print(f"读取配置文件时出错: {e}")
                return (f"<lora:{lora_name.strip()}:0.7>", "")
        else:
            print(f"配置文件不存在: {config_path}")
            return (f"<lora:{lora_name.strip()}:0.7>", "")
        
        # 查找匹配的行 - 使用 :lora_name: 模式确保精确匹配
        matched_line = ""
        search_pattern = f":{lora_name.strip()}:"
        
        for line in lines:
            # 检查行中是否包含 :lora_name: 模式（确保匹配的是Lora调用代码中的名称）
            if search_pattern in line:
                matched_line = line
                break
        
        # 如果没有找到匹配的行，返回默认值
        if not matched_line:
            return (f"<lora:{lora_name.strip()}:0.7>", "")
        
        # 解析匹配的行
        lora_call_string, prompt_text = self._parse_matched_line(matched_line, lora_name)
        
        return (lora_call_string, prompt_text)
    
    def _parse_matched_line(self, line, lora_name):
        """
        解析匹配的行，提取Lora调用字符串和提示词
        
        示例输入:
        <lora:baobao_pony_0_-000006:{0.7|0.8|0.9}>baobao,china_skirt
        <lora:Dragon Ball_XL:{0.7|0.8|0.9}>{1girl,android 18|1girl,bulma|1girl,chi-chi _(dragon ball_)|1girl,trunks _(dragon ball_)}
        """
        # 提取Lora调用部分（从开始到第一个>符号）
        if '>' in line:
            # 分割Lora调用和提示词部分
            parts = line.split('>', 1)
            lora_call_string = parts[0] + '>'  # 保留完整的Lora调用代码
            prompt_text = parts[1] if len(parts) > 1 else ""
        else:
            # 如果没有>符号，整行都作为Lora调用字符串
            lora_call_string = line
            prompt_text = ""
        
        return (lora_call_string, prompt_text)
                


class WxLoopModelList:
    """
    循环加载模型文件的节点
    
    特点:
    - 支持正向、反向、固定和随机四种循环模式
    - 自动保存状态到 states 目录（每个节点实例独立状态）
    - 输出模型名称、绝对路径、相对路径、文件列表
    - 自动校验参数合法性，兼容多级别模型目录
    """
    # 标记为输出节点，支持前端UI交互
    OUTPUT_NODE = True

    def __init__(self):
        # 实例化时初始化，无全局状态，确保多节点独立运行
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """定义节点输入参数（前端显示的配置项）"""
        return {
            "required": {
                # 模型目录（相对于 ComfyUI/models/ 目录，如 "loras"、"checkpoints"、"loras/character"）
                "model_path": ("STRING", {"default": "loras", "multiline": False}),
                # 循环起始索引（1基，与前端输入习惯一致）
                "start_index": ("INT", {"default": 1, "min": 1, "max": 999999}),
                # 下一次要加载的索引（前端回填用，1基）
                "next_index": ("INT", {"default": 1, "min": 1, "max": 999999}),
                # 循环结束索引（1基，超出文件总数时自动修正）
                "end_index": ("INT", {"default": 999999, "min": 1, "max": 999999}),
                # 循环模式（中文选项，符合用户使用习惯）
                "mode": (["正向", "反向", "固定", "随机"], {"default": "正向"}),
                # 强制刷新模型列表（修改目录后勾选生效）
                "refresh": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                # 节点唯一ID（ComfyUI自动注入，用于区分不同节点实例的状态）
                "unique_id": "UNIQUE_ID",
            }
        }

    # 定义输出参数类型和名称（与loop_models返回的result对应）
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("模型名称", "绝对路径", "相对路径", "文件列表")
    # 核心处理函数名（必须与下方定义的方法名一致）
    FUNCTION = "loop_models"
    # 节点在ComfyUI左侧菜单中的分类（自定义，方便查找）
    CATEGORY = "WX/提示词"

    @classmethod
    def IS_CHANGED(cls, model_path, start_index, next_index, end_index, mode, refresh, unique_id=None):
        """
        检测节点是否需要重新执行（ComfyUI核心机制）
        - 正向/反向/随机模式：每次强制执行（返回nan）
        - 固定模式：仅参数或模型目录变化时执行
        """
        # 动态模式（正向/反向/随机）每次都重新执行
        if mode in ["正向", "反向", "随机"]:
            return float("nan")

        # 固定模式：基于参数+模型目录变化判断
        file_info = "unknown"
        try:
            # 获取模型目录的修改时间和大小（检测目录是否有新增/删除文件）
            comfyui_base = folder_paths.base_path
            base_path = Path(comfyui_base) / "models" / model_path
            if base_path.exists():
                stat = base_path.stat()
                file_info = f"{stat.st_mtime}_{stat.st_size}"  # 用修改时间+大小标识目录变化
        except Exception as e:
            file_info = f"error_{str(e)[:20]}"  # 捕获异常，避免节点崩溃

        # 返回参数哈希值：任何参数变化都会触发重新执行
        return f"{model_path}_{start_index}_{next_index}_{end_index}_{mode}_{refresh}_{file_info}"

    def _get_state_file_path(self, unique_id):
        """获取当前节点实例的状态文件路径（states目录下，按unique_id命名）"""
        # 状态文件存放在插件上级目录的states文件夹（避免与其他插件冲突）
        base_dir = os.path.dirname(os.path.dirname(__file__))
        state_dir = os.path.join(base_dir, "states")
        os.makedirs(state_dir, exist_ok=True)  # 目录不存在则创建
        return os.path.join(state_dir, f"loop_model_{unique_id}.json")

    def _load_state(self, unique_id):
        """加载当前节点实例的历史状态（上次执行的索引、参数等）"""
        state_file = self._get_state_file_path(unique_id)
        if os.path.exists(state_file):
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WxLoopModelList] 加载状态失败：{e}（将使用默认状态）")
        # 无状态文件时返回默认空字典
        return {"last_processed_index": 0, "expected_next_index": 0, "inputs": {}}

    def _save_state(self, unique_id, state):
        """保存当前节点实例的状态（供下次执行时使用）"""
        state_file = self._get_state_file_path(unique_id)
        try:
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WxLoopModelList] 保存状态失败：{e}")

    @staticmethod
    def get_model_files(model_path, start_index=1, end_index=None):
        """
        静态方法：获取指定目录下的模型文件列表（按规则筛选）
        - 支持的模型格式：.ckpt/.pt/.bin/.pth/.safetensors
        - 自动按文件名排序，支持索引范围筛选
        """
        try:
            # 判断是否为绝对路径
            if os.path.isabs(model_path):
                # 如果是绝对路径，直接使用
                base_path = Path(model_path)
            else:
                # 如果是相对路径，构建完整模型目录路径（ComfyUI/models/ + 输入的model_path）
                comfyui_base = folder_paths.base_path
                base_path = Path(comfyui_base) / "models" / model_path

            # 目录不存在时返回空列表
            if not base_path.exists():
                print(f"[WxLoopModelList] 模型目录不存在：{base_path}")
                return []

            # 支持的模型文件扩展名（不区分大小写）
            valid_extensions = {".ckpt", ".pt", ".bin", ".pth", ".safetensors"}
            model_files = []

            # 遍历目录下所有符合条件的文件（包括子目录）
            for file_path in base_path.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                    model_files.append(file_path)

            # 按文件名排序（确保循环顺序稳定）
            sorted_files = sorted(model_files, key=lambda x: x.name)
            total_files = len(sorted_files)
            print(f"[WxLoopModelList] 在 {base_path} 找到 {total_files} 个模型文件")

            # 处理索引范围（1基转0基，避免超出列表长度）
            if end_index is None or end_index > total_files:
                end_index = total_files
            # 确保起始索引不小于1，结束索引不小于起始索引
            start_idx = max(0, start_index - 1)  # 1基→0基
            end_idx = min(total_files, end_index)  # 结束索引不超过总文件数

            # 返回筛选后的文件列表
            return sorted_files[start_idx:end_idx]

        except Exception as e:
            print(f"[WxLoopModelList] 获取模型文件失败：{e}")
            return []

    def loop_models(self, model_path, start_index, next_index, end_index, mode, refresh, unique_id):
        """
        核心方法：执行模型循环逻辑
        - 参数校验 → 加载模型列表 → 计算当前索引 → 保存状态 → 返回结果+UI数据
        """
        # --------------------------
        # 1. 前置参数校验（避免无效输入导致逻辑错误）
        # --------------------------
        # 修正start_index > end_index的情况（交换两者）
        if start_index > end_index:
            start_index, end_index = end_index, start_index
            print(f"[WxLoopModelList] 自动修正索引：start_index={start_index}, end_index={end_index}")

        # 获取模型文件列表（refresh=True时强制重新读取）
        model_files = self.get_model_files(model_path, start_index, end_index)
        total_valid_files = len(model_files)

        # 无模型文件时返回空结果+提示
        if total_valid_files == 0:
            status_info = f"状态：未找到模型文件\n目录：models/{model_path}\n筛选范围：{start_index}-{end_index}"
            return {
                "ui": {
                    "text": [status_info],  # 前端状态显示（列表格式）
                    "next_index": [start_index]  # 前端回填next_index（列表格式）
                },
                "result": ("", "", "", "")  # 输出参数（空值）
            }

        # --------------------------
        # 2. 加载历史状态（上次执行的索引和参数）
        # --------------------------
        state = self._load_state(unique_id)
        last_processed_index = state.get("last_processed_index", 0)  # 上次执行的1基索引
        expected_next_index = state.get("expected_next_index", 0)    # 期望的下一个索引（用于前端回填判断）
        saved_inputs = state.get("inputs", {})                       # 上次执行的输入参数

        # 构建当前输入参数字典（用于对比是否有手动修改）
        current_inputs = {
            "model_path": model_path,
            "start_index": start_index,
            "next_index": next_index,
            "end_index": end_index,
            "mode": mode,
            "refresh": refresh
        }

        # --------------------------
        # 3. 计算当前要加载的模型索引（核心循环逻辑）
        # --------------------------
        # 转换为0基索引（方便列表操作）
        start_idx_0 = start_index - 1
        end_idx_0 = start_idx_0 + total_valid_files - 1  # 有效范围的0基结束索引

        # 初始化当前0基索引
        current_idx_0 = 0

        if current_inputs != saved_inputs:
            # 情况1：输入参数有变化（用户手动修改）
            print(f"[WxLoopModelList] 检测到参数变化：{saved_inputs} → {current_inputs}")

            # 特殊判断：是否是前端自动回填next_index导致的参数变化
            is_auto_next_line = (
                expected_next_index > 0 
                and next_index == expected_next_index 
                and mode in ["正向", "反向"]
                # 除next_index外，其他参数一致
                and {k: v for k, v in current_inputs.items() if k != "next_index"} 
                == {k: v for k, v in saved_inputs.items() if k != "next_index"}
            )

            if is_auto_next_line:
                # 子情况1.1：前端自动回填next_index → 基于上次索引计算
                last_relative_idx = last_processed_index - start_index  # 上次索引相对于起始的偏移
                if 0 <= last_relative_idx < total_valid_files:
                    if mode == "正向":
                        current_idx_0 = (last_relative_idx + 1) % total_valid_files
                    else:  # 反向
                        current_idx_0 = (last_relative_idx - 1) % total_valid_files
                else:
                    current_idx_0 = 0  # 偏移无效时从第一个开始
                print(f"[WxLoopModelList] 前端自动回填，计算当前索引：{current_idx_0}")

            else:
                # 子情况1.2：用户手动修改参数 → 基于next_index计算
                relative_idx = next_index - start_index  # next_index相对于起始的偏移
                if 0 <= relative_idx < total_valid_files:
                    current_idx_0 = relative_idx
                else:
                    current_idx_0 = 0  # 偏移无效时从第一个开始
                print(f"[WxLoopModelList] 用户手动修改参数，当前索引：{current_idx_0}")

        else:
            # 情况2：输入参数无变化 → 按模式计算下一个索引
            if mode == "固定":
                # 固定模式：始终使用next_index对应的索引
                relative_idx = next_index - start_index
                current_idx_0 = relative_idx if 0 <= relative_idx < total_valid_files else 0

            elif mode == "随机":
                # 随机模式：每次随机选择一个索引
                current_idx_0 = random.randint(0, total_valid_files - 1)

            else:
                # 正向/反向模式：基于上次索引循环
                if last_processed_index == 0:
                    # 无历史记录 → 从next_index开始
                    relative_idx = next_index - start_index
                    current_idx_0 = relative_idx if 0 <= relative_idx < total_valid_files else 0
                else:
                    # 有历史记录 → 按模式递增/递减
                    last_relative_idx = last_processed_index - start_index
                    if mode == "正向":
                        current_idx_0 = (last_relative_idx + 1) % total_valid_files
                    else:  # 反向
                        current_idx_0 = (last_relative_idx - 1) % total_valid_files

            print(f"[WxLoopModelList] 参数无变化，{mode}模式计算当前索引：{current_idx_0}")

        # 确保当前索引在有效范围内（避免超出列表长度）
        current_idx_0 = max(0, min(current_idx_0, total_valid_files - 1))
        # 转换为1基索引（用于前端显示和状态保存）
        current_idx_1 = start_index + current_idx_0

        # --------------------------
        # 4. 计算前端回填的next_index（下一次要执行的索引）
        # --------------------------
        if mode == "正向":
            # 正向：当前+1，超出end_index则回start_index
            next_index_ui = current_idx_1 + 1 if current_idx_1 < end_index else start_index
        elif mode == "反向":
            # 反向：当前-1，小于start_index则回end_index
            next_index_ui = current_idx_1 - 1 if current_idx_1 > start_index else end_index
        else:
            # 固定/随机：保持当前索引不变
            next_index_ui = current_idx_1

        # 更新期望的下一个索引（用于下次参数变化判断）
        expected_next_index = next_index_ui

        # --------------------------
        # 5. 保存当前状态（供下次执行使用）
        # --------------------------
        new_state = {
            "inputs": current_inputs,
            "last_processed_index": current_idx_1,
            "expected_next_index": expected_next_index
        }
        self._save_state(unique_id, new_state)
        print(f"[WxLoopModelList] 保存状态：上次索引={current_idx_1}，下次回填={expected_next_index}")

        # --------------------------
        # 6. 处理当前模型文件的路径信息
        # --------------------------
        selected_file = model_files[current_idx_0]
        # 模型名称（不带扩展名）
        model_name = selected_file.stem
        # 绝对路径（完整路径，用于直接加载模型）
        absolute_path = str(selected_file.absolute())
        # 相对路径（相对于 ComfyUI/models/模型类型/，方便下游节点使用）
        try:
            # 提取模型类型（如model_path="loras/character" → 类型是"loras"）
            model_type = model_path.split("/")[0]
            models_base_dir = Path(folder_paths.base_path) / "models" / model_type
            relative_path = str(selected_file.relative_to(models_base_dir))
        except Exception as e:
            # 相对路径计算失败时，用绝对路径替代
            relative_path = absolute_path
            print(f"[WxLoopModelList] 计算相对路径失败：{e}（使用绝对路径）")

        # 构建文件列表字符串（每行一个文件，相对路径）
        file_list_str = ""
        try:
            models_base_dir = Path(folder_paths.base_path) / "models" / model_type
            file_list_str = "\n".join([str(f.relative_to(models_base_dir)) for f in model_files])
        except:
            file_list_str = "\n".join([str(f.absolute()) for f in model_files])

        # --------------------------
        # 7. 构建前端状态信息和返回结果
        # --------------------------
        status_info = (
            f"状态：正常运行\n"
            f"模型目录：models/{model_path}\n"
            f"筛选范围：{start_index}-{end_index}（共{total_valid_files}个有效模型）\n"
            f"当前索引：{current_idx_1}（{current_idx_0 + 1}/{total_valid_files}）\n"
            f"当前模型：{model_name}"
        )

        # 返回结果（包含UI数据和输出参数）
        return {
            "ui": {
                "text": [status_info],  # 前端状态显示（必须是列表）
                "next_index": [next_index_ui]  # 前端next_index回填（必须是列表）
            },
            "result": (
                model_name,        # 输出1：模型名称
                absolute_path,     # 输出2：绝对路径
                relative_path,     # 输出3：相对路径
                file_list_str      # 输出4：文件列表
            )
        }


# 在文件末尾添加节点映射
NODE_CLASS_MAPPINGS = {
    "WxLoopTextPrompt": WxLoopTextPrompt,
    "WxGetCkptPrompt": WxGetCkptPrompt,
    "WxGetLoraPrompt": WxGetLoraPrompt,
    "WxGetLoraFromTxt": WxGetLoraFromTxt,
    "WxLoopModelList": WxLoopModelList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WxLoopTextPrompt": "逐行读取TXT文件（循环）|WX",
    "WxGetCkptPrompt": "读取大模型提示词（从预设文件）|WX",
    "WxGetLoraPrompt": "读取Lora模型提示词（从预设文件）|WX",
    "WxGetLoraFromTxt": "读取TXT中的提示词（用Lora名称）|WX",
    "WxLoopModelList": "循环读取模型名（从目录）|WX",
}
    