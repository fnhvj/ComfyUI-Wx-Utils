import os
import re
import random
import yaml
import comfy.utils, comfy.sample, comfy.samplers, comfy.controlnet, comfy.model_base, comfy.model_management, comfy.sampler_helpers, comfy.supported_models

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
                "4Guofeng4XL_v12.safetensors": {
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
    CATEGORY = "WX"

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
    RETURN_NAMES = ('触发词', '含触发词的文本', '正向触发词例子', '反向触发词例子', '推荐噪声值', '随机噪声值')
    FUNCTION = "process_lora_prompt"
    CATEGORY = "WX"
          
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
            text_with_triggers,         # 含触发词的文本
            positive_examples_str,      # 正向触发词例子
            negative_examples_str,      # 反向触发词例子
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

class WxLoopTextPrompt:
    """
    循环读取文本文件的每一行
    """
    def __init__(self):
        self.current_line_index = {}  # 用于跟踪每个节点实例的当前行索引
        self.current_direction = {}   # 用于跟踪每个节点实例的方向（正向/反向）
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "prompts.txt"}),
                "start_line": ("INT", {"default": 1, "min": 1, "max": 999999}),
                "end_line": ("INT", {"default": 10, "min": 1, "max": 999999}),
                "mode": (["forward", "reverse", "random", "fixed"], {"default": "forward"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, file_path, start_line, end_line, mode, unique_id):
        """
        检测输入是否发生变化
        """
        # 对于forward、reverse和random模式，我们希望每次都改变
        if mode in ["forward", "reverse", "random"]:
            return float("nan")  # 强制每次都执行
        
        # 对于fixed模式，只有当参数变化时才改变
        file_info = ""
        try:
            if os.path.exists(file_path):
                stat = os.stat(file_path)
                file_info = f"{stat.st_mtime}_{stat.st_size}"
        except:
            file_info = "error"
            
        return f"{file_path}_{start_line}_{end_line}_{mode}_{file_info}"
    
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("文本内容", "文件总行数")
    FUNCTION = "get_text_line"
    CATEGORY = "WX"
    
    def get_text_line(self, file_path, start_line, end_line, mode, unique_id):
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
        total_lines = len(lines)
            
        # 如果没有内容，返回空字符串和总行数0
        if not lines:
            return ("", 0)
            
        # 限制行数范围
        start_idx = max(0, start_line - 1)
        end_idx = min(total_lines, end_line)
        
        # 如果范围无效，返回空字符串和总行数
        if start_idx >= end_idx:
            return ("", total_lines)
            
        # 获取有效范围内的行
        valid_lines = lines[start_idx:end_idx]
        valid_line_count = len(valid_lines)
        
        # 初始化当前实例的状态（如果不存在）
        if unique_id not in self.current_line_index:
            self.current_line_index[unique_id] = 0 if mode != "reverse" else valid_line_count - 1
            self.current_direction[unique_id] = 1  # 1表示正向，-1表示反向
            
        # 根据模式确定当前行
        current_idx = 0
        if mode == "fixed":
            current_idx = 0  # 总是返回第一行
        elif mode == "random":
            current_idx = random.randint(0, valid_line_count - 1)
        else:  # forward 或 reverse
            current_idx = self.current_line_index[unique_id]
            
            # 只有在非fixed和非random模式下才更新索引
            if mode in ["forward", "reverse"]:
                if mode == "forward":
                    current_idx = (current_idx + 1) % valid_line_count
                else:  # reverse
                    current_idx = (current_idx - 1) % valid_line_count
                    
                # 更新索引
                self.current_line_index[unique_id] = current_idx
                
        # 获取当前行文本
        if 0 <= current_idx < valid_line_count:
            return (valid_lines[current_idx], total_lines)
        else:
            return ("", total_lines)

NODE_CLASS_MAPPINGS = {
    "WxGetCkptPrompt": WxGetCkptPrompt,
    "WxGetLoraPrompt": WxGetLoraPrompt,
    "WxLoopTextPrompt": WxLoopTextPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WxGetCkptPrompt": "WX|读取模型提示词",
    "WxGetLoraPrompt": "WX|读取Lora模型提示词",
    "WxLoopTextPrompt": "WX|循环读取文本文件",
}