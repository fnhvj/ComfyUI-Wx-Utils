# file: wx_test.py
import torch
import numpy as np
from PIL import Image
import os
import folder_paths  # 添加此导入

class ExampleNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "Hello World","display": "输入文本"}),  # 输入参数定义
            }
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "process"
    CATEGORY = "WX/测试"
    OUTPUT_NODE = True  # 标记为输出节点，允许UI更新

    def process(self, text):
        # 处理逻辑
        # result = f"处理后的文本: {text}"
        
        # 返回 ui 字典用于前端显示
        return {
            "ui": {
                "string": [f"节点处理了文本: {text}"],
                "change_widget": ["text"]  # 标记需要更改的 widget
            },
            "result": ()
        }


# 在文件末尾添加节点映射
NODE_CLASS_MAPPINGS = {
    "ExampleNode": ExampleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExampleNode": "测试节点|WX",
}