import os
import sys
from zhipuai import ZhipuAI

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'comfy'))

class ZhiPuChat:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
 
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "model": (['glm-4', 'glm-3-turbo', 'characterglm'], {'default': 'glm-4'}),
                "question": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)

    FUNCTION = "zhipuchat"

    OUTPUT_NODE = True

    CATEGORY = "Utils|WX"

    def zhipuchat(self, question,api_key,model):
        if question=="" or api_key=="":
            return ("你没有填写api_key",)
        if question=="":
            return ("没有收到你的问题",)
        self.client = ZhipuAI(api_key=api_key)
        answer = ""
        response = self.client.chat.completions.create(
            model=model,  # 填写需要调用的模型名称
            messages=[
                {"role": "user", "content": question},
            ],
        )
        print(response.choices[0].message.content)
        answer = response.choices[0].message.content if response else answer
        return (answer,)


NODE_CLASS_MAPPINGS = {
    "WxZhiPuChat": ZhiPuChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WxZhiPuChat": "wx|智谱AI",
}
