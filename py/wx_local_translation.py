import os
import sys
import requests
import json


sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'comfy'))

class WxLocalTranslation:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
 
        return {
            "required": {
                "source_lan": (['en', 'zh', 'zt', 'ja'], {'default': 'en'}),
                "target_lan": (['en', 'zh', 'zt', 'ja'], {'default': 'zh'}),
                "service_url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:9911/translate"
                }),
                "source_text": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    #RETURN_NAMES = ("TRAGET_TEXT",)

    FUNCTION = "localtranslation"

    OUTPUT_NODE = True

    CATEGORY = "wx"

    def localtranslation(self, source_text,source_lan,target_lan,service_url):
        #print(f"{source_text}{source_lan}{target_lan}{service_url}")
        #result=requests.post(service_url,json={"q":source_text,"source":source_lan,"target":target_lan})
        #print(f"{result}")
        #res = result.json()
        #print(res)
        #print(res['translatedText'])
        #return (res['translatedText'],)
        #return ("错误",)
        #result=requests.post("http://127.0.0.1:9911/translate",json={"q":"今天天气真好","source":"zh","target":"en"})
        #if result.status_code==200:
            #res = result.json()
            #print(res['translatedText'])
            #return (res['translatedText'],)
        #url = "http://fnhvj.iiiii.info:19911/translate"
        payload = {
            "q": source_text,
            "source": source_lan,
            "target": target_lan
        }
        
        response = requests.post(service_url, json=payload)
        try:
            # 处理响应
            if response.status_code == 200:
                print(type(response))
                print(type(response.json()))
                res = response.json()
                return (res['translatedText'],)
            else:
                return ('网络连接出错',)
        finally:
            # 确保连接被关闭
            response.close()

#class WxPreviewText:
#    def __init__(self):
#        pass

#    @classmethod
#    def INPUT_TYPES(cls):
#        return {
#            "required": {
#                "text": ("STRING", {"forceInput": True}),
#                },
#        }

#    RETURN_TYPES = ()
#    INPUT_IS_LIST = True
#    FUNCTION = "preview_text"
#    OUTPUT_NODE = True
#    CATEGORY = "utils"
#    # OUTPUT_IS_LIST = True

#    def preview_text(self, text):

#        return {"ui": {"text": text}, "result": (text,)}

#NODE_CLASS_MAPPINGS = {
#    "WxLocalTranslation": WxLocalTranslation,
#    "WxPreviewText": WxPreviewText,
#}

#NODE_DISPLAY_NAME_MAPPINGS = {
#    "WxLocalTranslation": "wx|本地AI翻译",
#    "WxPreviewText": "wx|文本预览"
#}
