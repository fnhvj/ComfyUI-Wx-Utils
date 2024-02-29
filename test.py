class TextNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string_field": ("STRING", {
                    "multiline": True, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": "Hello World!"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "text_node"

    OUTPUT_NODE = True

    CATEGORY = "WX"

    def text_node(self, string_field):
        return (string_field,)



NODE_CLASS_MAPPINGS = {
    "TextNode": TextNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "TextNode": "Text Node"
}


c=TextNode()
print(c.text_node("aaaaa"))