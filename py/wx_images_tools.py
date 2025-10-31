import os
import re
import sys
import json
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
import locale
from datetime import datetime
from pathlib import Path
import base64
import torch
from torch import Tensor
from io import BytesIO
from PIL import Image, ImageOps, ImageSequence
import folder_paths

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'comfy'))
original_locale = locale.setlocale(locale.LC_TIME, '')

import folder_paths

class WxSaveImageExtended:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = 'output'
        self.prefix_append = ''

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'filename_prefix': ('STRING', {'default': 'ComfyUI_[time(%Y-%m-%d)]'}),
                'filename_keys': ('STRING', {'default': 'ckpt_name, vae_name, seed, steps, cfg', 'multiline': False}),
                'foldername_prefix': ('STRING', {'default': '[time(%Y-%m-%d)]'}),
                'foldername_keys': ('STRING', {'default': '', 'multiline': False}),
                'delimiter': (['underscore','dot', 'comma'], {'default': 'underscore'}),
                'save_job_data': (['disabled', 'prompt', 'basic, prompt', 'basic, sampler, prompt', 'basic, models, sampler, prompt'],{'default': 'disabled'}),
                'job_data_per_image': (['disabled', 'enabled'],{'default': 'disabled'}),
                'job_custom_text': ('STRING', {'default': '', 'multiline': False}),
                'save_metadata': (['disabled', 'enabled'], {'default': 'enabled'}),
                'counter_digits': ([2, 3, 4, 5, 6], {'default': 3}),
                'counter_position': (['first', 'last'], {'default': 'last'}),
                'one_counter_per_folder': (['disabled', 'enabled'], {'default': 'enabled'}),
                'image_preview': (['disabled', 'enabled'], {'default': 'disabled'}),
            },
            "optional": {
                    'images': ('IMAGE', ),
                    "pipe": ("PIPE_LINE",),
                    "positive_text_opt": ("STRING", {"default": "","forceInput": True}),
                    "negative_text_opt": ("STRING", {"default": "","forceInput": True}),
                    "step_opt" : ("INT", {"default": 20,"forceInput": True}),
                    "vae_name_text_opt" : ("STRING", {"forceInput": True}),
                    "size_opt" : ("STRING", {"forceInput": True}),
                    "seed_opt" : ("INT", {"default": 0,"forceInput": True}),
                    "model_text_opt" : ("STRING", {"forceInput": True}),
                    "sampler_text_opt" : ("STRING", {"default": "","forceInput": True}),
                    "cfg_opt" : ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,"forceInput": True}),
                    "clip_skip_opt" : ("INT", {"default": 0,"forceInput": True}),
                    "lora_name_text_opt" : ("STRING", {"forceInput": True}),
                    "denoise_opt" : ("FLOAT", {"default": 7,"forceInput": True}),
                    # "model_hash" : ("STRING", {"default": "","forceInput": False}),
                    # "lora_hashes" : ("STRING", {"forceInput": False}),
                    },
            'hidden': {'prompt': 'PROMPT', 'extra_pnginfo': 'EXTRA_PNGINFO'},
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = 'save_images'
    OUTPUT_NODE = True
    CATEGORY = "WX/图像"

    def get_subfolder_path(self, image_path, output_path):
        image_path = Path(image_path).resolve()
        output_path = Path(output_path).resolve()
        relative_path = image_path.relative_to(output_path)
        subfolder_path = relative_path.parent

        return str(subfolder_path)

    # Get current counter number from file names
    def get_latest_counter(self, one_counter_per_folder, folder_path, filename_prefix, counter_digits, counter_position='last'):
        counter = 1
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist, starting counter at 1.")
            return counter

        try:
            files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
            if files:
                if counter_position == 'last':
                    counters = [int(f[-(4 + counter_digits):-4]) if f[-(4 + counter_digits):-4].isdigit() else 0 for f in files if one_counter_per_folder == 'enabled' or f.startswith(filename_prefix)]
                elif counter_position == 'first':
                    counters = [int(f[:counter_digits]) if f[:counter_digits].isdigit() else 0 for f in files if one_counter_per_folder == 'enabled' or f[counter_digits +1:].startswith(filename_prefix)]
                else:
                    print("Invalid counter_position. Using 'last' as default.")
                    counters = [int(f[-(4 + counter_digits):-4]) if f[-(4 + counter_digits):-4].isdigit() else 0 for f in files if one_counter_per_folder == 'enabled' or f.startswith(filename_prefix)]

                if counters:
                    counter = max(counters) + 1

        except Exception as e:
            print(f"An error occurred while finding the latest counter: {e}")

        return counter

    def cover_string(self,string):
        new_string = string
        try:
            if new_string not in [None, '', "none", "."]:
                if re.findall(r"\[time\(.*?\)\]",string):
                    time_format = re.search(r"\[time\((.*?)\)\]",string).group(1)
                    time_string = datetime.strftime(datetime.now(),format=time_format)
                    new_string = re.sub(r"\[time\(.*?\)\]",time_string,string)
        except:
            print(f"{string} is worng format!!")
        new_string = re.sub(r'[\/:*?"<>|]','_',new_string)
        # print(new_string)
        return new_string
            
    @staticmethod
    def find_keys_recursively(d, keys_to_find, found_values):
        for key, value in d.items():
            if key in keys_to_find:
                found_values[key] = value
            if isinstance(value, dict):
                WxSaveImageExtended.find_keys_recursively(value, keys_to_find, found_values)

    @staticmethod
    def remove_file_extension(value):
        if isinstance(value, str) and value.endswith('.safetensors'):
            base_value = os.path.basename(value)
            value = base_value[:-12]
        if isinstance(value, str) and value.endswith('.pt'):
            base_value = os.path.basename(value)
            value = base_value[:-3]

        return value

    @staticmethod
    def find_parameter_values(target_keys, obj, found_values=None):
        if found_values is None:
            found_values = {}

        if not isinstance(target_keys, list):
            target_keys = [target_keys]

        loras_string = ''
        for key, value in obj.items():
            if 'loras' in target_keys:
                # Match both formats: lora_xx and lora_name_x
                if re.match(r'lora(_name)?(_\d+)?', key):
                    if value.endswith('.safetensors'):
                        value = WxSaveImageExtended.remove_file_extension(value)
                    if value != 'None':
                        loras_string += f'{WxSaveImageExtended.cover_string(value)}, '

            if key in target_keys:
                if (isinstance(value, str) and value.endswith('.safetensors')) or (isinstance(value, str) and value.endswith('.pt')):
                    value = WxSaveImageExtended.remove_file_extension(value)
                found_values[key] = WxSaveImageExtended.cover_string(value)

            if isinstance(value, dict):
                WxSaveImageExtended.find_parameter_values(target_keys, value, found_values)

        if 'loras' in target_keys and loras_string:
            found_values['loras'] = loras_string.strip(', ')

        if len(target_keys) == 1:
            return found_values.get(target_keys[0], None)

        return found_values

    @staticmethod
    def generate_custom_name(keys_to_extract, prefix, delimiter_char, resolution, prompt):
        custom_name = prefix

        if prompt is not None and len(keys_to_extract) > 0:
            found_values = {'resolution': resolution}
            WxSaveImageExtended.find_keys_recursively(prompt, keys_to_extract, found_values)
            for key in keys_to_extract:
                value = found_values.get(key)
                if value is not None:
                    if key == 'cfg' or key =='denoise':
                        try:
                            value = round(float(value), 1)
                        except ValueError:
                            pass

                    if (isinstance(value, str) and value.endswith('.safetensors')) or (isinstance(value, str) and value.endswith('.pt')):
                        value = WxSaveImageExtended.remove_file_extension(value)

                    custom_name += f'{delimiter_char}{value}'

        return custom_name.strip(delimiter_char)

    @staticmethod
    def save_job_to_json(save_job_data, prompt, filename_prefix, positive_text_opt, negative_text_opt, job_custom_text, resolution, output_path, filename):
        prompt_keys_to_save = {}
        if 'basic' in save_job_data:
            if len(filename_prefix) > 0:
                prompt_keys_to_save['filename_prefix'] = filename_prefix
            prompt_keys_to_save['resolution'] = resolution
        if len(job_custom_text) > 0:
            prompt_keys_to_save['custom_text'] = job_custom_text

        if 'models' in save_job_data:
            models = WxSaveImageExtended.find_parameter_values(['ckpt_name', 'loras', 'vae_name', 'model_name'], prompt)
            if models.get('ckpt_name'):
                prompt_keys_to_save['checkpoint'] = models['ckpt_name']
            if models.get('loras'):
                prompt_keys_to_save['loras'] = models['loras']
            if models.get('vae_name'):
                prompt_keys_to_save['vae'] = models['vae_name']
            if models.get('model_name'):
                prompt_keys_to_save['upscale_model'] = models['model_name']



        if 'sampler' in save_job_data:
            prompt_keys_to_save['sampler_parameters'] = WxSaveImageExtended.find_parameter_values(['seed', 'steps', 'cfg', 'sampler_name', 'scheduler', 'denoise'], prompt)

        if 'prompt' in save_job_data:
            if positive_text_opt is not None:
                if not (isinstance(positive_text_opt, list) and
                        len(positive_text_opt) == 2 and
                        isinstance(positive_text_opt[0], str) and
                        len(positive_text_opt[0]) < 6 and
                        isinstance(positive_text_opt[1], (int, float))):
                    prompt_keys_to_save['positive_prompt'] = positive_text_opt

            if negative_text_opt is not None:
                if not (isinstance(positive_text_opt, list) and len(negative_text_opt) == 2 and isinstance(negative_text_opt[0], str) and isinstance(negative_text_opt[1], (int, float))):
                    prompt_keys_to_save['negative_prompt'] = negative_text_opt

            #If no user input for prompts
            if positive_text_opt is None and negative_text_opt is None:
                if prompt is not None:
                    for key in prompt:
                        class_type = prompt[key].get('class_type', None)
                        inputs = prompt[key].get('inputs', {})

                        # Efficiency Loaders prompt structure
                        if class_type == 'Efficient Loader' or class_type == 'Eff. Loader SDXL':
                            if 'positive' in inputs and 'negative' in inputs:
                                prompt_keys_to_save['positive_prompt'] = inputs.get('positive')
                                prompt_keys_to_save['negative_prompt'] = inputs.get('negative')

                        # KSampler/UltimateSDUpscale prompt structure
                        elif class_type == 'KSampler' or class_type == 'KSamplerAdvanced' or class_type == 'UltimateSDUpscale':
                            positive_ref = inputs.get('positive', [])[0] if 'positive' in inputs else None
                            negative_ref = inputs.get('negative', [])[0] if 'negative' in inputs else None

                            positive_text = prompt.get(str(positive_ref), {}).get('inputs', {}).get('text', None)
                            negative_text = prompt.get(str(negative_ref), {}).get('inputs', {}).get('text', None)

                            # If we get non text inputs
                            if positive_text is not None:
                                if isinstance(positive_text, list):
                                    if len(positive_text) == 2:
                                        if isinstance(positive_text[0], str) and len(positive_text[0]) < 6:
                                            if isinstance(positive_text[1], (int, float)):
                                                continue
                                prompt_keys_to_save['positive_prompt'] = positive_text

                            if negative_text is not None:
                                if isinstance(negative_text, list):
                                    if len(negative_text) == 2:
                                        if isinstance(negative_text[0], str) and len(negative_text[0]) < 6:
                                            if isinstance(negative_text[1], (int, float)):
                                                continue
                                prompt_keys_to_save['positive_prompt'] = negative_text

        # Append data and save
        json_file_path = os.path.join(output_path, filename)
        existing_data = {}
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, 'r') as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                print(f"The file {json_file_path} is empty or malformed. Initializing with empty data.")
                existing_data = {}

        timestamp = datetime.now().strftime('%c')
        new_entry = {}
        new_entry[timestamp] = prompt_keys_to_save
        existing_data.update(new_entry)

        with open(json_file_path, 'w') as f:
            json.dump(existing_data, f, indent=4)


    def save_images(self,
                    counter_digits,
                    counter_position,
                    one_counter_per_folder,
                    delimiter,
                    filename_keys,
                    foldername_keys,
                    image_preview,
                    save_job_data,
                    job_data_per_image,
                    job_custom_text,
                    save_metadata,
                    filename_prefix='',
                    foldername_prefix='',
                    images=None,
                    pipe=None,
                    extra_pnginfo=None,
                    negative_text_opt=None,
                    positive_text_opt=None,
                    prompt=None,
                    step_opt=None,
                    size_opt=None,
                    vae_name_text_opt=None,
                    seed_opt=None,
                    model_text_opt=None,
                    sampler_text_opt=None,
                    cfg_opt=None,
                    clip_skip_opt=None,
                    lora_name_text_opt=None,
                    denoise_opt=None,
                ):

        delimiter_char = "_" if delimiter =='underscore' else '.' if delimiter =='dot' else ','
        # print(pipe)
        images = images if isinstance(images, list) and len(images) > 0 else (pipe.get("images") if pipe else images)
        
        # Get set resolution value
        i = 255. * images[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        resolution = f'{img.width}x{img.height}'

        filename_keys_to_extract = [item.strip() for item in filename_keys.split(',')]
        foldername_keys_to_extract = [item.strip() for item in foldername_keys.split(',')]
        custom_filename = WxSaveImageExtended.generate_custom_name(filename_keys_to_extract, filename_prefix, delimiter_char, resolution, prompt)
        custom_foldername = WxSaveImageExtended.generate_custom_name(foldername_keys_to_extract, foldername_prefix, delimiter_char, resolution, prompt)
        # print(custom_foldername)
        # print(custom_filename)
        # Create and save images
        try:
            full_output_folder, filename, _, _, custom_filename = folder_paths.get_save_image_path(custom_filename, self.output_dir, images[0].shape[1], images[0].shape[0])
            # print(custom_filename)
            custom_foldername=self.cover_string(custom_foldername)
            output_path = os.path.join(full_output_folder, custom_foldername)
            # print(custom_filename,output_path)
            os.makedirs(output_path, exist_ok=True)
            counter = self.get_latest_counter(one_counter_per_folder, output_path, filename, counter_digits, counter_position)

            results = list()
            for image in images:
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                metadata = None
                if save_metadata == 'enabled':
                    metadata = PngInfo()
                    if prompt is not None:
                        metadata.add_text('prompt', json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                    positive = positive_text_opt if positive_text_opt is not None else ''
                    negative = negative_text_opt if negative_text_opt is not None else ''
                    step = step_opt if step_opt is not None else 25
                    vae_name = vae_name_text_opt if vae_name_text_opt is not None else ''
                    seed = seed_opt if seed_opt is not None else 0
                    model = model_text_opt if model_text_opt is not None else ''
                    sampler = sampler_text_opt if sampler_text_opt is not None  else ''
                    cfg = cfg_opt if cfg_opt is not None else 7
                    clip_skip = clip_skip_opt if clip_skip_opt is not None else 1
                    lora_name = lora_name_text_opt if lora_name_text_opt is not None else ''
                    denoise = denoise_opt if denoise_opt is not None else 0.75
                    size = size_opt if size_opt is not None else resolution
                    if pipe is not None:
                        positive = pipe["loader_settings"]["positive"] if 'positive' in pipe['loader_settings'] else ''
                        negative = pipe["loader_settings"]["negative"] if 'negative' in pipe['loader_settings'] else ''
                        step = int(pipe["loader_settings"]["steps"]) if 'steps' in pipe['loader_settings'] else ''
                        vae_name = pipe["loader_settings"]["vae_name"] if 'vae_name' in pipe['loader_settings'] else ''
                        seed = pipe["seed"] if 'seed' in pipe else 0
                        model = pipe["loader_settings"]["ckpt_name"] if 'ckpt_name' in pipe['loader_settings'] else ''
                        sampler = pipe["loader_settings"]["sampler_name"] if 'sampler_name' in pipe['loader_settings'] else ''
                        cfg = round(float(pipe["loader_settings"]["cfg"]),2) if 'cfg' in pipe['loader_settings'] else ''
                        clip_skip = int(pipe["loader_settings"]["clip_skip"]) if 'clip_skip' in pipe['loader_settings'] else -2
                        lora_name = pipe["loader_settings"]["lora_name"] if 'lora_name' in pipe['loader_settings'] else ''
                        denoise = pipe["loader_settings"]["denoise"] if 'denoise' in pipe['loader_settings'] else 0.7
                    parameters  = f"""{positive_text_opt}\nNegative prompt: {negative_text_opt}\nSteps: {step}, VAE: {vae_name}, Seed: {seed}, Model: {model}, Sampler: {sampler}, CFG scale: {cfg}, Size: {size}, Denoising strength: {denoise}, Clip skip: {clip_skip}"""
                    metadata.add_text('parameters',parameters)

                if counter_position == 'last':
                    file = f'{filename}{delimiter_char}{counter:0{counter_digits}}.png'
                else:
                    file = f'{counter:0{counter_digits}}{delimiter_char}{filename}.png'
                # print(file)
                file = self.cover_string(file)
                image_path = os.path.join(output_path, file)
                # print(file,image_path)
                img.save(image_path, pnginfo=metadata, compress_level=4)

                if save_job_data != 'disabled' and job_data_per_image =='enabled':
                    WxSaveImageExtended.save_job_to_json(save_job_data, prompt, filename_prefix, positive_text_opt, negative_text_opt, job_custom_text, resolution, output_path, f'{file.strip(".png")}.json')

                subfolder = self.get_subfolder_path(image_path, self.output_dir)
                results.append({ 'filename': file, 'subfolder': subfolder, 'type': self.type})
                counter += 1

            if save_job_data != 'disabled' and job_data_per_image =='disabled':
                WxSaveImageExtended.save_job_to_json(save_job_data, prompt, filename_prefix, positive_text_opt, negative_text_opt, job_custom_text, resolution, output_path, 'jobs.json')

        except OSError as e:
            print(f'An error occurred while creating the subfolder or saving the image: {e}')
        else:
            if image_preview == 'disabled':
                results = list()
            return { 'ui': { 'images': results } }

class ImageLoaderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_path": ("STRING", {"default": "", "multiline": False, "display": "图片路径"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "load_image"
    CATEGORY = "WX/图像"
    OUTPUT_NODE = True  # 允许在UI中预览

    def load_image(self, image_path):
        # 检查文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        # 加载图片
        image = Image.open(image_path).convert("RGB")
        
        # 转换为numpy数组并归一化到0-1范围
        image_np = np.array(image).astype(np.float32) / 255.0
        
        # 转换为tensor (H, W, C) -> (1, H, W, C)
        image_tensor = torch.from_numpy(image_np)[None,]
        
        # 保存预览图片到输出目录
        output_dir = folder_paths.get_output_directory()
        preview_filename = os.path.basename(image_path)
        preview_path = os.path.join(output_dir, preview_filename)
        
        # 如果预览文件不存在，则复制原图作为预览
        if not os.path.exists(preview_path):
            image.save(preview_path)
        
        # 返回结果和UI预览数据
        return {
            "ui": {
                "images": [{
                    "filename": preview_filename,
                    "type": "output",
                    "subfolder": ""
                }]
            },
            "result": (image_tensor,)
        }

# class ImageLoadFromBase64:
#     def __init__(self):
#         pass

#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "base64_string": ("STRING", {}),
#             }
#         }

#     RETURN_TYPES = ("IMAGE", "MASK")
#     # RETURN_NAMES = ("any")

#     FUNCTION = "main"

#     # OUTPUT_NODE = False

#     CATEGORY = "image_io_helpers"

#     def main(self, base64_string: str):
#         # Remove the base64 prefix (e.g., "data:image/png;base64,")
#         if (base64_string.startswith("data:image/")):
#             _, base64_string = base64_string.split(",", 1)
#         decoded_bytes = base64.b64decode(base64_string)
#         file_like_object = BytesIO(decoded_bytes)
#         img = Image.open(file_like_object)

#         output_images = []
#         output_masks = []
#         for i in ImageSequence.Iterator(img):
#             i = ImageOps.exif_transpose(i)
#             image = i.convert("RGB")
#             image = np.array(image).astype(np.float32) / 255.0
#             image = torch.from_numpy(image)[None,]
#             if 'A' in i.getbands():
#                 mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
#                 mask = 1. - torch.from_numpy(mask)
#             else:
#                 mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
#             output_images.append(image)
#             output_masks.append(mask.unsqueeze(0))

#         if len(output_images) > 1:
#             output_image = torch.cat(output_images, dim=0)
#             output_mask = torch.cat(output_masks, dim=0)
#         else:
#             output_image = output_images[0]
#             output_mask = output_masks[0]

#         return (output_image, output_mask)


# class ImageLoadByPath:
#     def __init__(self):
#         pass

#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "file_path": ("STRING", {}),
#             }
#         }

#     RETURN_TYPES = ("IMAGE", "MASK")
#     # RETURN_NAMES = ("any")

#     FUNCTION = "main"

#     # OUTPUT_NODE = False

#     CATEGORY = "image_io_helpers"

#     def main(self, file_path: str):
#         img = Image.open(file_path)
#         output_images = []
#         output_masks = []
#         for i in ImageSequence.Iterator(img):
#             i = ImageOps.exif_transpose(i)
#             image = i.convert("RGB")
#             image = np.array(image).astype(np.float32) / 255.0
#             image = torch.from_numpy(image)[None,]
#             if 'A' in i.getbands():
#                 mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
#                 mask = 1. - torch.from_numpy(mask)
#             else:
#                 mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
#             output_images.append(image)
#             output_masks.append(mask.unsqueeze(0))

#         if len(output_images) > 1:
#             output_image = torch.cat(output_images, dim=0)
#             output_mask = torch.cat(output_masks, dim=0)
#         else:
#             output_image = output_images[0]
#             output_mask = output_masks[0]

#         return (output_image, output_mask)


# class ImageLoadAsMaskByPath:
#     _color_channels = ["alpha", "red", "green", "blue"]

#     def __init__(self):
#         pass

#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "file_path": ("STRING", {}),
#                 "channel": (cls._color_channels,)
#             }
#         }

#     RETURN_TYPES = ("MASK",)
#     # RETURN_NAMES = ("any")

#     FUNCTION = "main"

#     # OUTPUT_NODE = False

#     CATEGORY = "image_io_helpers"

#     def main(self, file_path: str, channel):
#         i = Image.open(file_path)
#         i = ImageOps.exif_transpose(i)
#         if i.getbands() != ("R", "G", "B", "A"):
#             i = i.convert("RGBA")
#         mask = None
#         c = channel[0].upper()
#         if c in i.getbands():
#             mask = np.array(i.getchannel(c)).astype(np.float32) / 255.0
#             mask = torch.from_numpy(mask)
#             if c == 'A':
#                 mask = 1. - mask
#         else:
#             mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
#         return (mask.unsqueeze(0),)


# class ImageSaveToPath:
#     def __init__(self):
#         self.type = "output"

#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "images": ("IMAGE", {}),
#                 "folder_path": ("STRING", {}),
#                 "filename_prefix": ("STRING", {
#                     "default": "ComfyUI"
#                 }),
#                 "save_prompt": ("BOOLEAN", {
#                     "default": True,
#                 }),
#                 "save_extra_pnginfo": ("BOOLEAN", {
#                     "default": True,
#                 }),
#                 "compress_level": ("INT", {
#                     "default": 4,
#                     "min": 0,
#                     "max": 9,
#                     "step": 1
#                 })
#             },
#             "hidden": {
#                 "prompt": "PROMPT",
#                 "extra_pnginfo": "EXTRA_PNGINFO"
#             },
#         }

#     RETURN_TYPES = ()

#     FUNCTION = "main"

#     OUTPUT_NODE = True

#     CATEGORY = "image_io_helpers"

#     def main(
#             self,
#             images: Tensor,
#             folder_path: str,
#             file_name: str,
#             prompt=None,
#             save_prompt=True,
#             extra_pnginfo=None,
#             save_extra_pnginfo=True,
#             compress_level=4):
#         file_paths = []
#         with os.scandir(folder_path) as entries:
#             for entry in entries:
#                 if entry.is_file():
#                     file_paths.append(entry.path)
#         png_paths = filter(lambda x: x.endswith(".png"), file_paths)
#         counter = 0
#         results = list()
#         for image in images:
#             i = 255. * image.cpu().numpy()
#             img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
#             metadata = None
#             if not args.disable_metadata:
#                 metadata = PngInfo()
#                 if save_prompt and prompt is not None:
#                     metadata.add_text("prompt", json.dumps(prompt))
#                 if save_extra_pnginfo and extra_pnginfo is not None:
#                     for x in extra_pnginfo:
#                         metadata.add_text(x, json.dumps(extra_pnginfo[x]))

#             file = f"{file_name}_{counter:05}.png"
#             full_file_path = os.path.join(folder_path, file)

#             while (full_file_path in png_paths):
#                 counter += 1
#                 file = f"{file_name}_{counter:05}.png"
#                 full_file_path = os.path.join(folder_path, file)

#             img.save(full_file_path, pnginfo=metadata, compress_level=compress_level)
#             results.append({
#                 "filename": file,
#                 "folder": folder_path,
#                 "full_path": full_file_path,
#                 "type": self.type
#             })
#             counter += 1

#         return {"ui": {"images": results}}


# class ImageSaveAsBase64:
#     def __init__(self):
#         self.type = "output"

#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "images": ("IMAGE", {}),
#                 "save_prompt": ("BOOLEAN", {
#                     "default": True,
#                 }),
#                 "save_extra_pnginfo": ("BOOLEAN", {
#                     "default": True,
#                 }),
#             },
#             "hidden": {
#                 "prompt": "PROMPT",
#                 "extra_pnginfo": "EXTRA_PNGINFO"
#             },
#         }

#     RETURN_TYPES = ()

#     FUNCTION = "main"

#     OUTPUT_NODE = True

#     CATEGORY = "image_io_helpers"

#     def main(
#             self,
#             images: Tensor,
#             save_prompt=True,
#             save_extra_pnginfo=True,
#             prompt=None,
#             extra_pnginfo=None):

#         results = list()
#         for image in images:
#             i = 255. * image.cpu().numpy()
#             img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
#             metadata = None
#             if not args.disable_metadata:
#                 metadata = PngInfo()
#                 if save_prompt and prompt is not None:
#                     metadata.add_text("prompt", json.dumps(prompt))
#                 if save_extra_pnginfo and extra_pnginfo is not None:
#                     for x in extra_pnginfo:
#                         metadata.add_text(x, json.dumps(extra_pnginfo[x]))

#             # Create a BytesIO object to simulate a file-like object
#             image_stream = BytesIO()

#             # Save the image to the BytesIO stream
#             img.save(image_stream, pnginfo=metadata, format="PNG")

#             # Get raw bytes from the buffer
#             image_bytes = image_stream.getvalue()

#             # Encode the BytesIO stream content to base64
#             base64_string = "data:image/png;base64," + base64.b64encode(image_bytes).decode(
#                 "utf-8")  # Decode for text representation

#             results.append({
#                 "base64_string": base64_string,
#             })

#         return {"ui": {"images": results}}

NODE_CLASS_MAPPINGS = {
    'WxSaveImageExtended': WxSaveImageExtended,
    "ImageLoaderNode": ImageLoaderNode,
    # 'ImageLoadFromBase64(ImageIOHelpers)': ImageLoadFromBase64,
    # 'ImageLoadByPath(ImageIOHelpers)': ImageLoadByPath,
    # 'ImageLoadAsMaskByPath(ImageIOHelpers)': ImageLoadAsMaskByPath,
    # 'ImageSaveToPath(ImageIOHelpers)': ImageSaveToPath,
    # 'ImageSaveAsBase64(ImageIOHelpers)': ImageSaveAsBase64,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'WxSaveImageExtended': '图像保存(含信息)|WX',
    "ImageLoaderNode": "图片加载（从图片路径）|WX",
}
