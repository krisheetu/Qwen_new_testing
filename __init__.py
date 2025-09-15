import gc
import os.path as osp
import torch
import folder_paths
import comfy.model_management as mm

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()
torch_dtype = torch.bfloat16
now_dir = osp.dirname(__file__)
aifsh_dir = osp.join(folder_paths.models_dir,"AIFSH")

import random
import numpy as np
from PIL import Image,ImageFont,ImageDraw
from huggingface_hub import snapshot_download
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

class LoadQwenImageDiffSynthiPipe:

    def __init__(self):
        pipe_path = osp.join(aifsh_dir,"Qwen-Image")
        if not osp.exists(osp.join(pipe_path,"vae/diffusion_pytorch_model.safetensors")):
            snapshot_download(repo_id="Qwen/Qwen-Image",local_dir=pipe_path)
        self.pipe_path = pipe_path

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "offload":("BOOLEAN",{
                    "default":True
                }),
                "fp8_quantization":("BOOLEAN",{
                    "default":False
                }),
            },
            "optional":{
                "lora":(folder_paths.get_filename_list("loras"),),
                "lora_alpha":("FLOAT",{
                    "default":1.0
                })
            }
        }
    
    RETURN_TYPES = ("QwenImageDiffSynthiPipe",)
    RETURN_NAMES = ("pipe",)

    FUNCTION = "load_pipe"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH/QwenImageDiffSynth"

    def load_pipe(self,offload,fp8_quantization,lora=None,lora_alpha=1.0):
        pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(model_id=self.pipe_path,
                            offload_device="cpu" if offload else None,
                            offload_dtype=torch.float8_e4m3fn if fp8_quantization else None,
                            origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                            skip_download=True),
                ModelConfig(model_id=self.pipe_path,
                            offload_device="cpu" if offload else None,
                            offload_dtype=torch.float8_e4m3fn if fp8_quantization else None,
                            origin_file_pattern="text_encoder/model*.safetensors",
                            skip_download=True),
                ModelConfig(model_id=self.pipe_path,
                            offload_device="cpu" if offload else None,
                            offload_dtype=torch.float8_e4m3fn if fp8_quantization else None,
                            origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                            skip_download=True),
            ],
            tokenizer_config=ModelConfig(model_id=self.pipe_path, origin_file_pattern="tokenizer/",
                                         skip_download=True),
        )
        if lora is not None:
            pipe.load_lora(pipe.dit,
                           path=folder_paths.get_full_path_or_raise("loras",lora),
                           alpha=lora_alpha)
        pipe.enable_vram_management()

        return (pipe, )

class SetEligenArgs:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "mask":("IMAGE",),
                "prompt":("STRING",),
            },
        }
    RETURN_TYPES = ("EligenArgs","IMAGE",)
    RETURN_NAMES = ("eligen_args","mask",)

    FUNCTION = "set_args"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH/QwenImageDiffSynth"

    def set_args(self,mask,prompt):
        eligen_args = dict(masks=[comfy2pil(mask)],prompts=[prompt])
        mask = visualize_masks(comfy2pil(mask),masks=eligen_args['masks'],
                               mask_prompts=eligen_args['prompts'])
        return (eligen_args,pil2comfy(mask),)

class EligenArgsConcat:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "a_eligen_args":("EligenArgs",),
                "b_eligen_args":("EligenArgs",),
            },
            "optional":{
                "c_eligen_args":("EligenArgs",),
            }
        }
    RETURN_TYPES = ("EligenArgs","IMAGE",)
    RETURN_NAMES = ("eligen_args","mask",)

    FUNCTION = "set_args"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH/QwenImageDiffSynth"

    def set_args(self,a_eligen_args,b_eligen_args,c_eligen_args=None):
        masks=a_eligen_args["masks"]+b_eligen_args["masks"]
        prompts=a_eligen_args["prompts"]+b_eligen_args["prompts"]
        if c_eligen_args is not None:
            masks += c_eligen_args["masks"]
            prompts += c_eligen_args["prompts"]
        eligen_args = dict(masks=masks, prompts=prompts)
        
        empty_pil = Image.new("RGB",size=eligen_args['masks'][0].size,color=0)
        mask = visualize_masks(empty_pil,masks=masks,mask_prompts=prompts)
        
        return (eligen_args,pil2comfy(mask),)


class QwenImageRatio2Size:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "aspect_ratio":(["1:1","16:9","9:16","4:3","3:4"],)
            }
        }
    
    RETURN_TYPES = ("INT","INT",)
    RETURN_NAMES = ("width","height",)

    FUNCTION = "get_image_size"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH/QwenImageDiffSynth"

    # (1664, 928), (1472, 1140), (1328, 1328)
    def get_image_size(self,aspect_ratio):
        if aspect_ratio == "1:1":
            return (1328, 1328,)
        elif aspect_ratio == "16:9":
            return (1664, 928,)
        elif aspect_ratio == "9:16":
            return (928, 1664,)
        elif aspect_ratio == "4:3":
            return (1472, 1140,)
        elif aspect_ratio == "3:4":
            return (1140, 1472,)
        else:
            return (1328, 1328,)
        

class QwenImageDiffSynthSampler:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "pipe":("QwenImageDiffSynthiPipe",),
                "prompt":("STRING",),
                "negative_prompt":("STRING",),
                "width":("INT",{
                    "default":982
                }),
                "height":("INT",{
                    "default":1664
                }),
                "num_inference_steps":("INT",{
                    "default":30
                }),
                "guidance_scale":("FLOAT",{
                    "default":4,
                }),
                "seed":("INT",{
                    "default":42
                }),
            },
            "optional":{
                "eligen_args":("EligenArgs",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "sample"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH/QwenImageDiffSynth"

    def sample(self,pipe,prompt,negative_prompt,
               width,height,num_inference_steps,
               guidance_scale,seed,eligen_args=None):
        if eligen_args is None:
            eligen_args = dict(masks=None,prompts=None)
        masks = eligen_args["masks"]
        prompts = eligen_args["prompts"]
        image = pipe(
            prompt=prompt,
            cfg_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            seed=seed,
            height=height,
            width=width,
            eligen_entity_prompts=prompts,
            eligen_entity_masks=masks,
        )
        
        return (pil2comfy(image),)
    

def comfy2pil(image):
    i = 255. * image.cpu().numpy()[0]
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    return img
    
def pil2comfy(pil):
    # image = pil.convert("RGB")
    image = np.array(pil).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    return image

def visualize_masks(image, masks, mask_prompts,font_size=35, use_random_colors=False):
    # Create a blank image for overlays
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))

    colors = [
        (165, 238, 173, 80),
        (76, 102, 221, 80),
        (221, 160, 77, 80),
        (204, 93, 71, 80),
        (145, 187, 149, 80),
        (134, 141, 172, 80),
        (157, 137, 109, 80),
        (153, 104, 95, 80),
        (165, 238, 173, 80),
        (76, 102, 221, 80),
        (221, 160, 77, 80),
        (204, 93, 71, 80),
        (145, 187, 149, 80),
        (134, 141, 172, 80),
        (157, 137, 109, 80),
        (153, 104, 95, 80),
    ]
    # Generate random colors for each mask
    if use_random_colors:
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 80) for _ in range(len(masks))]

    # Font settings
    try:
        font = ImageFont.truetype(osp.join(now_dir,"font/Arial-Unicode-Regular.ttf"), font_size)  # Adjust as needed
    except IOError:
        font = ImageFont.load_default(font_size)

    # Overlay each mask onto the overlay image
    for mask, mask_prompt, color in zip(masks, mask_prompts, colors):
        # Convert mask to RGBA mode
        mask_rgba = mask.convert('RGBA')
        mask_data = mask_rgba.getdata()
        new_data = [(color if item[:3] == (255, 255, 255) else (0, 0, 0, 0)) for item in mask_data]
        mask_rgba.putdata(new_data)

        # Draw the mask prompt text on the mask
        draw = ImageDraw.Draw(mask_rgba)
        mask_bbox = mask.getbbox()  # Get the bounding box of the mask
        text_position = (mask_bbox[0] + 10, mask_bbox[1] + 10)  # Adjust text position based on mask position
        draw.text(text_position, mask_prompt, fill=(255, 255, 255, 255), font=font)

        # Alpha composite the overlay with this mask
        overlay = Image.alpha_composite(overlay, mask_rgba)

    # Composite the overlay onto the original image
    result = Image.alpha_composite(image.convert('RGBA'), overlay)

    return result


NODE_CLASS_MAPPINGS = {
    "LoadQwenImageDiffSynthiPipe": LoadQwenImageDiffSynthiPipe,
    "QwenImageDiffSynthSampler":QwenImageDiffSynthSampler,
    "QwenImageRatio2Size":QwenImageRatio2Size,
    "SetEligenArgs":SetEligenArgs,
    "EligenArgsConcat":EligenArgsConcat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadQwenImageDiffSynthiPipe": "LoadQwenImageDiffSynthiPipe@关注超级面爸微信公众号",
    "QwenImageDiffSynthSampler":"QwenImageDiffSynthSampler@关注超级面爸微信公众号",
    "QwenImageRatio2Size":"QwenImageRatio2Size@关注超级面爸微信公众号",
    "SetEligenArgs":"SetEligenArgs@关注超级面爸微信公众号",
    "EligenArgsConcat":"EligenArgsConcat@关注超级面爸微信公众号",
}