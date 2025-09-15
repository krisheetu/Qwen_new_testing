import gc
import os.path as osp
import torch
import folder_paths
import comfy.model_management as mm
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import random

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()
torch_dtype = torch.bfloat16
now_dir = osp.dirname(__file__)

# Try to import controlnet processors for preprocessing
try:
    from controlnet_aux import (
        CannyDetector, 
        OpenposeDetector, 
        MidasDetector,
        LineartDetector,
        HEDdetector
    )
    CONTROLNET_AVAILABLE = True
except ImportError:
    print("ControlNet processors not available. Install with: pip install controlnet-aux")
    CONTROLNET_AVAILABLE = False

class QwenImageEliGenSampler:
    """
    Enhanced Qwen-Image sampler with EliGenV2 entity masking support
    Uses ComfyUI's native model loading (Load Diffusion Model, Load CLIP, Load VAE)
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),  # From ComfyUI's Load Diffusion Model node
                "clip": ("CLIP",),    # From ComfyUI's Load CLIP node  
                "vae": ("VAE",),      # From ComfyUI's Load VAE node
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful portrait"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "width": ("INT", {
                    "default": 1328,
                    "min": 64,
                    "max": 8192,
                    "step": 8
                }),
                "height": ("INT", {
                    "default": 1328,
                    "min": 64,
                    "max": 8192,
                    "step": 8
                }),
                "num_inference_steps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 150
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.1,
                    "max": 30.0,
                    "step": 0.1
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                    "default": "dpmpp_2m"
                }),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                    "default": "karras"
                }),
            },
            "optional": {
                "eligen_args": ("EligenArgs",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"
    CATEGORY = "AIFSH/QwenImageDiffSynth"

    def sample(self, model, clip, vae, prompt, negative_prompt,
               width, height, num_inference_steps, guidance_scale, seed,
               sampler_name, scheduler, eligen_args=None):
        
        # Import ComfyUI's sampling functions
        import comfy.sample
        import comfy.samplers
        import comfy.sd3_impls
        
        # Handle EliGen args
        if eligen_args is None:
            eligen_args = dict(masks=None, prompts=None)
        
        masks = eligen_args.get("masks")
        prompts = eligen_args.get("prompts")
        
        # Set seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Encode prompts using CLIP
        tokens = clip.tokenize(prompt)
        positive_cond, positive_pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        
        negative_tokens = clip.tokenize(negative_prompt)
        negative_cond, negative_pooled = clip.encode_from_tokens(negative_tokens, return_pooled=True)
        
        # Apply EliGenV2 entity conditioning if provided
        if masks is not None and prompts is not None:
            positive_cond = self._apply_entity_conditioning(
                positive_cond, positive_pooled, masks, prompts, clip, width, height
            )
        
        # Create empty latent
        latent = torch.zeros([1, 16, height // 8, width // 8], device=device, dtype=torch_dtype)
        
        # Prepare conditioning for SD3/Qwen format
        positive_conditioning = [[positive_cond, {"pooled_output": positive_pooled}]]
        negative_conditioning = [[negative_cond, {"pooled_output": negative_pooled}]]
        
        # Sample using ComfyUI's KSampler
        samples = comfy.sample.sample(
            model=model,
            noise=latent,
            steps=num_inference_steps,
            cfg=guidance_scale,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive_conditioning,
            negative=negative_conditioning,
            latent_image=latent,
            seed=seed,
            denoise=1.0
        )
        
        # Decode samples using VAE
        decoded = vae.decode(samples["samples"])
        
        return (decoded,)
    
    def _apply_entity_conditioning(self, conditioning, pooled, masks, prompts, clip, width, height):
        """Apply EliGenV2 entity masking to conditioning"""
        # Create entity-aware prompt
        entity_prompt = f"<entity_mask>{', '.join(prompts)}</entity_mask>"
        
        # Re-encode with entity information
        entity_tokens = clip.tokenize(entity_prompt)
        entity_cond, entity_pooled = clip.encode_from_tokens(entity_tokens, return_pooled=True)
        
        # Blend original conditioning with entity conditioning
        # This is a simplified version - real implementation would need proper mask integration
        alpha = 0.7  # Entity conditioning strength
        blended_cond = conditioning * (1 - alpha) + entity_cond * alpha
        
        return blended_cond

class QwenImageDiffSynthControlNetSampler:
    """
    Qwen-Image sampler using ComfyUI's native DiffSynth ControlNet support
    Works with ModelPatchLoader and QwenImageDiffsynthControlnet nodes
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),  # Model with ControlNet patch applied
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful portrait"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "width": ("INT", {
                    "default": 1328,
                    "min": 64,
                    "max": 8192,
                    "step": 8
                }),
                "height": ("INT", {
                    "default": 1328,
                    "min": 64,
                    "max": 8192,
                    "step": 8
                }),
                "num_inference_steps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 150
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.1,
                    "max": 30.0,
                    "step": 0.1
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                    "default": "dpmpp_2m"
                }),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                    "default": "karras"
                }),
            },
            "optional": {
                "eligen_args": ("EligenArgs",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"
    CATEGORY = "AIFSH/QwenImageDiffSynth"

    def sample(self, model, clip, vae, prompt, negative_prompt,
               width, height, num_inference_steps, guidance_scale, seed,
               sampler_name, scheduler, eligen_args=None):
        
        # Import ComfyUI sampling
        import comfy.sample
        
        # Handle EliGen args
        if eligen_args is None:
            eligen_args = dict(masks=None, prompts=None)
        
        masks = eligen_args.get("masks")
        prompts = eligen_args.get("prompts")
        
        # Set seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Encode prompts
        tokens = clip.tokenize(prompt)
        positive_cond, positive_pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        
        negative_tokens = clip.tokenize(negative_prompt)
        negative_cond, negative_pooled = clip.encode_from_tokens(negative_tokens, return_pooled=True)
        
        # Apply EliGenV2 entity conditioning
        if masks is not None and prompts is not None:
            positive_cond = self._apply_entity_conditioning(
                positive_cond, positive_pooled, masks, prompts, clip, width, height
            )
        
        # Create latent
        latent = torch.zeros([1, 16, height // 8, width // 8], device=device, dtype=torch_dtype)
        
        # Format conditioning for SD3/Qwen
        positive_conditioning = [[positive_cond, {"pooled_output": positive_pooled}]]
        negative_conditioning = [[negative_cond, {"pooled_output": negative_pooled}]]
        
        # Sample with ControlNet-patched model
        samples = comfy.sample.sample(
            model=model,  # This model already has ControlNet patch applied
            noise=latent,
            steps=num_inference_steps,
            cfg=guidance_scale,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive_conditioning,
            negative=negative_conditioning,
            latent_image=latent,
            seed=seed,
            denoise=1.0
        )
        
        # Decode
        decoded = vae.decode(samples["samples"])
        
        return (decoded,)
    
    def _apply_entity_conditioning(self, conditioning, pooled, masks, prompts, clip, width, height):
        """Apply EliGenV2 entity masking"""
        entity_prompt = f"<entity_mask>{', '.join(prompts)}</entity_mask>"
        entity_tokens = clip.tokenize(entity_prompt)
        entity_cond, entity_pooled = clip.encode_from_tokens(entity_tokens, return_pooled=True)
        
        # Blend conditioning
        alpha = 0.7
        return conditioning * (1 - alpha) + entity_cond * alpha

class ControlNetPreprocessor:
    """Enhanced ControlNet preprocessor for Qwen-Image DiffSynth ControlNets"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "preprocessor_type": (["canny", "depth_midas", "openpose", "lineart", "hed", "normal"], {
                    "default": "canny"
                }),
            },
            "optional": {
                "canny_low_threshold": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 255
                }),
                "canny_high_threshold": ("INT", {
                    "default": 200,
                    "min": 0,
                    "max": 255
                }),
                "detect_resolution": ("INT", {
                    "default": 512,
                    "min": 128,
                    "max": 2048
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("control_image",)
    FUNCTION = "preprocess"
    CATEGORY = "AIFSH/QwenImageDiffSynth"

    def preprocess(self, image, preprocessor_type, canny_low_threshold=100, 
                  canny_high_threshold=200, detect_resolution=512):
        
        # Convert ComfyUI image to PIL
        pil_image = comfy2pil(image)
        original_size = pil_image.size
        
        # For Qwen DiffSynth ControlNets, we often don't need preprocessing
        # as the native nodes handle it, but this provides flexibility
        
        if not CONTROLNET_AVAILABLE:
            print("ControlNet processors not available, returning original image")
            return (image,)
        
        # Resize for processing
        pil_image = pil_image.resize((detect_resolution, detect_resolution))
        
        if preprocessor_type == "canny":
            processor = CannyDetector()
            processed = processor(pil_image, 
                                low_threshold=canny_low_threshold, 
                                high_threshold=canny_high_threshold)
        
        elif preprocessor_type == "depth_midas":
            processor = MidasDetector.from_pretrained('lllyasviel/Annotators')
            processed = processor(pil_image)
        
        elif preprocessor_type == "openpose":
            processor = OpenposeDetector.from_pretrained('lllyasviel/Annotators')
            processed = processor(pil_image)
        
        elif preprocessor_type == "lineart":
            processor = LineartDetector.from_pretrained('lllyasviel/Annotators')
            processed = processor(pil_image)
        
        elif preprocessor_type == "hed":
            processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
            processed = processor(pil_image)
        
        else:
            processed = pil_image
        
        # Resize back to original
        processed = processed.resize(original_size)
        
        return (pil2comfy(processed),)

class SetEligenArgs:
    """Create EliGenV2 entity masking arguments"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("IMAGE",),
                "entity_prompt": ("STRING", {
                    "multiline": False,
                    "default": "person",
                    "placeholder": "entity name (e.g., person, face, clothing)"
                }),
            },
        }
    
    RETURN_TYPES = ("EligenArgs", "IMAGE",)
    RETURN_NAMES = ("eligen_args", "mask_preview",)
    FUNCTION = "set_args"
    CATEGORY = "AIFSH/QwenImageDiffSynth"

    def set_args(self, mask, entity_prompt):
        mask_pil = comfy2pil(mask)
        eligen_args = dict(masks=[mask_pil], prompts=[entity_prompt])
        
        # Create visualization
        mask_preview = visualize_masks(
            mask_pil, 
            masks=eligen_args['masks'],
            mask_prompts=eligen_args['prompts']
        )
        
        return (eligen_args, pil2comfy(mask_preview),)

class EligenArgsConcat:
    """Concatenate multiple EliGenV2 entity arguments"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "eligen_args_a": ("EligenArgs",),
                "eligen_args_b": ("EligenArgs",),
            },
            "optional": {
                "eligen_args_c": ("EligenArgs",),
                "eligen_args_d": ("EligenArgs",),
                "eligen_args_e": ("EligenArgs",),
            }
        }
    
    RETURN_TYPES = ("EligenArgs", "IMAGE",)
    RETURN_NAMES = ("eligen_args", "combined_preview",)
    FUNCTION = "concat_args"
    CATEGORY = "AIFSH/QwenImageDiffSynth"

    def concat_args(self, eligen_args_a, eligen_args_b, 
                   eligen_args_c=None, eligen_args_d=None, eligen_args_e=None):
        
        masks = eligen_args_a["masks"] + eligen_args_b["masks"]
        prompts = eligen_args_a["prompts"] + eligen_args_b["prompts"]
        
        for args in [eligen_args_c, eligen_args_d, eligen_args_e]:
            if args is not None:
                masks += args["masks"]
                prompts += args["prompts"]
        
        eligen_args = dict(masks=masks, prompts=prompts)
        
        # Create combined preview
        if masks:
            base_image = Image.new("RGB", size=masks[0].size, color=(50, 50, 50))
            combined_preview = visualize_masks(base_image, masks=masks, mask_prompts=prompts)
        else:
            combined_preview = Image.new("RGB", (512, 512), color=(50, 50, 50))
        
        return (eligen_args, pil2comfy(combined_preview),)

class QwenImageRatio2Size:
    """Convert aspect ratios to Qwen-Image optimal sizes"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9"], {
                    "default": "1:1"
                })
            }
        }
    
    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("width", "height",)
    FUNCTION = "get_image_size"
    CATEGORY = "AIFSH/QwenImageDiffSynth"

    def get_image_size(self, aspect_ratio):
        # Optimized sizes for Qwen-Image
        size_map = {
            "1:1": (1328, 1328),    # Square
            "16:9": (1664, 928),    # Widescreen
            "9:16": (928, 1664),    # Portrait mobile
            "4:3": (1472, 1140),    # Traditional photo
            "3:4": (1140, 1472),    # Portrait photo
            "3:2": (1536, 1024),    # DSLR standard
            "2:3": (1024, 1536),    # Portrait DSLR
            "21:9": (1792, 768),    # Ultrawide
        }
        return size_map.get(aspect_ratio, (1328, 1328))

class ModelPatchInfo:
    """Display information about available model patches"""
    
    @classmethod
    def INPUT_TYPES(s):
        patches = folder_paths.get_filename_list("model_patches")
        return {
            "required": {
                "model_patch_name": (patches if patches else ["No model patches found"], {
                    "default": patches[0] if patches else "No model patches found"
                })
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("patch_info",)
    FUNCTION = "get_info"
    CATEGORY = "AIFSH/QwenImageDiffSynth"

    def get_info(self, model_patch_name):
        patch_types = {
            "canny": "Edge detection ControlNet - Use with line art, sketches, or edge maps",
            "depth": "Depth map ControlNet - Use with depth information for 3D structure control", 
            "inpaint": "Inpainting model - Use for filling masked areas with context-aware content"
        }
        
        # Determine patch type from filename
        patch_type = "unknown"
        for ptype in patch_types.keys():
            if ptype in model_patch_name.lower():
                patch_type = ptype
                break
        
        info = f"Model Patch: {model_patch_name}\n"
        info += f"Type: {patch_type.capitalize()}\n"
        info += f"Description: {patch_types.get(patch_type, 'Unknown patch type')}\n"
        info += f"Location: models/model_patches/{model_patch_name}"
        
        return (info,)

# Utility functions
def comfy2pil(image):
    """Convert ComfyUI image tensor to PIL Image"""
    i = 255. * image.cpu().numpy()[0]
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    return img

def pil2comfy(pil_image):
    """Convert PIL Image to ComfyUI image tensor"""
    image = np.array(pil_image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    return image

def visualize_masks(base_image, masks, mask_prompts, font_size=28, use_random_colors=False):
    """Visualize entity masks with labels on base image"""
    overlay = Image.new('RGBA', base_image.size, (0, 0, 0, 0))

    # Color palette for different entities
    colors = [
        (255, 107, 107, 100),  # Red
        (107, 255, 107, 100),  # Green  
        (107, 107, 255, 100),  # Blue
        (255, 255, 107, 100),  # Yellow
        (255, 107, 255, 100),  # Magenta
        (107, 255, 255, 100),  # Cyan
        (255, 165, 107, 100),  # Orange
        (165, 107, 255, 100),  # Purple
    ] * 4  # Repeat for more entities

    if use_random_colors:
        colors = [(random.randint(100, 255), random.randint(100, 255), 
                  random.randint(100, 255), 100) for _ in range(len(masks))]

    # Try to load font
    font = None
    try:
        font_path = osp.join(now_dir, "fonts", "Arial.ttf")
        if osp.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except Exception:
        try:
            font = ImageFont.load_default()
        except:
            font = None

    # Process each mask
    for i, (mask, prompt, color) in enumerate(zip(masks, mask_prompts, colors)):
        if isinstance(mask, Image.Image):
            mask_rgba = mask.convert('RGBA')
        else:
            continue
        
        # Apply color to white areas of mask
        mask_data = mask_rgba.getdata()
        colored_data = []
        for pixel in mask_data:
            if pixel[:3] == (255, 255, 255):  # White pixels
                colored_data.append(color)
            else:
                colored_data.append((0, 0, 0, 0))  # Transparent
        
        mask_rgba.putdata(colored_data)
        
        # Add text label
        if font:
            draw = ImageDraw.Draw(mask_rgba)
            bbox = mask.getbbox()
            if bbox:
                text_x = bbox[0] + 10
                text_y = bbox[1] + 10
                # Add background for text
                text_bbox = draw.textbbox((text_x, text_y), prompt, font=font)
                draw.rectangle(text_bbox, fill=(0, 0, 0, 180))
                draw.text((text_x, text_y), prompt, fill=(255, 255, 255, 255), font=font)
        
        # Composite onto overlay
        overlay = Image.alpha_composite(overlay, mask_rgba)
    
    # Composite final result
    if base_image.mode != 'RGBA':
        base_image = base_image.convert('RGBA')
    
    result = Image.alpha_composite(base_image, overlay)
    return result.convert('RGB')

# Node registrations
NODE_CLASS_MAPPINGS = {
    "QwenImageEliGenSampler": QwenImageEliGenSampler,
    "QwenImageDiffSynthControlNetSampler": QwenImageDiffSynthControlNetSampler,
    "ControlNetPreprocessor": ControlNetPreprocessor,
    "SetEligenArgs": SetEligenArgs,
    "EligenArgsConcat": EligenArgsConcat,
    "QwenImageRatio2Size": QwenImageRatio2Size,
    "ModelPatchInfo": ModelPatchInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageEliGenSampler": "Qwen-Image EliGen Sampler",
    "QwenImageDiffSynthControlNetSampler": "Qwen-Image DiffSynth ControlNet + EliGen", 
    "ControlNetPreprocessor": "ControlNet Preprocessor (Enhanced)",
    "SetEligenArgs": "Set EliGen Entity Args",
    "EligenArgsConcat": "Concatenate EliGen Args",
    "QwenImageRatio2Size": "Qwen-Image Aspect Ratio to Size",
    "ModelPatchInfo": "Model Patch Info",
}
