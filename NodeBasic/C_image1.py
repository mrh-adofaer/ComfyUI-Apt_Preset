
import torch
import numpy as np
from comfy.utils import common_upscale
from PIL import Image,  ImageFilter
import torch.nn.functional as F
import node_helpers
import cv2
def pil2tensor(image):  #多维度的图像也可以
    np_image = np.array(image).astype(np.float32) / 255.0
    if np_image.ndim == 2:
        np_image = np_image[None, None, ...]
    elif np_image.ndim == 3:
        np_image = np_image[None, ...]
    return torch.from_numpy(np_image)

#region --------batch-------------------------

class Blend: # 调用
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "blend_factor": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "blend_mode": (["normal", "multiply", "screen", "overlay", "soft_light", "difference"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend_images"

    CATEGORY = "image/postprocessing"

    def blend_images(self, image1: torch.Tensor, image2: torch.Tensor, blend_factor: float, blend_mode: str):
        image1, image2 = node_helpers.image_alpha_fix(image1, image2)
        image2 = image2.to(image1.device)
        if image1.shape != image2.shape:
            image2 = image2.permute(0, 3, 1, 2)
            image2 = comfy.utils.common_upscale(image2, image1.shape[2], image1.shape[1], upscale_method='bicubic', crop='center')
            image2 = image2.permute(0, 2, 3, 1)

        blended_image = self.blend_mode(image1, image2, blend_mode)
        blended_image = image1 * (1 - blend_factor) + blended_image * blend_factor
        blended_image = torch.clamp(blended_image, 0, 1)
        return (blended_image,)

    def blend_mode(self, img1, img2, mode):
        if mode == "normal":
            return img2
        elif mode == "multiply":
            return img1 * img2
        elif mode == "screen":
            return 1 - (1 - img1) * (1 - img2)
        elif mode == "overlay":
            return torch.where(img1 <= 0.5, 2 * img1 * img2, 1 - 2 * (1 - img1) * (1 - img2))
        elif mode == "soft_light":
            return torch.where(img2 <= 0.5, img1 - (1 - 2 * img2) * img1 * (1 - img1), img1 + (2 * img2 - 1) * (self.g(img1) - img1))
        elif mode == "difference":
            return img1 - img2
        else:
            raise ValueError(f"Unsupported blend mode: {mode}")

    def g(self, x):
        return torch.where(x <= 0.25, ((16 * x - 12) * x + 4) * x, torch.sqrt(x))


class Mask_transform_sum:
    def __init__(self):
        self.colors = {"white": (255, 255, 255), "black": (0, 0, 0), "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0), "cyan": (0, 255, 255), "magenta": (255, 0, 255)}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bg_mode": (["crop_image","image", "transparent", "white", "black", "red", "green", "blue"],),
                "mask_mode": (["original", "fill", "fill_block", "outline", "outline_block", "circle", "outline_circle"], {"default": "original"}),
                "ignore_threshold": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "outline_thickness": ("INT", {"default": 3, "min": 1, "max": 400, "step": 1}),
                "smoothness": ("INT", {"default": 0, "min": 0, "max": 150, "step": 1}),
                "mask_expand": ("INT", {"default": 0, "min": -500, "max": 1000, "step": 0.1}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
                "mask_min": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 1.0, "step": 0.01}),
                "mask_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "crop_to_mask": ("BOOLEAN", {"default": False}),
                "expand_width": ("INT", {"default": 0, "min": -500, "max": 1000, "step": 1}),
                "expand_height": ("INT", {"default": 0, "min": -500, "max": 1000, "step": 1}),
                "rescale_crop": ("FLOAT", {"default": 1.00, "min": 0.1, "max": 10.0, "step": 0.01}),
                "divisible_by": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}),
            },
            "optional": {"base_image": ("IMAGE",), "mask": ("MASK",)}
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "separate"
    CATEGORY = "Apt_Preset/mask"
    
    def separate(self, bg_mode, mask_mode="fill", 
                 ignore_threshold=100, opacity=1.0, outline_thickness=1, 
                 smoothness=1, mask_expand=0,
                 expand_width=0, expand_height=0, rescale_crop=1.0,
                 tapered_corners=True, mask_min=0.0, mask_max=1.0,
                 base_image=None, mask=None, crop_to_mask=False, divisible_by=8):
        
        if mask is None:
            if base_image is not None:
                combined_image_tensor = base_image
                empty_mask = torch.zeros_like(base_image[:, :, :, 0])
            else:
                empty_mask = torch.zeros(1, 64, 64, dtype=torch.float32)
                combined_image_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (combined_image_tensor, empty_mask)
        
        def tensorMask2cv2img(tensor_mask):
            mask_np = tensor_mask.cpu().numpy().squeeze()
            if len(mask_np.shape) == 3:
                mask_np = mask_np[:, :, 0]
            return (mask_np * 255).astype(np.uint8)
        
        opencv_gray_image = tensorMask2cv2img(mask)
        _, binary_mask = cv2.threshold(opencv_gray_image, 1, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= ignore_threshold:
                filtered_contours.append(contour)
        
        contours_with_positions = []
        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            contours_with_positions.append((x, y, contour))
        contours_with_positions.sort(key=lambda item: (item[1], item[0]))
        sorted_contours = [item[2] for item in contours_with_positions]
        
        final_mask = np.zeros_like(binary_mask)
        c = 0 if tapered_corners else 1
        kernel = np.array([[c, 1, c], [1, 1, 1], [c, 1, c]], dtype=np.uint8)
        
        for contour in sorted_contours[:8]:
            temp_mask = np.zeros_like(binary_mask)
            
            if mask_mode == "original":
                cv2.drawContours(temp_mask, [contour], 0, 255, -1)
                temp_mask = cv2.bitwise_and(opencv_gray_image, temp_mask)
            elif mask_mode == "fill":
                cv2.drawContours(temp_mask, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)
            elif mask_mode == "fill_block":
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(temp_mask, (x, y), (x+w, y+h), (255, 255, 255), thickness=cv2.FILLED)
            elif mask_mode == "outline":
                cv2.drawContours(temp_mask, [contour], 0, (255, 255, 255), thickness=outline_thickness)
            elif mask_mode == "outline_block":
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(temp_mask, (x, y), (x+w, y+h), (255, 255, 255), thickness=outline_thickness)
            elif mask_mode == "circle":
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(temp_mask, center, radius, (255, 255, 255), thickness=cv2.FILLED)
            elif mask_mode == "outline_circle":
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(temp_mask, center, radius, (255, 255, 255), thickness=outline_thickness)
            
            if mask_expand != 0:
                expand_amount = abs(mask_expand)
                if mask_expand > 0:
                    temp_mask = cv2.dilate(temp_mask, kernel, iterations=expand_amount)
                else:
                    temp_mask = cv2.erode(temp_mask, kernel, iterations=expand_amount)
            
            final_mask = cv2.bitwise_or(final_mask, temp_mask)
        
        if smoothness > 0:
            final_mask_pil = Image.fromarray(final_mask)
            final_mask_pil = final_mask_pil.filter(ImageFilter.GaussianBlur(radius=smoothness))
            final_mask = np.array(final_mask_pil)
        
        original_h, original_w = final_mask.shape[:2]
        coords = cv2.findNonZero(final_mask)
        crop_params = None

        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            center_x = x + w / 2.0
            center_y = y + h / 2.0
            max_expand_left = center_x - 0
            max_expand_right = original_w - center_x
            max_expand_top = center_y - 0
            max_expand_bottom = original_h - center_y
            actual_expand_x = min(expand_width, max_expand_left, max_expand_right)
            actual_expand_y = min(expand_height, max_expand_top, max_expand_bottom)
            x_start = int(round(center_x - (w / 2.0) - actual_expand_x))
            x_end = int(round(center_x + (w / 2.0) + actual_expand_x))
            y_start = int(round(center_y - (h / 2.0) - actual_expand_y))
            y_end = int(round(center_y + (h / 2.0) + actual_expand_y))
            x_start = max(0, x_start)
            y_start = max(0, y_start)
            x_end = min(original_w, x_end)
            y_end = min(original_h, y_end)
            width = x_end - x_start
            height = y_end - y_start
            if width % 2 != 0:
                if x_end < original_w:
                    x_end += 1
                elif x_start > 0:
                    x_start -= 1
            if height % 2 != 0:
                if y_end < original_h:
                    y_end += 1
                elif y_start > 0:
                    y_start -= 1
            x_start = max(0, x_start)
            y_start = max(0, y_start)
            x_end = min(original_w, x_end)
            y_end = min(original_h, y_end)
            crop_params = (x_start, y_start, x_end, y_end)
        else:
            crop_params = (0, 0, original_w, original_h)

        if base_image is None:
            base_image_np = np.zeros((original_h, original_w, 3), dtype=np.float32)
        else:
            base_image_np = base_image[0].cpu().numpy() * 255.0
            base_image_np = base_image_np.astype(np.float32)
        
        if crop_to_mask and crop_params is not None:
            x_start, y_start, x_end, y_end = crop_params[:4]
            cropped_final_mask = final_mask[y_start:y_end, x_start:x_end]
            cropped_base_image = base_image_np[y_start:y_end, x_start:x_end].copy()
            
            if rescale_crop != 1.0:
                scaled_w = int(cropped_final_mask.shape[1] * rescale_crop)
                scaled_h = int(cropped_final_mask.shape[0] * rescale_crop)
                cropped_final_mask = cv2.resize(cropped_final_mask, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
                cropped_base_image = cv2.resize(cropped_base_image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            final_mask = cropped_final_mask
            base_image_np = cropped_base_image
        else:
            if base_image_np.shape[:2] != (original_h, original_w):
                base_image_np = cv2.resize(base_image_np, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        h, w = base_image_np.shape[:2]
        background = np.zeros((h, w, 3), dtype=np.float32)
        if bg_mode in self.colors:
            background[:] = self.colors[bg_mode]
        elif bg_mode == "image" and base_image is not None:
            background = base_image_np.copy()
        elif bg_mode == "transparent":
            background = np.zeros((h, w, 3), dtype=np.float32)
        
        if background.shape[:2] != (h, w):
            background = cv2.resize(background, (w, h), interpolation=cv2.INTER_LINEAR)
        
        if bg_mode == "crop_image":
            combined_image = base_image_np.copy()
        elif bg_mode in ["white", "black", "red", "green", "blue", "transparent"]:
            mask_float = final_mask.astype(np.float32) / 255.0
            if mask_float.ndim == 3:
                mask_float = mask_float.squeeze()
            mask_max_val = np.max(mask_float) if np.max(mask_float) > 0 else 1
            mask_float = (mask_float / mask_max_val) * (mask_max - mask_min) + mask_min
            mask_float = np.clip(mask_float, 0.0, 1.0)
            mask_float = mask_float[:, :, np.newaxis]
            combined_image = mask_float * base_image_np + (1 - mask_float) * background
        elif bg_mode == "image":
            combined_image = background.copy()
            mask_float = final_mask.astype(np.float32) / 255.0
            if mask_float.ndim == 3:
                mask_float = mask_float.squeeze()
            mask_max_val = np.max(mask_float) if np.max(mask_float) > 0 else 1
            mask_float = (mask_float / mask_max_val) * (mask_max - mask_min) + mask_min
            mask_float = np.clip(mask_float, 0.0, 1.0)
            color = np.array(self.colors["white"], dtype=np.float32)
            for c in range(3):
                combined_image[:, :, c] = (mask_float * (opacity * color[c] + (1 - opacity) * combined_image[:, :, c]) + 
                                         (1 - mask_float) * combined_image[:, :, c])
        
        combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)
        final_mask = final_mask.astype(np.uint8)
        
        if divisible_by > 1:
            h, w = combined_image.shape[:2]
            new_h = ((h + divisible_by - 1) // divisible_by) * divisible_by
            new_w = ((w + divisible_by - 1) // divisible_by) * divisible_by
            if new_h != h or new_w != w:
                padded_image = np.zeros((new_h, new_w, 3), dtype=combined_image.dtype)
                padded_image[:h, :w, :] = combined_image
                padded_mask = np.zeros((new_h, new_w), dtype=final_mask.dtype)
                padded_mask[:h, :w] = final_mask
                combined_image = padded_image
                final_mask = padded_mask
        
        combined_image_tensor = torch.from_numpy(combined_image).float() / 255.0
        combined_image_tensor = combined_image_tensor.unsqueeze(0)
        final_mask_tensor = torch.from_numpy(final_mask).float() / 255.0
        final_mask_tensor = final_mask_tensor.unsqueeze(0)
        
        return (combined_image_tensor, final_mask_tensor)



class Image_Resize_sum:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 0, "max": 9999, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 0, "max": 9999, "step": 1, }),
                "upscale_method":  (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {"default": "bilinear" }),
                "keep_proportion": (["resize", "stretch", "pad", "pad_edge", "crop"], ),
                "pad_color": (["black", "white", "red", "green", "blue", "gray"], { "default": "black" }),
                "crop_position": (["center", "top", "bottom", "left", "right"], { "default": "center" }),
                "divisible_by": ("INT", { "default": 2, "min": 0, "max": 512, "step": 1, }),
                "pad_mask_remove": ("BOOLEAN", {"default": True,}),
            },
            "optional" : {
                "mask": ("MASK",),
                "get_image_size": ("IMAGE",),
                "mask_stack": ("MASK_STACK2",),
            },

        }

    # 增加了remove_pad_mask输出
    RETURN_TYPES = ("IMAGE", "MASK", "STITCH3", "FLOAT", )
    RETURN_NAMES = ("IMAGE", "mask", "stitch",  "scale_factor", )
    FUNCTION = "resize"
    CATEGORY = "Apt_Preset/image"

    DESCRIPTION = """
    - 输入参数：
    - resize：按比例缩放图像至宽和高的限制范围，保持宽高比，不填充、不裁剪
    - stretch：拉伸图像以完全匹配指定的宽度和高度，保持宽高比、像素扭曲
    - pad：按比例缩放图像后，在目标尺寸内居中放置，用指定颜色填充多余区域
    - pad_edge：与pad类似，但使用图像边缘像素颜色进行填充
    - crop：按目标尺寸比例裁剪原图像，然后缩放到指定尺寸
    - -----------------------  
    - 输出参数：
    - scale_factor：缩放倍率，用于精准还原，可以减少一次缩放导致的模糊
    - remove_pad_mask：移除填充部分的遮罩，保持画布尺寸不变
    """



    def resize(self, image, width, height, keep_proportion, upscale_method, divisible_by, pad_color, crop_position, get_image_size=None, mask=None, mask_stack=None,pad_mask_remove=True):
        if len(image.shape) == 3:
            B, H, W, C = 1, image.shape[0], image.shape[1], image.shape[2]
            original_image = image.unsqueeze(0)
        else:  
            B, H, W, C = image.shape
            original_image = image.clone()
            
        original_H, original_W = H, W

        if width == 0:
            width = W
        if height == 0:
            height = H

        if get_image_size is not None:
            _, height, width, _ = get_image_size.shape
        
        new_width, new_height = width, height
        pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0
        crop_x, crop_y, crop_w, crop_h = 0, 0, W, H
        scale_factor = 1.0
        
        processed_mask = mask
        if mask is not None and mask_stack is not None:
            mask_mode, smoothness, mask_expand, mask_min, mask_max = mask_stack
            
            separated_result = Mask_transform_sum().separate(  
                bg_mode="crop_image", 
                mask_mode=mask_mode,
                ignore_threshold=0, 
                opacity=1, 
                outline_thickness=1, 
                smoothness=smoothness,
                mask_expand=mask_expand,
                expand_width=0, 
                expand_height=0,
                rescale_crop=1.0,
                tapered_corners=True,
                mask_min=mask_min, 
                mask_max=mask_max,
                base_image=image.clone(), 
                mask=mask, 
                crop_to_mask=False,
                divisible_by=1
            )
            processed_mask = separated_result[1]
        
        if keep_proportion == "resize" or keep_proportion.startswith("pad"):
            if width == 0 and height != 0:
                scale_factor = height / H
                new_width = round(W * scale_factor)
                new_height = height
            elif height == 0 and width != 0:
                scale_factor = width / W
                new_width = width
                new_height = round(H * scale_factor)
            elif width != 0 and height != 0:
                scale_factor = min(width / W, height / H)
                new_width = round(W * scale_factor)
                new_height = round(H * scale_factor)

            if keep_proportion.startswith("pad"):
                if crop_position == "center":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top
                elif crop_position == "top":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = 0
                    pad_bottom = height - new_height
                elif crop_position == "bottom":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = height - new_height
                    pad_bottom = 0
                elif crop_position == "left":
                    pad_left = 0
                    pad_right = width - new_width
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top
                elif crop_position == "right":
                    pad_left = width - new_width
                    pad_right = 0
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top

        elif keep_proportion == "crop":
            old_aspect = W / H
            new_aspect = width / height
            
            if old_aspect > new_aspect:
                crop_h = H
                crop_w = round(H * new_aspect)
                scale_factor = height / H
            else:
                crop_w = W
                crop_h = round(W / new_aspect)
                scale_factor = width / W
            
            if crop_position == "center":
                crop_x = (W - crop_w) // 2
                crop_y = (H - crop_h) // 2
            elif crop_position == "top":
                crop_x = (W - crop_w) // 2
                crop_y = 0
            elif crop_position == "bottom":
                crop_x = (W - crop_w) // 2
                crop_y = H - crop_h
            elif crop_position == "left":
                crop_x = 0
                crop_y = (H - crop_h) // 2
            elif crop_position == "right":
                crop_x = W - crop_w
                crop_y = (H - crop_h) // 2

        final_width = new_width
        final_height = new_height
        if divisible_by > 1:
            final_width = final_width - (final_width % divisible_by)
            final_height = final_height - (final_height % divisible_by)
            if new_width != 0:
                scale_factor *= (final_width / new_width)
            if new_height != 0:
                scale_factor *= (final_height / new_height)

        out_image = image.clone()
        out_mask = processed_mask.clone() if processed_mask is not None else None
        padding_mask = None

        if keep_proportion == "crop":
            out_image = out_image.narrow(-2, crop_x, crop_w).narrow(-3, crop_y, crop_h)
            if out_mask is not None:
                out_mask = out_mask.narrow(-1, crop_x, crop_w).narrow(-2, crop_y, crop_h)

        out_image = common_upscale(
            out_image.movedim(-1, 1),
            final_width,
            final_height,
            upscale_method,
            crop="disabled"
        ).movedim(1, -1)

        if out_mask is not None:
            if upscale_method == "lanczos":
                out_mask = common_upscale(
                    out_mask.unsqueeze(1).repeat(1, 3, 1, 1),
                    final_width,
                    final_height,
                    upscale_method,
                    crop="disabled"
                ).movedim(1, -1)[:, :, :, 0]
            else:
                out_mask = common_upscale(
                    out_mask.unsqueeze(1),
                    final_width,
                    final_height,
                    upscale_method,
                    crop="disabled"
                ).squeeze(1)

        # 保存原始out_mask用于创建remove_pad_mask
        original_out_mask = out_mask.clone() if out_mask is not None else None

        if keep_proportion.startswith("pad") and (pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0):
            padded_width = final_width + pad_left + pad_right
            padded_height = final_height + pad_top + pad_bottom
            if divisible_by > 1:
                width_remainder = padded_width % divisible_by
                height_remainder = padded_height % divisible_by
                if width_remainder > 0:
                    extra_width = divisible_by - width_remainder
                    pad_right += extra_width
                    padded_width += extra_width
                if height_remainder > 0:
                    extra_height = divisible_by - height_remainder
                    pad_bottom += extra_height
                    padded_height += extra_height
            
            color_map = {
                "black": "0, 0, 0",
                "white": "255, 255, 255",
                "red": "255, 0, 0",
                "green": "0, 255, 0",
                "blue": "0, 0, 255",
                "gray": "128, 128, 128"
            }
            pad_color_value = color_map[pad_color]
            
            out_image, padding_mask = self.resize_pad(
                out_image,
                pad_left,
                pad_right,
                pad_top,
                pad_bottom,
                0,
                pad_color_value,
                "edge" if keep_proportion == "pad_edge" else "color"
            )
            
            if out_mask is not None:
                out_mask = out_mask.unsqueeze(1).repeat(1, 3, 1, 1).movedim(1, -1)
                out_mask, _ = self.resize_pad(
                    out_mask,
                    pad_left,
                    pad_right,
                    pad_top,
                    pad_bottom,
                    0,
                    pad_color_value,
                    "edge" if keep_proportion == "pad_edge" else "color"
                )
                out_mask = out_mask[:, :, :, 0]
            else:
                out_mask = torch.ones((B, padded_height, padded_width), dtype=out_image.dtype, device=out_image.device)
                out_mask[:, pad_top:pad_top+final_height, pad_left:pad_left+final_width] = 0.0

        if out_mask is None:
            if keep_proportion != "crop":
                out_mask = torch.zeros((out_image.shape[0], out_image.shape[1], out_image.shape[2]), dtype=torch.float32)
            else:
                out_mask = torch.zeros((out_image.shape[0], out_image.shape[1], out_image.shape[2]), dtype=torch.float32)

        if padding_mask is not None:
            composite_mask = torch.clamp(padding_mask + out_mask, 0, 1)
        else:
            composite_mask = out_mask.clone()

        if keep_proportion.startswith("pad") and (pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0):
            # 获取最终尺寸
            final_padded_height, final_padded_width = composite_mask.shape[1], composite_mask.shape[2]

            remove_pad_mask = torch.zeros_like(composite_mask)
            
            if original_out_mask is not None:
                if original_out_mask.shape[1] != final_height or original_out_mask.shape[2] != final_width:
                    resized_original_mask = common_upscale(
                        original_out_mask.unsqueeze(1),
                        final_width,
                        final_height,
                        upscale_method,
                        crop="disabled"
                    ).squeeze(1)
                else:
                    resized_original_mask = original_out_mask
        
                remove_pad_mask[:, pad_top:pad_top+final_height, pad_left:pad_left+final_width] = resized_original_mask
            else:
                remove_pad_mask[:, pad_top:pad_top+final_height, pad_left:pad_left+final_width] = 0.0
        else:
            remove_pad_mask = composite_mask.clone()

        stitch_info = {
            "original_image": original_image,
            "original_shape": (original_H, original_W),
            "resized_shape": (out_image.shape[1], out_image.shape[2]),
            "crop_position": (crop_x, crop_y),
            "crop_size": (crop_w, crop_h),
            "pad_info": (pad_left, pad_right, pad_top, pad_bottom),
            "keep_proportion": keep_proportion,
            "upscale_method": upscale_method,
            "scale_factor": scale_factor,
            "final_size": (final_width, final_height),
            "image_position": (pad_left, pad_top) if keep_proportion.startswith("pad") else (0, 0),
            "has_input_mask": mask is not None,
            "original_mask": mask.clone() if mask is not None else None
        }
        
        scale_factor = 1/scale_factor

        if pad_mask_remove:
           Fina_mask =  remove_pad_mask.cpu()
        else:
           Fina_mask =  composite_mask.cpu()

        return (out_image.cpu(), Fina_mask, stitch_info, scale_factor, )


    def resize_pad(self, image, left, right, top, bottom, extra_padding, color, pad_mode, mask=None, target_width=None, target_height=None):
        B, H, W, C = image.shape

        if mask is not None:
            BM, HM, WM = mask.shape
            if HM != H or WM != W:
                mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest-exact').squeeze(1)

        bg_color = [int(x.strip()) / 255.0 for x in color.split(",")]
        if len(bg_color) == 1:
            bg_color = bg_color * 3
        bg_color = torch.tensor(bg_color, dtype=image.dtype, device=image.device)

        # 新增逻辑：判断是否需要跳过缩放
        should_skip_resize = False
        if target_width is not None and target_height is not None:
            # 判断长边是否已经等于目标尺寸
            current_long_side = max(W, H)
            target_long_side = max(target_width, target_height)
            if current_long_side == target_long_side:
                should_skip_resize = True

        if not should_skip_resize and target_width is not None and target_height is not None:
            if extra_padding > 0:
                image = common_upscale(image.movedim(-1, 1), W - extra_padding, H - extra_padding, "bilinear", "disabled").movedim(1, -1)
                B, H, W, C = image.shape

            pad_left = (target_width - W) // 2
            pad_right = target_width - W - pad_left
            pad_top = (target_height - H) // 2
            pad_bottom = target_height - H - pad_top
        else:
            pad_left = left + extra_padding
            pad_right = right + extra_padding
            pad_top = top + extra_padding
            pad_bottom = bottom + extra_padding

        padded_width = W + pad_left + pad_right
        padded_height = H + pad_top + pad_bottom

        out_image = torch.zeros((B, padded_height, padded_width, C), dtype=image.dtype, device=image.device)
        for b in range(B):
            if pad_mode == "edge":
                top_edge = image[b, 0, :, :]
                bottom_edge = image[b, H-1, :, :]
                left_edge = image[b, :, 0, :]
                right_edge = image[b, :, W-1, :]

                out_image[b, :pad_top, :, :] = top_edge.mean(dim=0)
                out_image[b, pad_top+H:, :, :] = bottom_edge.mean(dim=0)
                out_image[b, :, :pad_left, :] = left_edge.mean(dim=0)
                out_image[b, :, pad_left+W:, :] = right_edge.mean(dim=0)
                out_image[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = image[b]
            else:
                out_image[b, :, :, :] = bg_color.unsqueeze(0).unsqueeze(0)
                out_image[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = image[b]

        padding_mask = torch.ones((B, padded_height, padded_width), dtype=image.dtype, device=image.device)
        for m in range(B):
            padding_mask[m, pad_top:pad_top+H, pad_left:pad_left+W] = 0.0

        return (out_image, padding_mask)

class Image_solo_crop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_mode": (["no_crop", "no_scale_crop", "scale_crop_image", "scale_bj_image"], {"default": "no_scale_crop"}),
                "long_side": ("INT", {"default": 512, "min": 16, "max": 2048, "step": 2}),
                "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {"default": "bilinear"}),
                "expand_width": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 1}),
                "expand_height": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 1}),
                "divisible_by": ("INT", {"default": 2, "min": 0, "max": 128, "step": 2}),

            },
            "optional": {
                "mask": ("MASK",),
                "mask_stack": ("MASK_STACK2",),
                "crop_img_bj": (["image", "white", "black", "red", "green", "blue", "yellow", "cyan", "magenta", "gray"], {"default": "image"}),
                "auto_expand_square": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "Apt_Preset/image"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "MASK", "STITCH2")
    RETURN_NAMES = ("bj_image", "bj_mask", "crop_image", "crop_mask", "stitch")
    FUNCTION = "inpaint_crop"
    DESCRIPTION = """
    - no_scale_crop: 原始裁切图。不支持缩放
    - scale_crop_image: 原始裁切图的长边缩放。
    - scale_bj_image: 背景图的长边缩放。不支持扩展
    - no_crop: 不进行裁剪，仅处理遮罩。
    - auto_expand_square自动扩展正方形，仅no_scale_crop和scale_crop_image模式
    - 遮罩控制: 微调尺寸【目标尺寸相差2~8个像素时】
    """

    def get_mask_bounding_box(self, mask):
        mask_np = (mask[0].cpu().numpy() > 0.5).astype(np.uint8)
        coords = cv2.findNonZero(mask_np)
        if coords is None:
            raise ValueError("Mask is empty")
        x, y, w, h = cv2.boundingRect(coords)
        return w, h, x, y

    def process_resize(self, image, mask, crop_mode, long_side, divisible_by, upscale_method="bilinear"):
        batch_size, img_height, img_width, channels = image.shape
        image_ratio = img_width / img_height
        mask_w, mask_h, mask_x, mask_y = self.get_mask_bounding_box(mask)
        mask_ratio = mask_w / mask_h
        new_width, new_height = img_width, img_height

        if crop_mode == "scale_bj_image":
            if img_width >= img_height:
                new_width = long_side
                new_height = int(new_width / image_ratio)
            else:
                new_height = long_side
                new_width = int(new_height * image_ratio)
        elif crop_mode == "scale_crop_image":
            if mask_w >= mask_h:
                new_mask_width = long_side
                new_mask_height = int(new_mask_width / mask_ratio)
                mask_scale = new_mask_width / mask_w
            else:
                new_mask_height = long_side
                new_mask_width = int(new_mask_height * mask_ratio)
                mask_scale = new_mask_height / mask_h
            new_width = int(img_width * mask_scale)
            new_height = int(img_height * mask_scale)
        elif crop_mode == "no_crop":
            new_width, new_height = img_width, img_height

        if divisible_by > 1:
            if new_width % divisible_by != 0:
                new_width += (divisible_by - new_width % divisible_by)
            if new_height % divisible_by != 0:
                new_height += (divisible_by - new_height % divisible_by)
        else:
            if new_width % 2 != 0:
                new_width += 1
            if new_height % 2 != 0:
                new_height += 1

        torch_upscale_method = upscale_method
        if upscale_method == "lanczos":
            torch_upscale_method = "bicubic"

        image_t = image.permute(0, 3, 1, 2)
        crop_image = F.interpolate(image_t, size=(new_height, new_width), mode=upscale_method, align_corners=False if upscale_method in ["bilinear", "bicubic"] else None)
        crop_image = crop_image.permute(0, 2, 3, 1)

        mask_t = mask.unsqueeze(1) if mask.ndim == 3 else mask
        crop_mask = F.interpolate(mask_t, size=(new_height, new_width), mode="nearest")
        crop_mask = crop_mask.squeeze(1)

        return (crop_image, crop_mask)

    def inpaint_crop(self, image, crop_mode, long_side, upscale_method="bilinear",
                    expand_width=0, expand_height=0, auto_expand_square=False, divisible_by=2,
                    mask=None, mask_stack=None, crop_img_bj="image"):
        colors = {
            "white": (1.0, 1.0, 1.0),
            "black": (0.0, 0.0, 0.0),
            "red": (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
            "yellow": (1.0, 1.0, 0.0),
            "cyan": (0.0, 1.0, 1.0),
            "magenta": (1.0, 0.0, 1.0),
            "gray": (0.5, 0.5, 0.5)
        }
        batch_size, height, width, _ = image.shape
        if mask is None:
            mask = torch.ones((batch_size, height, width), dtype=torch.float32, device=image.device)

        if mask_stack is not None:
            mask_mode, smoothness, mask_expand, mask_min, mask_max = mask_stack
            if hasattr(mask, 'convert'):
                mask_tensor = pil2tensor(mask.convert('L'))
            else:
                if isinstance(mask, torch.Tensor):
                    mask_tensor = mask if len(mask.shape) <= 3 else mask.squeeze(-1) if mask.shape[-1] == 1 else mask
                else:
                    mask_tensor = mask
            separated_result = Mask_transform_sum().separate(
                bg_mode="crop_image",
                mask_mode=mask_mode,
                ignore_threshold=0,
                opacity=1,
                outline_thickness=1,
                smoothness=smoothness,
                mask_expand=mask_expand,
                expand_width=0,
                expand_height=0,
                rescale_crop=1.0,
                tapered_corners=True,
                mask_min=mask_min,
                mask_max=mask_max,
                base_image=image,
                mask=mask_tensor,
                crop_to_mask=False,
                divisible_by=1
            )
            processed_mask = separated_result[1]
        else:
            processed_mask = mask

        crop_image, original_crop_mask = self.process_resize(
            image, processed_mask, crop_mode, long_side, divisible_by, upscale_method)

        # 第一步：先计算auto_expand_square=False时的原始扩展结果（获取基准长边）
        # 1.1 基于原始扩展量计算边界
        orig_expand_w, orig_expand_h = expand_width, expand_height
        ideal_x_new = x - (orig_expand_w // 2) if 'x' in locals() else 0
        ideal_y_new = y - (orig_expand_h // 2) if 'y' in locals() else 0
        ideal_x_end = (x + w + (orig_expand_w // 2)) if 'x' in locals() else 0
        ideal_y_end = (y + h + (orig_expand_h // 2)) if 'y' in locals() else 0

        # 1.2 处理遮罩边界（提前计算，为后续基准长边获取做准备）
        image_np = crop_image[0].cpu().numpy()
        mask_np = original_crop_mask[0].cpu().numpy()
        original_h, original_w = image_np.shape[0], image_np.shape[1]
        coords = cv2.findNonZero((mask_np > 0.5).astype(np.uint8))
        if coords is None:
            raise ValueError("Mask is empty after processing")
        x, y, w, h = cv2.boundingRect(coords)

        # 1.3 计算False时的原始扩展边界
        false_x_new = max(0, x - (orig_expand_w // 2))
        false_y_new = max(0, y - (orig_expand_h // 2))
        false_x_end = min(original_w, x + w + (orig_expand_w // 2))
        false_y_end = min(original_h, y + h + (orig_expand_h // 2))

        # 1.4 处理False时的边界补偿
        if (x - (orig_expand_w // 2)) < 0:
            add = abs(x - (orig_expand_w // 2))
            false_x_end = min(original_w, false_x_end + add)
        elif (x + w + (orig_expand_w // 2)) > original_w:
            add = (x + w + (orig_expand_w // 2)) - original_w
            false_x_new = max(0, false_x_new - add)

        if (y - (orig_expand_h // 2)) < 0:
            add = abs(y - (orig_expand_h // 2))
            false_y_end = min(original_h, false_y_end + add)
        elif (y + h + (orig_expand_h // 2)) > original_h:
            add = (y + h + (orig_expand_h // 2)) - original_h
            false_y_new = max(0, false_y_new - add)

        # 1.5 计算False时的最终尺寸（获取基准长边）
        false_w = false_x_end - false_x_new
        false_h = false_y_end - false_y_new
        false_long_side = max(false_w, false_h)  # 这是auto_expand_square=False时的长边，作为正方形基准

        # 第二步：根据auto_expand_square状态分支处理
        if auto_expand_square and crop_mode in ["no_scale_crop", "scale_crop_image"]:
            # 正方形模式：以False时的长边为目标边长，修正扩展量
            target_square_side = false_long_side
            # 计算需要的总扩展量（目标边长 - 原始遮罩尺寸）
            total_needed_expand_w = target_square_side - w
            total_needed_expand_h = target_square_side - h
            # 分配扩展量（左右/上下均分）
            expand_width = total_needed_expand_w
            expand_height = total_needed_expand_h

            # 重新计算正方形扩展边界
            ideal_x_new = x - (expand_width // 2)
            ideal_y_new = y - (expand_height // 2)
            ideal_x_end = x + w + (expand_width // 2)
            ideal_y_end = y + h + (expand_height // 2)

            # 处理正方形边界限制
            x_new = max(0, ideal_x_new)
            y_new = max(0, ideal_y_new)
            x_end = min(original_w, ideal_x_end)
            y_end = min(original_h, ideal_y_end)

            # 补偿扩展确保边长达标
            if x_new > ideal_x_new:
                x_end = min(original_w, x_end + (ideal_x_new - x_new))
            if x_end < ideal_x_end:
                x_new = max(0, x_new - (ideal_x_end - x_end))
            if y_new > ideal_y_new:
                y_end = min(original_h, y_end + (ideal_y_new - y_new))
            if y_end < ideal_y_end:
                y_new = max(0, y_new - (ideal_y_end - y_end))

            # 最终修正为正方形（确保宽高=目标边长）
            current_w = x_end - x_new
            current_h = y_end - y_new
            if current_w != target_square_side:
                diff = target_square_side - current_w
                x_new = max(0, x_new - (diff // 2))
                x_end = min(original_w, x_end + (diff - (diff // 2)))
            if current_h != target_square_side:
                diff = target_square_side - current_h
                y_new = max(0, y_new - (diff // 2))
                y_end = min(original_h, y_end + (diff - (diff // 2)))

            # 兼容divisible_by要求
            if divisible_by > 1:
                final_side = x_end - x_new
                remainder = final_side % divisible_by
                if remainder != 0:
                    final_side += (divisible_by - remainder)
                    diff = final_side - (x_end - x_new)
                    x_new = max(0, x_new - (diff // 2))
                    x_end = min(original_w, x_end + (diff - (diff // 2)))
                    y_new = max(0, y_new - (diff // 2))
                    y_end = min(original_h, y_end + (diff - (diff // 2)))
            x_end = x_new + (x_end - x_new)
            y_end = y_new + (x_end - x_new)  # 强制高=宽，确保正方形
        else:
            # 非正方形模式：完全沿用False时的原始逻辑结果
            x_new, y_new = false_x_new, false_y_new
            x_end, y_end = false_x_end, false_y_end

            # 原始尺寸修正逻辑
            if divisible_by > 1:
                current_w = x_end - x_new
                remainder_w = current_w % divisible_by
                if remainder_w != 0:
                    if x_end + (divisible_by - remainder_w) <= original_w:
                        x_end += (divisible_by - remainder_w)
                    elif x_new - (divisible_by - remainder_w) >= 0:
                        x_new -= (divisible_by - remainder_w)
                    else:
                        current_w -= remainder_w
                        x_end = x_new + current_w

                current_h = y_end - y_new
                remainder_h = current_h % divisible_by
                if remainder_h != 0:
                    if y_end + (divisible_by - remainder_h) <= original_h:
                        y_end += (divisible_by - remainder_h)
                    elif y_new - (divisible_by - remainder_h) >= 0:
                        y_new -= (divisible_by - remainder_h)
                    else:
                        current_h -= remainder_h
                        y_end = y_new + current_h
            else:
                current_w = x_end - x_new
                if current_w % 2 != 0:
                    if x_end < original_w:
                        x_end += 1
                    elif x_new > 0:
                        x_new -= 1

                current_h = y_end - y_new
                if current_h % 2 != 0:
                    if y_end < original_h:
                        y_end += 1
                    elif y_new > 0:
                        y_new -= 1

        # 最终裁剪尺寸
        current_w = x_end - x_new
        current_h = y_end - y_new

        bj_mask_tensor = original_crop_mask
        bj_image = crop_image.clone()

        if crop_img_bj != "image" and crop_img_bj in colors:
            r, g, b = colors[crop_img_bj]
            h_bg, w_bg, _ = crop_image.shape[1:]
            background = torch.zeros((crop_image.shape[0], h_bg, w_bg, 3), device=crop_image.device)
            background[:, :, :, 0] = r
            background[:, :, :, 1] = g
            background[:, :, :, 2] = b
            if crop_image.shape[3] >= 4:
                alpha = crop_image[:, :, :, 3].unsqueeze(3)
                image_rgb = crop_image[:, :, :, :3]
                crop_image = image_rgb * alpha + background * (1 - alpha)
            else:
                alpha = original_crop_mask.unsqueeze(3)
                image_rgb = crop_image[:, :, :, :3]
                crop_image = image_rgb * alpha + background * (1 - alpha)

        mask_x_start = 0
        mask_y_start = 0
        mask_x_end = 0
        mask_y_end = 0

        if crop_mode == "no_crop":
            cropped_image_tensor = crop_image.clone()
            cropped_mask_tensor = original_crop_mask.clone()
            current_crop_position = (0, 0)
            current_crop_size = (original_w, original_h)
            mask_x_start = x
            mask_y_start = y
            mask_x_end = x + w
            mask_y_end = y + h
        else:
            cropped_image_tensor = crop_image[:, y_new:y_end, x_new:x_end, :].clone()
            cropped_mask_tensor = original_crop_mask[:, y_new:y_end, x_new:x_end].clone()
            mask_x_start = max(0, x - x_new)
            mask_y_start = max(0, y - y_new)
            mask_x_end = min(current_w, (x + w) - x_new)
            mask_y_end = min(current_h, (y + h) - y_new)
            current_crop_position = (x_new, y_new)
            current_crop_size = (current_w, current_h)

        orig_long_side = max(original_w, original_h)
        crop_long_side = max(current_crop_size[0], current_crop_size[1])
        original_image_h, original_image_w = image.shape[1], image.shape[2]
        stitch = {
            "original_shape": (original_h, original_w),
            "original_image_shape": (original_image_h, original_image_w),
            "crop_position": current_crop_position,
            "crop_size": current_crop_size,
            "expand_width": expand_width,
            "expand_height": expand_height,
            "auto_expand_square": auto_expand_square,
            "expanded_region": (x_new, y_new, x_end, y_end),
            "mask_original_position": (x, y, w, h),
            "mask_cropped_position": (mask_x_start, mask_y_start, mask_x_end, mask_y_end),
            "original_long_side": orig_long_side,
            "crop_long_side": crop_long_side,
            "input_long_side": long_side,
            "false_long_side": false_long_side,  # 记录False时的基准长边
            "bj_image": bj_image,
            "original_image": image
        }

        return (bj_image, bj_mask_tensor, cropped_image_tensor, cropped_mask_tensor, stitch)





def create_mask_feather(mask, smoothness):
    if smoothness <= 0:
        return mask.clone() if isinstance(mask, torch.Tensor) else torch.tensor(mask).float()
    if isinstance(mask, torch.Tensor):
        mask_np = mask.squeeze().cpu().detach().numpy()
        device = mask.device
    else:
        mask_np = mask.squeeze()
        device = torch.device("cpu")
    mask_np = (mask_np > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(mask_np, distanceType=cv2.DIST_L2, maskSize=5)
    dist = np.clip(dist, 0, smoothness)
    feather_mask = dist / smoothness
    feather_mask = torch.tensor(feather_mask).float().unsqueeze(0).to(device)
    if isinstance(mask, torch.Tensor) and mask.ndim == 3:
        feather_mask = feather_mask.repeat(mask.shape[0], 1, 1)
    return feather_mask

def create_feather_mask(width, height, feather_size):
    if feather_size <= 0:
        return np.ones((height, width), dtype=np.float32)
    feather = min(feather_size, min(width, height) // 2)
    mask = np.ones((height, width), dtype=np.float32)
    for y in range(feather):
        mask[y, :] = y / feather
    for y in range(height - feather, height):
        mask[y, :] = (height - y) / feather
    for x in range(feather):
        mask[:, x] = np.minimum(mask[:, x], x / feather)
    for x in range(width - feather, width):
        mask[:, x] = np.minimum(mask[:, x], (width - x) / feather)
    return mask

class Image_solo_stitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inpainted_image": ("IMAGE",),
                "mask": ("MASK",),
                "stitch": ("STITCH2",),
                "smoothness": ("INT", {"default": 0, "min": 0, "max": 500, "step": 1, "display": "slider"}),
                "blend_factor": ("FLOAT", {"default": 1.0,"min": 0.0,"max": 1.0,"step": 0.01}),
                "blend_mode": (["normal", "multiply", "screen", "overlay", "soft_light", "difference"],),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "stitch_mode": (["crop_mask", "crop_image"], {"default": "crop_mask"}),
                "recover_method":  (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {"default": "bilinear" }),
            },
        }

    CATEGORY = "Apt_Preset/image"
    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE")
    RETURN_NAMES = ("image","recover_image","original_image")
    FUNCTION = "inpaint_stitch"

    def apply_smooth_blur(self, image, mask, smoothness, bg_color="Alpha"):
        batch_size = image.shape[0]
        result_images = []
        smoothed_masks = []
        color_map = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "gray": (128, 128, 128)
        }
        for i in range(batch_size):
            current_image = image[i].clone()
            current_mask = mask[i] if i < mask.shape[0] else mask[0]
            if smoothness > 0:
                mask_tensor = create_mask_feather(current_mask, smoothness)
            else:
                mask_tensor = current_mask.clone()
            if mask_tensor.dim() == 1:
                mask_tensor = mask_tensor.unsqueeze(0)
            elif mask_tensor.dim() > 2:
                mask_tensor = mask_tensor.squeeze()
                while mask_tensor.dim() > 2:
                    mask_tensor = mask_tensor.squeeze(0)
            smoothed_mask = mask_tensor.clone()
            unblurred_tensor = current_image.clone()
            if current_image.shape[-1] != 3:
                if current_image.shape[-1] == 4:
                    current_image = current_image[:, :, :3]
                    unblurred_tensor = unblurred_tensor[:, :, :3]
                elif current_image.shape[-1] == 1:
                    current_image = current_image.repeat(1, 1, 3)
                    unblurred_tensor = unblurred_tensor.repeat(1, 1, 3)
            mask_expanded = mask_tensor.unsqueeze(-1).repeat(1, 1, 3)
            result_tensor = current_image * mask_expanded + unblurred_tensor * (1 - mask_expanded)
            if bg_color != "Alpha":
                bg_tensor = torch.zeros_like(current_image)
                if bg_color in color_map:
                    r, g, b = color_map[bg_color]
                    bg_tensor[:, :, 0] = r / 255.0
                    bg_tensor[:, :, 1] = g / 255.0
                    bg_tensor[:, :, 2] = b / 255.0
                result_tensor = result_tensor * mask_expanded + bg_tensor * (1 - mask_expanded)
            result_images.append(result_tensor.unsqueeze(0))
            smoothed_masks.append(smoothed_mask.unsqueeze(0))
        final_image = torch.cat(result_images, dim=0)
        final_mask = torch.cat(smoothed_masks, dim=0)
        return (final_image, final_mask)

    def inpaint_stitch(self, inpainted_image, smoothness, mask, stitch, blend_factor, blend_mode, opacity, stitch_mode, recover_method):
        original_h, original_w = stitch["original_shape"]
        crop_x, crop_y = stitch["crop_position"]
        crop_w, crop_h = stitch["crop_size"]
        mask_crop_x, mask_crop_y, mask_crop_x2, mask_crop_y2 = stitch["mask_cropped_position"]
        original_image_h, original_image_w = stitch["original_image_shape"]
        if "bj_image" in stitch:
            bj_image = stitch["bj_image"]
        else:
            bj_image = torch.zeros((1, original_h, original_w, 3), dtype=torch.float32)
        if "original_image" in stitch:
            original_image = stitch["original_image"]
        else:
            original_image = torch.zeros((1, original_image_h, original_image_w, 3), dtype=torch.float32)
        if opacity < 1.0:
            inpainted_image = inpainted_image * opacity
        if inpainted_image.shape[1:3] != mask.shape[1:3]:
            mask = F.interpolate(mask.unsqueeze(1), size=(inpainted_image.shape[1], inpainted_image.shape[2]), mode='nearest').squeeze(1)
        inpainted_np = (inpainted_image[0].cpu().numpy() * 255).astype(np.uint8)
        mask_np = (mask[0].cpu().numpy() * 255).astype(np.uint8)
        background_np = (bj_image[0].cpu().numpy() * 255).astype(np.uint8)
        inpainted_resized = cv2.resize(inpainted_np, (crop_w, crop_h))
        mask_resized = cv2.resize(mask_np, (crop_w, crop_h))
        background_resized = cv2.resize(background_np, (original_w, original_h))
        result = np.zeros((original_h, original_w, 4), dtype=np.uint8)
        result[:, :, :3] = background_resized.copy()
        result[:, :, 3] = 255
        if stitch_mode == "crop_mask":
            inpainted_image, mask = self.apply_smooth_blur(inpainted_image, mask, smoothness, bg_color="Alpha")
            inpainted_blurred = (inpainted_image[0].cpu().numpy() * 255).astype(np.uint8)
            mask_blurred = (mask[0].cpu().numpy() * 255).astype(np.uint8)
            inpainted_blurred = cv2.resize(inpainted_blurred, (crop_w, crop_h))
            mask_blurred = cv2.resize(mask_blurred, (crop_w, crop_h))
            mask_content = mask_blurred[mask_crop_y:mask_crop_y2, mask_crop_x:mask_crop_x2]
            inpaint_content = inpainted_blurred[mask_crop_y:mask_crop_y2, mask_crop_x:mask_crop_x2]
            if mask_content.size == 0 or inpaint_content.size == 0:
                print("Warning: Mask content is empty, returning background image")
                final_image_tensor = torch.from_numpy(background_resized / 255.0).float().unsqueeze(0)
                fimage = Blend().blend_images(bj_image, final_image_tensor, blend_factor, blend_mode)[0]
                recover_img = fimage
                return (fimage, recover_img, original_image)
            paste_x_start = max(0, crop_x + mask_crop_x)
            paste_x_end = min(original_w, crop_x + mask_crop_x2)
            paste_y_start = max(0, crop_y + mask_crop_y)
            paste_y_end = min(original_h, crop_y + mask_crop_y2)
            if paste_x_start >= paste_x_end or paste_y_start >= paste_y_end:
                print("Warning: Invalid paste region, returning background image")
                final_image_tensor = torch.from_numpy(background_resized / 255.0).float().unsqueeze(0)
                fimage = Blend().blend_images(bj_image, final_image_tensor, blend_factor, blend_mode)[0]
                recover_img = fimage
                return (fimage, recover_img, original_image)
            alpha = mask_content / 255.0
            expected_h = paste_y_end - paste_y_start
            expected_w = paste_x_end - paste_x_start
            if alpha.shape[0] != expected_h or alpha.shape[1] != expected_w:
                alpha = cv2.resize(alpha, (expected_w, expected_h))
            alpha = np.expand_dims(alpha, axis=-1)
            background_content = result[paste_y_start:paste_y_end, paste_x_start:paste_x_end, :3]
            if (background_content.shape[0] != alpha.shape[0] or 
                background_content.shape[1] != alpha.shape[1]):
                print("Warning: Dimension mismatch after processing, returning background image")
                final_image_tensor = torch.from_numpy(background_resized / 255.0).float().unsqueeze(0)
                fimage = Blend().blend_images(bj_image, final_image_tensor, blend_factor, blend_mode)[0]
                recover_img = fimage
                return (fimage, recover_img, original_image)
            if (inpaint_content.shape[0] < alpha.shape[0] or 
                inpaint_content.shape[1] < alpha.shape[1]):
                inpaint_content = cv2.resize(inpaint_content, (alpha.shape[1], alpha.shape[0]))
            inpaint_content = inpaint_content[:alpha.shape[0], :alpha.shape[1]]
            if len(inpaint_content.shape) == 3 and inpaint_content.shape[2] > 3:
                inpaint_content = inpaint_content[:, :, :3]
            elif len(inpaint_content.shape) == 2:
                inpaint_content = np.stack([inpaint_content, inpaint_content, inpaint_content], axis=-1)
            if len(background_content.shape) == 2:
                background_content = np.stack([background_content, background_content, background_content], axis=-1)
            elif len(background_content.shape) == 3 and background_content.shape[2] > 3:
                background_content = background_content[:, :, :3]
            try:
                blended = (inpaint_content * alpha + background_content * (1 - alpha)).astype(np.uint8)
                result[paste_y_start:paste_y_end, paste_x_start:paste_x_end, :3] = blended
                result[paste_y_start:paste_y_end, paste_x_start:paste_x_end, 3] = (alpha * 255).astype(np.uint8).squeeze()
            except Exception as e:
                print(f"Warning: Error during blending operation: {e}, returning background image")
                final_image_tensor = torch.from_numpy(background_resized / 255.0).float().unsqueeze(0)
                fimage = Blend().blend_images(bj_image, final_image_tensor, blend_factor, blend_mode)[0]
                recover_img = fimage
                return (fimage, recover_img, original_image)
        else:
            feather_mask = create_feather_mask(crop_w, crop_h, smoothness)
            paste_x_start = max(0, crop_x)
            paste_x_end = min(original_w, crop_x + crop_w)
            paste_y_start = max(0, crop_y)
            paste_y_end = min(original_h, crop_y + crop_h)
            inpaint_content = inpainted_resized[
                max(0, paste_y_start - crop_y) : max(0, paste_y_end - crop_y),
                max(0, paste_x_start - crop_x) : max(0, paste_x_end - crop_x)
            ]
            if inpaint_content.size == 0:
                print("Warning: Inpaint content is empty in crop_image mode, returning background image")
                final_image_tensor = torch.from_numpy(background_resized / 255.0).float().unsqueeze(0)
                fimage = Blend().blend_images(bj_image, final_image_tensor, blend_factor, blend_mode)[0]
                recover_img = fimage
                return (fimage, recover_img, original_image)
            if paste_x_start >= paste_x_end or paste_y_start >= paste_y_end:
                print("Warning: Invalid paste region in crop_image mode, returning background image")
                final_image_tensor = torch.from_numpy(background_resized / 255.0).float().unsqueeze(0)
                fimage = Blend().blend_images(bj_image, final_image_tensor, blend_factor, blend_mode)[0]
                recover_img = fimage
                return (fimage, recover_img, original_image)
            alpha_mask = feather_mask[
                max(0, paste_y_start - crop_y) : max(0, paste_y_end - crop_y),
                max(0, paste_x_start - crop_x) : max(0, paste_x_end - crop_x)
            ]
            alpha = np.expand_dims(alpha_mask, axis=-1)
            background_content = result[paste_y_start:paste_y_end, paste_x_start:paste_x_end, :3]
            if (background_content.shape[0] != alpha.shape[0] or 
                background_content.shape[1] != alpha.shape[1] or
                inpaint_content.shape[0] != alpha.shape[0] or
                inpaint_content.shape[1] != alpha.shape[1]):
                print("Warning: Dimension mismatch in crop_image mode, returning background image")
                final_image_tensor = torch.from_numpy(background_resized / 255.0).float().unsqueeze(0)
                fimage = Blend().blend_images(bj_image, final_image_tensor, blend_factor, blend_mode)[0]
                recover_img = fimage
                return (fimage, recover_img, original_image)
            if len(inpaint_content.shape) == 3 and inpaint_content.shape[2] > 3:
                inpaint_content = inpaint_content[:, :, :3]
            elif len(inpaint_content.shape) == 2:
                inpaint_content = np.stack([inpaint_content, inpaint_content, inpaint_content], axis=-1)
            if len(background_content.shape) == 2:
                background_content = np.stack([background_content, background_content, background_content], axis=-1)
            elif len(background_content.shape) == 3 and background_content.shape[2] > 3:
                background_content = background_content[:, :, :3]
            try:
                blended = (inpaint_content * alpha + background_content * (1 - alpha)).astype(np.uint8)
                result[paste_y_start:paste_y_end, paste_x_start:paste_x_end, :3] = blended
                result[paste_y_start:paste_y_end, paste_x_start:paste_x_end, 3] = (alpha * 255).astype(np.uint8).squeeze()
            except Exception as e:
                print(f"Warning: Error during blending operation in crop_image mode: {e}, returning background image")
                final_image_tensor = torch.from_numpy(background_resized / 255.0).float().unsqueeze(0)
                fimage = Blend().blend_images(bj_image, final_image_tensor, blend_factor, blend_mode)[0]
                recover_img = fimage
                return (fimage, recover_img, original_image)
        final_rgb = result[:, :, :3]
        final_image_tensor = torch.from_numpy(final_rgb / 255.0).float().unsqueeze(0)
        fimage = Blend().blend_images(bj_image, final_image_tensor, blend_factor, blend_mode)[0]
        recover_img, Fina_mask, stitch_info, scale_factor = Image_Resize_sum().resize(
            image=fimage,
            width=original_image_w,
            height=original_image_h,
            keep_proportion="stretch",
            upscale_method=recover_method,
            divisible_by=1,
            pad_color="black",
            crop_position="center",
            get_image_size=None,
            mask=None,
            mask_stack=None,
            pad_mask_remove=True)
        return (fimage, recover_img, original_image)






#endregion----------------裁切组合--------------



class Mask_simple_adjust:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "smoothness": ("INT", {"default": 0, "min": 0, "max": 150, "step": 1}),
                "mask_expand": ("INT", {"default": 0, "min": -500, "max": 1000, "step": 0.1}),
                "is_fill": ("BOOLEAN", {"default": False}),
                "is_invert": ("BOOLEAN", {"default": False}),
                "input_mask": ("MASK",),
                "mask_min": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 1.0, "step": 0.01}),
                "mask_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {}
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("processed_mask",)
    FUNCTION = "process_mask"
    CATEGORY = "Apt_Preset/mask"
    
    def process_mask(self, smoothness=0, mask_expand=0, is_fill=False, is_invert=False, input_mask=None, mask_min=0.0, mask_max=1.0):
        if input_mask is None:
            empty_mask = torch.zeros(1, 64, 64, dtype=torch.float32)
            return (empty_mask,)
        
        def tensorMask2cv2img(tensor_mask):
            mask_np = tensor_mask.cpu().numpy().squeeze()
            if len(mask_np.shape) == 3:
                mask_np = mask_np[:, :, 0]
            return (mask_np * 255).astype(np.uint8)
        
        def cv2img2tensorMask(cv2_mask):
            mask_np = cv2_mask.astype(np.float32) / 255.0
            # 应用mask_min和mask_max调整蒙版动态范围
            mask_max_val = np.max(mask_np) if np.max(mask_np) > 0 else 1.0
            mask_np = (mask_np / mask_max_val) * (mask_max - mask_min) + mask_min
            mask_np = np.clip(mask_np, 0.0, 1.0)
            return torch.from_numpy(mask_np).unsqueeze(0)
        
        opencv_gray_mask = tensorMask2cv2img(input_mask)
        _, binary_mask = cv2.threshold(opencv_gray_mask, 1, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= 1]
        
        final_mask = np.zeros_like(binary_mask)
        expand_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        
        for contour in valid_contours:
            temp_mask = np.zeros_like(binary_mask)
            if is_fill:
                cv2.drawContours(temp_mask, [contour], 0, 255, thickness=cv2.FILLED)
            else:
                cv2.drawContours(temp_mask, [contour], 0, 255, -1)
                temp_mask = cv2.bitwise_and(opencv_gray_mask, temp_mask)
            
            if mask_expand != 0:
                expand_iter = abs(int(mask_expand))
                if mask_expand > 0:
                    temp_mask = cv2.dilate(temp_mask, expand_kernel, iterations=expand_iter)
                else:
                    temp_mask = cv2.erode(temp_mask, expand_kernel, iterations=expand_iter)
            
            final_mask = cv2.bitwise_or(final_mask, temp_mask)
        
        if smoothness > 0:
            mask_pil = Image.fromarray(final_mask)
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=smoothness))
            final_mask = np.array(mask_pil)
        
        if is_invert:
            final_mask = cv2.bitwise_not(final_mask)
            _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
        
        processed_mask_tensor = cv2img2tensorMask(final_mask)
        return (processed_mask_tensor,)
