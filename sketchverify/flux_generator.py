"""
FLUX Control Removal pipeline for background generation
"""
import os
import torch
import numpy as np
from PIL import Image
import cv2
from diffusers.utils import check_min_version
from diffusers import FluxTransformer2DModel

from pipeline_flux_control_removal import FluxControlRemovalPipeline
from utils import composite_images_with_mask

check_min_version("0.30.2")


class FluxControlRemovalGenerator:
    """FLUX Control Removal pipeline for background generation"""

    def __init__(self, device="cuda"):
        self.device = device
        self.pipe = None
        self._load_flux_control_removal()

    def _load_flux_control_removal(self):
        """Load FLUX Control Removal pipeline with LoRA"""
        print("Loading FLUX Control Removal pipeline...")

        # Build transformer with expanded input channels
        transformer = FluxTransformer2DModel.from_pretrained(
            'black-forest-labs/FLUX.1-dev',
            subfolder="transformer",
            torch_dtype=torch.bfloat16
        )

        with torch.no_grad():
            initial_input_channels = transformer.config.in_channels
            new_linear = torch.nn.Linear(
                transformer.x_embedder.in_features * 4,
                transformer.x_embedder.out_features,
                bias=transformer.x_embedder.bias is not None,
                dtype=transformer.dtype,
                device=transformer.device,
            )
            new_linear.weight.zero_()
            new_linear.weight[:, :initial_input_channels].copy_(transformer.x_embedder.weight)
            if transformer.x_embedder.bias is not None:
                new_linear.bias.copy_(transformer.x_embedder.bias)
            transformer.x_embedder = new_linear
            transformer.register_to_config(in_channels=initial_input_channels * 4)

        # Build pipeline
        self.pipe = FluxControlRemovalPipeline.from_pretrained(
            'black-forest-labs/FLUX.1-dev',
            transformer=transformer,
            torch_dtype=torch.bfloat16
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.transformer.to(torch.bfloat16)

        # Verify transformer config
        assert (
            self.pipe.transformer.config.in_channels == initial_input_channels * 4
        ), f"Transformer input channels mismatch: {self.pipe.transformer.config.in_channels}"

        # Load LoRA weights
        self.pipe.load_lora_weights(
            'theSure/Omnieraser',
            weight_name="pytorch_lora_weights.safetensors"
        )

        print("FLUX Control Removal pipeline loaded successfully!")

    def generate_background_with_flux_control(self, frame1_image, detections_to_remove, proposals,
                                            guidance_scale=3.5, num_inference_steps=28,
                                            target_size=(1024, 1024), use_feathering=True, output_path=None):
        """
        Generate clean background using FLUX Control Removal with proper compositing

        Note: use_feathering parameter is kept for API compatibility but currently not used.
        FLUX works better with hard mask edges.
        """

        if not detections_to_remove:
            return frame1_image, [], "no objects detected", None

        # Filter detections with masks
        detections_with_masks = [d for d in detections_to_remove
                               if hasattr(d, 'mask') and d.mask is not None]

        if not detections_with_masks:
            return frame1_image, [], "no valid masks", None

        # Store original image for compositing
        original_image = frame1_image.copy()

        # Combine all masks
        height, width = np.array(frame1_image).shape[:2]
        combined_mask = np.zeros((height, width), dtype=np.uint8)

        for detection in detections_with_masks:
            mask = detection.mask
            if mask.shape != (height, width):
                mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
                mask_resized = mask_pil.resize((width, height), Image.Resampling.NEAREST)
                mask = np.array(mask_resized)
                mask = (mask > 127).astype(np.uint8)
            combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)

        # Expand mask for better coverage
        kernel = np.ones((5, 5), np.uint8)
        expanded_mask = cv2.dilate(combined_mask, kernel, iterations=2)

        # Note: use_feathering parameter is available but currently feathering
        # is not applied to FLUX masks (uses hard edges for better control)
        # Could apply feathering here in the future if needed:
        # if use_feathering:
        #     expanded_mask = apply_mask_feathering(expanded_mask, feather_radius=3)

        # Convert mask to RGB format for FLUX pipeline
        mask_rgb = np.stack([expanded_mask, expanded_mask, expanded_mask], axis=-1) * 255
        mask_pil = Image.fromarray(mask_rgb.astype(np.uint8))

        # Resize image and mask to target size
        image_resized = frame1_image.resize(target_size, Image.Resampling.LANCZOS)
        mask_resized = mask_pil.resize(target_size, Image.Resampling.NEAREST)

        # Generate optimized prompt for object removal
        prompt = self._generate_removal_prompt(proposals)

        # Generate with FLUX Control Removal
        generator = torch.Generator(device=self.device).manual_seed(42)
        # if output_path and os.path.exists(os.path.join(output_path, "generated_background.png")):
        #     print(f"Skipping generation, output already exists at {output_path}")
        #     generated_image = Image.open(os.path.join(output_path, "generated_background.png")).convert("RGB")
        # else:
        generated_image = self.pipe(
            prompt=prompt,
            control_image=image_resized,
            control_mask=mask_resized,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            max_sequence_length=512,
            height=target_size[1],
            width=target_size[0],
        ).images[0]

        # Resize generated image back to original size if needed
        if generated_image.size != frame1_image.size:
            generated_image = generated_image.resize(frame1_image.size, Image.Resampling.LANCZOS)
            mask_resized = mask_resized.resize(frame1_image.size, Image.Resampling.NEAREST)

        # Composite the images using the mask
        final_result = composite_images_with_mask(
            original_image,
            generated_image,
            mask_resized
        )

        return final_result, detections_with_masks, prompt, mask_resized

    def _generate_removal_prompt(self, proposals):
        """Generate removal prompt based on proposals"""
        removal_prompt = "Clean background scene without any objects."

        # Add context from scene analysis if available
        if proposals and "scene_analysis" in proposals:
            scene_context = proposals["scene_analysis"]
            removal_prompt = f"Clean {scene_context.lower()} background without any moving objects."

        return removal_prompt
