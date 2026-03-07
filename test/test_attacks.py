#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adversarial Attack Testing Module for VideoSeal
Tests robustness of watermarks under various transformations.
"""

import argparse
import io
import json
import logging
import sys
from pathlib import Path
from typing import Callable, Tuple, List

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageEnhance, ImageFilter

from demo import WATERMARK_TEXT as DEMO_WATERMARK_TEXT, VideoSealConfig, VideoSealEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Attack Definitions ---
def attack_jpeg(img: Image.Image, quality: int = 50) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def attack_noise(img: Image.Image, std: float = 15.0) -> Image.Image:
    img_arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, std, img_arr.shape)
    img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img_arr)

def attack_crop(img: Image.Image, factor: float = 0.8) -> Image.Image:
    w, h = img.size
    new_w, new_h = int(w * factor), int(h * factor)
    left, top = (w - new_w) // 2, (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))

def attack_rotation(img: Image.Image, angle: float = 5.0) -> Image.Image:
    # Expand=False keeps the original size, filled with white
    return img.rotate(angle, resample=Image.BILINEAR, fillcolor=(255, 255, 255))

def attack_scale(img: Image.Image, factor: float = 0.5) -> Image.Image:
    w, h = img.size
    scaled = img.resize((int(w * factor), int(h * factor)), Image.BICUBIC)
    return scaled.resize((w, h), Image.BICUBIC)

def attack_blur(img: Image.Image, radius: float = 2.0) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius))

def attack_brightness(img: Image.Image, factor: float = 1.5) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)

# Registry of single attacks
SINGLE_ATTACKS = {
    "JPEG_Compression_50": lambda img: attack_jpeg(img, 50),
    "JPEG_Compression_30": lambda img: attack_jpeg(img, 30),
    "Gaussian_Noise_15": lambda img: attack_noise(img, 15.0),
    "Gaussian_Noise_25": lambda img: attack_noise(img, 25.0),
    "Center_Crop_80%": lambda img: attack_crop(img, 0.8),
    "Center_Crop_50%": lambda img: attack_crop(img, 0.5),
    "Rotation_5_deg": lambda img: attack_rotation(img, 5.0),
    "Rotation_15_deg": lambda img: attack_rotation(img, 15.0),
    "Scale_Down_50%": lambda img: attack_scale(img, 0.5),
    "Gaussian_Blur_r2": lambda img: attack_blur(img, 2.0),
    "Brightness_Inc_1.5x": lambda img: attack_brightness(img, 1.5),
    "Brightness_Dec_0.5x": lambda img: attack_brightness(img, 0.5),
}

# Registry of combination attacks
COMBINED_ATTACKS = {
    "JPEG_50 + Noise_15": lambda img: attack_noise(attack_jpeg(img, 50), 15.0),
    "Crop_80% + JPEG_50": lambda img: attack_jpeg(attack_crop(img, 0.8), 50),
    "Rotation_5 + Crop_80%": lambda img: attack_crop(attack_rotation(img, 5.0), 0.8),
    "Scale_50% + JPEG_50": lambda img: attack_jpeg(attack_scale(img, 0.5), 50),
    "Blur_r2 + Noise_15 + JPEG_50": lambda img: attack_jpeg(attack_noise(attack_blur(img, 2.0), 15.0), 50),
    "Rotation_5 + Scale_50% + JPEG_30": lambda img: attack_jpeg(attack_scale(attack_rotation(img, 5.0), 0.5), 30),
}


def test_attacks_on_image(config: VideoSealConfig, image_path: Path, expected_text: str):
    logger.info("Initializing VideoSeal Engine...")
    engine = VideoSealEngine(config)

    logger.info(f"Loading image from {image_path}")
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return

    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")

    # Helper for testing a single condition
    def test_single_condition(condition_name: str, img_to_test: Image.Image):
        try:
            extracted_text, flips = engine.extract(img_to_test)
            success = (extracted_text == expected_text)
            status = "✅ PASS" if success else "❌ FAIL"
            flips_info = f"(ECC recovered {flips} bits)" if flips >= 0 else "(Unrecoverable)"
            
            print(f"{condition_name:<32} | {status} | Extracted: '{extracted_text}' {flips_info}")
            return success
        except Exception as e:
            logger.error(f"Error during extraction for {condition_name}: {e}")
            print(f"{condition_name:<32} | ⚠️ ERROR  | {str(e)}")
            return False

    print("\n" + "="*85)
    print(f"Testing Adversarial Attacks on: {image_path.name}")
    print(f"Expected Watermark Text   : '{expected_text}'")
    print("="*85 + "\n")

    # Test baseline
    print("--- Baseline (No Attack) ---")
    test_single_condition("Clean_Image", img_rgb)
    print()

    # Test single attacks
    print("--- Single Vulnerability Tests ---")
    passed_single = 0
    total_single = len(SINGLE_ATTACKS)
    for name, attack_func in SINGLE_ATTACKS.items():
        attacked_img = attack_func(img_rgb)
        if test_single_condition(name, attacked_img):
            passed_single += 1
    
    print()

    # Test combintations
    print("--- Combinatorial Attack Tests ---")
    passed_combined = 0
    total_combined = len(COMBINED_ATTACKS)
    for name, attack_func in COMBINED_ATTACKS.items():
        attacked_img = attack_func(img_rgb)
        if test_single_condition(name, attacked_img):
            passed_combined += 1

    print("\n" + "="*85)
    print("SUMMARY OF ROBUSTNESS:")
    print(f"Single Attacks Passed   : {passed_single} / {total_single} ({passed_single/total_single*100:.1f}%)")
    print(f"Combined Attacks Passed : {passed_combined} / {total_combined} ({passed_combined/total_combined*100:.1f}%)")
    print("="*85 + "\n")


if __name__ == "__main__":
    # ==========================================================================
    # ⚙️ USER CONFIGURATION
    # Modify these paths to test different images or configurations.
    # ==========================================================================
    
    TEST_IMAGE_PATHStr = "val2017_subset_wm/000000001675_wm.jpg"
    CONFIG_JSON_PATHStr = "bch_config.json"
    
    # The expected watermark string to verify against (imported from demo.py)
    WATERMARK_TEXT = DEMO_WATERMARK_TEXT
    
    # ==========================================================================

    SCRIPT_ROOT = Path(__file__).parent.absolute()
    image_path = SCRIPT_ROOT / TEST_IMAGE_PATHStr
    config_path = SCRIPT_ROOT / CONFIG_JSON_PATHStr

    # Load configuration
    try:
        v_config = VideoSealConfig.from_json(
            config_path=config_path,
            input_path=image_path,
            watermark_text=WATERMARK_TEXT
        )
        
        expected_text = v_config.watermark_text
        logger.info(f"Using defined watermark text: '{expected_text}'")
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    test_attacks_on_image(v_config, image_path, expected_text)
