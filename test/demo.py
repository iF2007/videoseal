#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VideoSeal Professional Demo Script
Industrial-grade blind watermarking for images with BCH error correction and deep neural networks.
"""

import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import bchlib
import torch
import torchvision.transforms as T
from PIL import Image

import videoseal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress Mac library conflict warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


@dataclass
class VideoSealConfig:
    """Configuration container for VideoSeal operations."""
    # Logic parameters (editable)
    input_path: Path
    watermark_text: str = ""
    save_quality: int = 0
    supported_exts: List[str] = field(default_factory=lambda: ['.jpg', '.png', '.jpeg', '.webp'])

    # Dynamic parameters (loaded from bch_config.json)
    strength: float = 0.0
    model_name: str = ""
    device: str = ""
    charset: str = ""
    max_watermark_length: int = 0
    bits_per_char: int = 0
    bch_poly: int = 0
    bch_t: int = 0
    data_bytes: int = 0

    @classmethod
    def from_json(cls, config_path: Path, input_path: Union[str, Path], watermark_text: str = None) -> 'VideoSealConfig':
        """Load configuration from a JSON file and combine with CLI/runtime parameters."""
        if not config_path.exists():
            generator_path = config_path.parent / "bch_config_generator.py"
            logger.error(f"Configuration file not found: {config_path}")
            logger.error(f"Please run 'python {generator_path}' first to generate parameters.")
            sys.exit(1)

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return cls(
                input_path=Path(input_path),
                watermark_text=watermark_text or cls.watermark_text,
                strength=data["STRENGTH"],
                model_name=data["MODEL_NAME"],
                device=data["DEVICE"],
                charset=data["CHARSET"],
                max_watermark_length=data["MAX_WATERMARK_LENGTH"],
                bits_per_char=data["BITS_PER_CHAR"],
                bch_poly=data["BCH_POLY"],
                bch_t=data["BCH_T"],
                data_bytes=data["DATA_BYTES"]
            )
        except Exception as e:
            logger.error(f"Failed to load configuration JSON: {e}")
            sys.exit(1)


class VideoSealEngine:
    """
    Core engine for embedding and extracting blind watermarks.
    Combines neural networks with BCH (Bose-Chaudhuri-Hocquenghem) error correction.
    """

    def __init__(self, config: VideoSealConfig):
        self.config = config

        # --- 强制优先级：MPS (Mac GPU) > CUDA (Nvidia GPU) > CPU ---
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # 这一行确保你能看到最终结果
        logger.info(f"🚀 HARDWARE ACCELERATION ENABLED: {self.device}")
        logger.info(f"Loading {config.model_name} model on {self.device}...")

        # Initialize BCH Engine
        try:
            # Compatible with both old and new bchlib API
            try:
                self.bch = bchlib.BCH(config.bch_poly, config.bch_t)
                self.bch_api = "old"
            except RuntimeError:
                self.bch = bchlib.BCH(config.bch_t, prim_poly=config.bch_poly)
                self.bch_api = "new"
        except Exception as e:
            raise RuntimeError(f"BCH Engine initialization failed: {e}")

        # Initialize Neural Network
        logger.info(f"Loading {config.model_name} model on {self.device}...")
        try:
            self.model = videoseal.load(config.model_name).to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load VideoSeal model: {e}")

    def _text_to_tensor(self, text: str) -> torch.Tensor:
        """Encodes text into a 256-bit tensor via compression and BCH ECC."""
        if len(text) > self.config.max_watermark_length:
            raise ValueError(f"Text too long: Maximum {self.config.max_watermark_length} characters allowed.")

        # Pad text to maintain fixed length
        padded_text = text.ljust(self.config.max_watermark_length, '_')

        # 1. Convert characters to bit stream based on charset
        bits_arr = []
        for char in padded_text:
            if char not in self.config.charset:
                raise ValueError(f"Invalid character '{char}' not in charset table.")
            char_idx = self.config.charset.index(char)
            bits_arr.extend([int(b) for b in format(char_idx, f'0{self.config.bits_per_char}b')])

        # 2. Pack bits into bytes
        data_bytes = bytearray()
        for i in range(0, len(bits_arr), 8):
            byte_bits = bits_arr[i:i+8]
            if len(byte_bits) < 8:
                byte_bits.extend([0] * (8 - len(byte_bits)))
            byte_val = sum(b << (7-j) for j, b in enumerate(byte_bits))
            data_bytes.append(byte_val)

        # 3. Apply BCH encoding
        ecc_bytes = self.bch.encode(data_bytes)
        packet = data_bytes + ecc_bytes

        # 4. Flatten to 256-bit float tensor for Neural Network
        final_bits = []
        for b in packet:
            final_bits.extend([int(x) for x in format(b, '08b')])

        if len(final_bits) < 256:
            final_bits.extend([0] * (256 - len(final_bits)))

        return torch.tensor([final_bits[:256]], dtype=torch.float32).to(self.device)

    def _tensor_to_text(self, bits_tensor: torch.Tensor) -> Tuple[str, int]:
        """Decodes a 256-bit tensor back to text via BCH correction and decompression."""
        bits = bits_tensor.squeeze().cpu().tolist()[:256]

        # 1. Extract bytes from bit stream
        byte_list = []
        for i in range(0, 256, 8):
            byte_bits = bits[i:i+8]
            byte_list.append(sum(int(b) << (7-j) for j, b in enumerate(byte_bits)))

        packet = bytearray(byte_list)
        data = packet[:self.config.data_bytes]
        ecc = packet[self.config.data_bytes:self.config.data_bytes + self.bch.ecc_bytes]

        # 2. BCH Error Detection and Correction
        if self.bch_api == "new":
            flips = self.bch.decode(data, ecc)
            if flips >= 0:
                self.bch.correct(data, ecc)
        else:
            flips = self.bch.decode_inplace(data, ecc)

        if flips < 0:
            return "ERROR_UNRECOVERABLE", -1

        # 3. Convert bytes back to bit stream
        bits_arr = []
        for b in data:
            bits_arr.extend([int(x) for x in format(b, '08b')])

        # 4. Decompress bits to characters using charset
        chars = []
        total_bits = self.config.max_watermark_length * self.config.bits_per_char
        for i in range(0, total_bits, self.config.bits_per_char):
            char_bits = bits_arr[i:i+self.config.bits_per_char]
            char_idx = sum(b << (self.config.bits_per_char-1-j) for j, b in enumerate(char_bits))
            if char_idx < len(self.config.charset):
                chars.append(self.config.charset[char_idx])
            else:
                chars.append("?") # Noise or corruption indicator

        extracted_text = ''.join(chars).rstrip('_')
        return extracted_text, flips

    def embed(self, image_pil: Image.Image, text: str) -> Image.Image:
        """Embeds text into a PIL image."""
        try:
            img_t = T.ToTensor()(image_pil.convert("RGB")).unsqueeze(0).to(self.device)
            msg_t = self._text_to_tensor(text)

            with torch.no_grad():
                out = self.model.embed(img_t, msgs=msg_t)
                residual = out["imgs_w"] - img_t
                # Apply strength factor to residuals
                w_img_t = torch.clamp(img_t + (residual * self.config.strength), 0, 1)

            return T.ToPILImage()(w_img_t[0].cpu())
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise

    def extract(self, image_pil: Image.Image) -> Tuple[str, int]:
        """Extracts text from a PIL image."""
        try:
            img_t = T.ToTensor()(image_pil.convert("RGB")).unsqueeze(0).to(self.device)

            with torch.no_grad():
                preds = self.model.detect(img_t)["preds"][0, 1:]
                # Aggregate spatial predictions if necessary
                agg = preds.mean(dim=(-2, -1)) if preds.ndim >= 2 else preds
                bits = (agg > 0).float()

            return self._tensor_to_text(bits)
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return "ERROR_SYSTEM", -2


def run_pipeline(config: VideoSealConfig):
    """Orchestrates the watermark processing pipeline."""
    import gc

    engine = VideoSealEngine(config)
    input_path = config.input_path

    # 1. Determine Output Directory
    if input_path.is_file():
        output_dir = input_path.parent / (input_path.stem + "_wm" + input_path.suffix)
    else:
        output_dir = Path(str(input_path).rstrip("/") + "_wm")

    # 2. Preparative Cleanup
    if output_dir.exists():
        logger.warning(f"Clearing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Identify Target Files
    if input_path.is_file():
        img_paths = [input_path]
    elif input_path.is_dir():
        img_paths = [p for p in input_path.iterdir() if p.suffix.lower() in config.supported_exts]
    else:
        logger.error(f"Input path does not exist: {input_path}")
        return

    logger.info(f"Starting processing on {len(img_paths)} files...")

    # 4. Processing Loop
    for idx, img_path in enumerate(img_paths, 1):
        # logger.info(f"[{idx}/{len(img_paths)}] Processing: {img_path.name}")
        print(f"[{idx}/{len(img_paths)}] Processing: {img_path.name}...", end=" ", flush=True)

        try:
            # Step 1: Embedding
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                watermarked_img = engine.embed(img_rgb, config.watermark_text)

            # Step 2: Serialization (Use optimal JPEG settings for durability)
            save_path = output_dir / f"{img_path.stem}_wm.jpg"
            watermarked_img.save(
                save_path,
                format="JPEG",
                quality=config.save_quality,
                subsampling=0, # Disable chroma subsampling (4:4:4)
                optimize=True
            )

            # Step 3: Verification
            with Image.open(save_path) as check_img:
                check_rgb = check_img.convert("RGB")
                extracted_text, flips = engine.extract(check_rgb)

            if extracted_text == config.watermark_text:
                print(f"✅ PASS (ECC recovered {flips} bits)")
            else:
                print(f"❌ FAIL (Extracted: '{extracted_text}')")

            # Memory Hygiene
            del watermarked_img
            gc.collect()

        except Exception as e:
            print(f"⚠️ ERROR: {e}")


# ==========================================================================
# ⚙️ USER CONFIGURATION
# Modify these parameters directly to control the watermarking process
# ==========================================================================

# Path to the input image or directory containing images
INPUT_PATHStr = "val2017_subset"

# The watermark text (string) to embed into the images
WATERMARK_TEXT = "1234567890"

# Path to the dynamically generated BCH configuration file
CONFIG_JSON_PATHStr = "bch_config.json"

# Output JPEG quality (1-100, 95 is recommended for preserving watermark)
SAVE_QUALITY = 95

# ==========================================================================

if __name__ == "__main__":

    SCRIPT_ROOT = Path(__file__).parent.absolute()
    input_path = SCRIPT_ROOT / INPUT_PATHStr
    config_path = SCRIPT_ROOT / CONFIG_JSON_PATHStr

    # Load and execute
    try:
        v_config = VideoSealConfig.from_json(
            config_path=config_path,
            input_path=input_path,
            watermark_text=WATERMARK_TEXT
        )
        v_config.save_quality = SAVE_QUALITY
    except Exception as e:
        logger.error(f"Failed to initialize configuration: {e}")
        sys.exit(1)

    run_pipeline(v_config)