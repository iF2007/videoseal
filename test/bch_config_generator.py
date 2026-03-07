import sys
import math
import json
from pathlib import Path

# ==========================================
# 用户配置区 (User Configuration)
# 参数修改后请重新运行本生成器以更新 JSON 配置
# ==========================================
class UserConfig:
    STRENGTH = 1.5                                         # 水印嵌入强度 (推荐: 1.0 - 2.5)
    MAX_WATERMARK_LENGTH = 32                              # 预期写入的最大文本长度
    CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"  # 字符集字典
    MODEL_NAME = "pixelseal"                               # 使用的隐写模型版本
    DEVICE = "cpu"                                         # 推理硬件 (cpu / cuda / mps)
    OUTPUT_FILE = str(Path(__file__).parent / "bch_config.json") # 配置文件输出路径
# ==========================================

class ConfigGenerator:
    def __init__(self, output_path=UserConfig.OUTPUT_FILE):
        self.output_path = output_path

    def generate(self, strength, max_watermark_length, charset, model_name, device):
        print("\n--- BCH Configuration Generator ---")
        print(f"[Input Parameters]")
        print(f"  - STRENGTH\t\t\t: {strength}")
        print(f"  - MAX_WATERMARK_LENGTH\t: {max_watermark_length}")
        print(f"  - MODEL_NAME\t\t\t: {model_name}")
        print(f"  - DEVICE\t\t\t: {device}")
        print(f"  - CHARSET\t\t\t: {charset} (Length: {len(charset)})")
        print("-" * 50)
        
        charset_len = len(charset)
        bits_per_char = math.ceil(math.log2(charset_len))
        print(f"[Calculation] Charset implies {bits_per_char} bits/char.")
        
        data_bits = max_watermark_length * bits_per_char
        print(f"[Calculation] Total data payload requirements: {data_bits} bits.")
        
        if data_bits > 256:
            raise ValueError(f"Error: Data payload ({data_bits} bits) exceeds maximum tensor capacity (256 bits). Please reduce MAX_WATERMARK_LENGTH or CHARSET size.")
            
        data_bytes = math.ceil(data_bits / 8)
        ecc_bytes_available = 32 - data_bytes
        
        if ecc_bytes_available <= 0:
            raise ValueError("Error: Insufficient space for ECC (Error Correction Code). Data payload is too large.")
            
        print(f"[Allocation] Payload bits allocated: {data_bytes} Bytes. Remaining ECC capacity: {ecc_bytes_available} Bytes.")
        
        # =======================================================
        # Compute viable BCH parameters (m, poly) based on remaining ECC space.
        # Mathematical Constraints for BCH:
        #   1. Capacity constraint: Number of correction bits (m * t) <= ecc_bits_available
        #   2. Algebraic bound: Maximum theoretical t <= (2^m - 1) / m
        # =======================================================
        bch_candidates = [
            {"m": 5, "poly": 37},
            {"m": 6, "poly": 67},
            {"m": 7, "poly": 137},
            {"m": 8, "poly": 285},
            {"m": 9, "poly": 529},
            {"m": 10, "poly": 1033}
        ]
        
        ecc_bits_available = ecc_bytes_available * 8
        
        valid_bch_plans = []
        for cand in bch_candidates:
            m = cand["m"]
            poly = cand["poly"]
            
            # Bound 1: Available physical volume for ECC bits
            t_physical_bound = ecc_bits_available // m 
            # Bound 2: Mathematical limitation of the generated field
            t_math_bound = ((2 ** m) - 1) // m
            
            # Theoretical t_max across constraints
            t_max_paper = min(t_physical_bound, t_math_bound)
            
            if t_max_paper > 0:
                # ----------------------------------------------------
                # Expected Error Correction Capacity Adjustments
                # ----------------------------------------------------
                # Our tensor shape acts as a fixed 256-bit block length constraint.
                # However, polynomials with domain N = (2^m - 1) shorter than 256
                # will suffer from aliasing. Errors occurring outside their coverage 
                # domain act as uncorrectable data loss given normal noise distributions.
                
                block_len = (2 ** m) - 1
                
                if block_len >= 256:
                    t_real_effective = t_max_paper
                else:
                    # Apply probability penalty based on domain coverage ratio.
                    coverage_ratio = block_len / 256.0
                    t_real_effective = math.floor(t_max_paper * coverage_ratio)
                    # Deduct systematic aliasing penalty if mapped domain falls behind tensor boundary.
                    if t_real_effective > 0:
                        t_real_effective -= 1 

                valid_bch_plans.append({
                    "m": m,
                    "poly": poly,
                    "t_paper": t_max_paper,
                    "t_real": max(0, t_real_effective),
                    "block_len": block_len,
                    "ecc_bytes_used": math.ceil((m * t_max_paper) / 8)
                })
                
        if not valid_bch_plans:
            raise ValueError("Error: Provided parameters do not allow for even 1 bit of correction.")
            
        # Priority: Highest practical expected error correction rate; fallback to minimal m on ties.
        valid_bch_plans.sort(key=lambda x: (-x["t_real"], x["m"]))
        best_plan = valid_bch_plans[0]
        
        bch_poly = best_plan["poly"]
        m = best_plan["m"]
        bch_t = best_plan["t_paper"]
        bch_t_real = best_plan["t_real"]
        
        print(f"[Result] Selected Optimal BCH Configuration: m={m} (poly={bch_poly})")
        print(f"   - Nominal Error Correction Limit: {bch_t} bits")
        print(f"   - Exp. Effective Correction Across 256 bits: {bch_t_real} bits")
        
        config = {
            "_COMMENTS": {
                "STRENGTH": "Watermark intensity scalar.",
                "MODEL_NAME": "Model architecture utilized.",
                "DEVICE": "Hardware inference device.",
                "CHARSET": "String defining allowable encoding dimension.",
                "MAX_WATERMARK_LENGTH": "Length limitation of the encoded payload string.",
                "BITS_PER_CHAR": "Mathematical bits ceiling required per character based on CHARSET.",
                "BCH_POLY": "BCH generating polynomial primitive integer.",
                "BCH_T": "Maximum correctable bit errors initialized by the BCH instance.",
                "DATA_BYTES": "Real volume requirement to house watermark data payload.",
                "ECC_BYTES": "Reserved byte volume allocated for Error Correction."
            },
            "STRENGTH": strength,
            "MODEL_NAME": model_name,
            "DEVICE": device,
            "CHARSET": charset,
            "MAX_WATERMARK_LENGTH": max_watermark_length,
            "BITS_PER_CHAR": bits_per_char,
            "BCH_POLY": bch_poly,
            "BCH_T": bch_t,
            "DATA_BYTES": data_bytes,
            "ECC_BYTES": math.ceil((m * bch_t) / 8)
        }
        
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
            
        print(f"Output saved to: {self.output_path}\n")

        # ==========================================
        # Parameter Reference Table
        # ==========================================
        print("======== Reference: BCH Alternative Benchmarks ========")
        print(f"  Configuration assumes 256 bits total tensor length.")
        print(f"  Data Payload: {data_bytes} Bytes. ECC Capacity: {ecc_bytes_available} Bytes.\n")
        
        print("                 \t       \t[Nominal Spec]\t[Expected Global Performance]")
        print("  Scheme (Domain-N)\tPoly\tMax Correction\tEffective Expected\tECC Allocation")
        print("  --------------------------------------------------------------------------------------")
        
        for plan in valid_bch_plans:
            is_best = " (Selected)" if plan == best_plan else ""
            danger_tag = "!" if plan["block_len"] < 256 else "*"
            
            print(f"  {danger_tag} m={plan['m']} (N={plan['block_len']})\t{plan['poly']}\t{plan['t_paper']} bits\t{plan['t_real']} bits\t\tAllocated {plan['ecc_bytes_used']} B {is_best}")
            
        print("\n  Note: Schemes prefixed with '!' fail to address the entire 256-bit space, generating alias risks.")
        print("        To override the selected configuration, overwrite BCH_POLY and BCH_T in bch_config.json.")
        print("=======================================================\n")

if __name__ == "__main__":
    generator = ConfigGenerator(UserConfig.OUTPUT_FILE)
    generator.generate(
        strength=UserConfig.STRENGTH,
        max_watermark_length=UserConfig.MAX_WATERMARK_LENGTH,
        charset=UserConfig.CHARSET,
        model_name=UserConfig.MODEL_NAME,
        device=UserConfig.DEVICE
    )
