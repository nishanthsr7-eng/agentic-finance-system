"""
Diagnose bitsandbytes CUDA support for RTX 5050 (sm_120)
"""

import subprocess
import sys

import torch

print(f"Torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
cap = torch.cuda.get_device_capability()
print(f"Compute capability: {cap[0]}.{cap[1]} (sm_{cap[0]}{cap[1]})")

print("\n--- Testing bitsandbytes 4-bit load ---")
try:
    import bitsandbytes as bnb

    print(f"bitsandbytes: {bnb.__version__}")

    # Try a tiny 4-bit linear layer as a smoke test
    layer = bnb.nn.Linear4bit(64, 64, bias=False, quant_type="nf4")
    layer = layer.cuda()
    x = torch.randn(1, 64).cuda()
    out = layer(x)
    print(f"4-bit layer test PASSED. Output shape: {out.shape}")
except Exception as e:
    print(f"FAILED: {e}")
