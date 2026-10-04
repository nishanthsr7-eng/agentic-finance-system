import platform

import psutil
import torch


def get_stats():
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"CPU: {platform.processor()}")
    print(f"System RAM: {psutil.virtual_memory().total / 1024**3:.2f} GB")

    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        free_mem, total_mem = torch.cuda.mem_get_info()
        print(f"Total VRAM: {total_mem / 1024**3:.2f} GB")
        print(f"Free VRAM: {free_mem / 1024**3:.2f} GB")

        # Capability check
        if total_mem >= 10 * 1024**3:
            print("\nRecommended Model: Mistral-7B (Full or 4-bit)")
        elif total_mem >= 6 * 1024**3:
            print(
                "\nRecommended Model: Phi-3-mini (3.8B) or Mistral-7B (Strict 4-bit with Offloading)"
            )
        else:
            print("\nRecommended Model: TinyLlama-1.1B or CPU-only")
    else:
        print("\nNo GPU detected.")


if __name__ == "__main__":
    get_stats()
