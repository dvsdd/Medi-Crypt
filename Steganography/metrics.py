import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
try:
    import tifffile
except ImportError:
    raise SystemExit("\nERROR: Run '!pip install tifffile' first.")

# --- SET YOUR EXACT FILE PATHS HERE ---
COVER_FILE = "/content/drive/MyDrive/Medi Crypt/cover_float32.tif"
STEGO_FILE = "/content/dct5_stego_part_1.tif"  # Update if saved somewhere else

print(" Memory-mapping High-Precision Images...")
cover_mmap = tifffile.memmap(COVER_FILE)
stego_mmap = tifffile.memmap(STEGO_FILE)

# 1. Target the center of the image (The busiest part of the celestial map)
H, W, C = cover_mmap.shape
cy, cx = H // 2, W // 2
crop_size = 1000  # Creates a 2000x2000 crop

print(" Extracting pixels")
cover_crop = cover_mmap[cy-crop_size:cy+crop_size, cx-crop_size:cx+crop_size].copy()
stego_crop = stego_mmap[cy-crop_size:cy+crop_size, cx-crop_size:cx+crop_size].copy()

# 2. Calculate MSE
print("\n Calculating  Mean Squared Error (MSE)...")
mse = np.mean((cover_crop - stego_crop) ** 2)

# 3. Calculate PSNR (Max value is 1.0 for normalized float32)
print("\n Calculating  Peak Signal-to-Noise (PSNR)...")
if mse == 0:
    psnr = float('inf')
else:
    psnr = 20 * math.log10(1.0 / math.sqrt(mse))

# 4. Calculate SSIM (Explicitly setting data_range to 1.0)
print(" Computing Structural Similarity (SSIM)...")
ssim_value = ssim(cover_crop, stego_crop, data_range=1.0, channel_axis=-1)

# --- FINAL DASHBOARD ---
print("\n" + "=" * 40)
print("FINAL METRICS REPORT ")
print("=" * 40)
print(f"Mean Squared Error (MSE) : {mse:.8f}")
print(f"Peak Signal-to-Noise (PSNR): {psnr:.2f} dB")
print(f"Structural Similarity (SSIM): {ssim_value:.4f}")
print("=" * 40)

# Clean up RAM
del cover_mmap, stego_mmap, cover_crop, stego_crop