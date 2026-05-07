import tifffile
import matplotlib.pyplot as plt
import numpy as np

# --- FILE PATHS ---
COVER_FILE = "/content/drive/MyDrive/Medi Crypt/cover_float32.tif"
STEGO_FILE = "/content/drive/MyDrive/Medi Crypt/dct5_stego_part_1.tif"

print("  massive Float32 images...")
cover_mmap = tifffile.memmap(COVER_FILE)
stego_mmap = tifffile.memmap(STEGO_FILE)

# Float32 images must be clipped strictly between 0.0 and 1.0 for Matplotlib
# Warning: This operation will pull all 246 million pixels directly into active RAM
print(" Processing pixels for display...")
cover_display = np.clip(cover_mmap, 0.0, 1.0)
stego_display = np.clip(stego_mmap, 0.0, 1.0)

# --- PLOTTING ---
print(" Sending raw data !")
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.imshow(cover_display)
plt.title("Original Cover Image", fontsize=16)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(stego_display)
plt.title("Stego Part 1", fontsize=16)
plt.axis('off')

plt.subplots_adjust(wspace=0.05)
plt.tight_layout()

# This is the line where the browser will likely freeze
plt.show()

# Clean up RAM (If the tab survives)
del cover_mmap, stego_mmap, cover_display, stego_display
print(" DONE.")