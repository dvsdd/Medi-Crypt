import os, math, struct, gc, glob
import numpy as np
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
try:
    import tifffile
except ImportError:
    raise SystemExit("\nERROR: Run '!pip install tifffile' first.")
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

BLOCK_SIZE = 8
CHUNK_SIZE = 5120

COEFF_POSITIONS = [(3,3), (3,4), (4,3), (4,4), (5,3)]
BITS_PER_BLOCK = len(COEFF_POSITIONS) * 3
QUANTIZATION_STEP = 12

STEGO_FILES = sorted(glob.glob("/content/dct5_stego_part_*.tif"))
EXTRACTED_BIN = "extracted_enc_ima.bin"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dct_matrix(N=8):
    C = np.zeros((N, N), dtype=np.float32)
    for k in range(N):
        for n in range(N):
            C[k, n] = 1.0 / np.sqrt(N) if k == 0 else np.sqrt(2.0 / N) * np.cos(np.pi * k * (2.0 * n + 1.0) / (2.0 * N))
    return torch.tensor(C, device=device)

C_mat = get_dct_matrix(8)
C_mat_T = C_mat.t()

print("\nLoading ResNet50...")
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)
resnet.eval()
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-3]).to(device)
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class BitWriter:
    def __init__(self, filepath):
        self.file = open(filepath, "wb")
        self.buffer = []
        self.payload_len = None
        self.bytes_written = 0
        self.header_bits = 64

    def push_bits(self, bits):
        self.buffer.extend(bits)
        if self.payload_len is None and len(self.buffer) >= self.header_bits:
            header_bits = self.buffer[:self.header_bits]
            self.buffer = self.buffer[self.header_bits:]
            self.payload_len = struct.unpack(">Q", np.packbits(header_bits).tobytes())[0]
            print(f"📡 Header Decoded: Expecting payload of {self.payload_len} bytes.")

        if self.payload_len is not None:
            needed_bytes = self.payload_len - self.bytes_written
            bytes_to_write = min(needed_bytes, len(self.buffer) // 8)

            if bytes_to_write > 0:
                bits_to_pack = self.buffer[:bytes_to_write * 8]
                self.buffer = self.buffer[bytes_to_write * 8:]
                self.file.write(np.packbits(bits_to_pack).tobytes())
                self.bytes_written += bytes_to_write
    def is_done(self): return self.payload_len is not None and self.bytes_written >= self.payload_len

writer = BitWriter(EXTRACTED_BIN)

for stego_file in STEGO_FILES:
    if writer.is_done(): break

    print(f"\n Extracting from {os.path.basename(stego_file)}...")
    stego_mmap = tifffile.memmap(stego_file)
    H, W, _ = stego_mmap.shape

    for y in range(0, H, CHUNK_SIZE):
        if writer.is_done(): break
        for x in range(0, W, CHUNK_SIZE):
            if writer.is_done(): break

            chunk_h = min(CHUNK_SIZE, H - y)
            chunk_w = min(CHUNK_SIZE, W - x)
            chunk_float = stego_mmap[y:y+chunk_h, x:x+chunk_w].copy()

            # The exact same Spatial Isolation
            mid_w = (chunk_w // 2 // BLOCK_SIZE) * BLOCK_SIZE
            if mid_w == 0: continue

            # Extract Heatmap from the Pristine Left Half
            control_crop = chunk_float[:, :mid_w]
            control_uint8 = (control_crop * 255.0).astype(np.uint8)
            inp = preprocess(Image.fromarray(control_uint8)).unsqueeze(0).to(device)

            with torch.no_grad():
                fmap = feature_extractor(inp).squeeze(0).cpu().numpy()

            heat = np.mean(fmap, axis=0)
            heat_resized = cv2.resize(heat, (control_crop.shape[1], control_crop.shape[0]))

            blocks = []
            for i in range(0, chunk_h, BLOCK_SIZE):
                for j in range(0, mid_w, BLOCK_SIZE):
                    # Extract from the mirror location on the Right Half
                    target_j = mid_w + j

                    if i + BLOCK_SIZE <= chunk_h and target_j + BLOCK_SIZE <= chunk_w:
                        score = float(np.mean(heat_resized[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]))
                        blocks.append(((i, target_j), score))

            blocks.sort(key=lambda b: (b[1], -b[0][0], -b[0][1]), reverse=True)

            B = len(blocks)
            X_np = np.zeros((B, 3, 8, 8), dtype=np.float32)
            r_coords = np.array([blk[0][0] for blk in blocks])
            c_coords = np.array([blk[0][1] for blk in blocks])

            for i in range(B): X_np[i] = chunk_float[r_coords[i]:r_coords[i]+8, c_coords[i]:c_coords[i]+8].transpose(2, 0, 1) * 255.0
            X_t = torch.tensor(X_np, device=device)
            Y_t = C_mat @ X_t @ C_mat_T

            pos_r = [p[0] for p in COEFF_POSITIONS]
            pos_c = [p[1] for p in COEFF_POSITIONS]
            coeffs = Y_t[:, :, pos_r, pos_c]
            q_val = torch.round(coeffs / QUANTIZATION_STEP).to(torch.int32)
            extracted_bits_t = q_val & 1

            extracted_bits = extracted_bits_t.view(-1).cpu().numpy().tolist()
            writer.push_bits(extracted_bits)

            del chunk_float, control_crop, control_uint8, inp, fmap, heat, heat_resized, blocks
            del X_np, X_t, Y_t, coeffs, q_val, extracted_bits_t
            gc.collect()
            torch.cuda.empty_cache()

writer.file.close()
print(f"\n EXTRACTION COMPLETE! Reassembled file saved as: {EXTRACTED_BIN}")