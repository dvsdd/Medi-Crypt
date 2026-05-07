import os, math, struct, gc, glob
import numpy as np
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

try:
    import tifffile
except ImportError:
    raise SystemExit("\nERROR: Run '!pip install tifffile' first.")

Image.MAX_IMAGE_PIXELS = None

# --- CLEANUP ---
old_files = glob.glob("/content/dct5_stego_part_*.tif")
for f in old_files: os.remove(f)

# --- SETTINGS ---
BLOCK_SIZE = 8
CHUNK_SIZE = 5120
COEFF_POSITIONS = [(3,3), (3,4), (4,3), (4,4), (5,3)]
BITS_PER_BLOCK = len(COEFF_POSITIONS) * 3
QUANTIZATION_STEP = 12

ENC_FILE = "enc_ima.bin"
COVER_FILE = "/content/drive/MyDrive/Medi Crypt/cover_float32.tif"  # file path is variable
STEGO_OUT = "/content/dct5_stego_part_{}.tif" # file path is variable

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

class BitStreamer:
    def __init__(self, filepath):
        self.file = open(filepath, "rb")
        payload_len = os.path.getsize(filepath)
        header_bytes = struct.pack(">Q", payload_len)
        self.buffer = np.unpackbits(np.frombuffer(header_bytes, dtype=np.uint8)).tolist()
        self.total_bits = (8 + payload_len) * 8
        self.bits_read = 0

    def get_bits(self, count):
        while len(self.buffer) < count:
            chunk = self.file.read(1024 * 1024)
            if not chunk: break
            self.buffer.extend(np.unpackbits(np.frombuffer(chunk, dtype=np.uint8)).tolist())
        extracted = self.buffer[:count]
        self.buffer = self.buffer[count:]
        self.bits_read += len(extracted)
        return extracted
    def is_empty(self): return self.bits_read >= self.total_bits

streamer = BitStreamer(ENC_FILE)
cover_np = tifffile.imread(COVER_FILE)
H, W, C = cover_np.shape

part = 1
while not streamer.is_empty():
    out_name = STEGO_OUT.format(part)
    print(f"\n--- Creating {out_name} ---")

    # Created with Native H and W (Zero Footprint)
    out_mmap = tifffile.memmap(out_name, shape=(H, W, 3), dtype=np.float32, photometric='rgb')

    for y in range(0, H, CHUNK_SIZE):
        for x in range(0, W, CHUNK_SIZE):
            chunk_h = min(CHUNK_SIZE, H - y)
            chunk_w = min(CHUNK_SIZE, W - x)
            chunk = cover_np[y:y+chunk_h, x:x+chunk_w].copy()

            if streamer.is_empty():
                out_mmap[y:y+chunk_h, x:x+chunk_w] = chunk
                continue

            # THE SPATIAL ISOLATION FIX
            # Split the chunk exactly in half down the block grid
            mid_w = (chunk_w // 2 // BLOCK_SIZE) * BLOCK_SIZE
            if mid_w == 0: continue

            # The AI only sees the Pristine Left Half
            control_crop = chunk[:, :mid_w]
            control_uint8 = (control_crop * 255.0).astype(np.uint8)
            inp = preprocess(Image.fromarray(control_uint8)).unsqueeze(0).to(device)

            with torch.no_grad():
                fmap = feature_extractor(inp).squeeze(0).cpu().numpy()

            heat = np.mean(fmap, axis=0)
            heat_resized = cv2.resize(heat, (control_crop.shape[1], control_crop.shape[0]))

            blocks = []
            for i in range(0, chunk_h, BLOCK_SIZE):
                for j in range(0, mid_w, BLOCK_SIZE):
                    # Data goes in the mirror location on the Right Half
                    target_j = mid_w + j

                    # Strictly stay inside chunk bounds (replaces the need for padding)
                    if i + BLOCK_SIZE <= chunk_h and target_j + BLOCK_SIZE <= chunk_w:
                        score = float(np.mean(heat_resized[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]))
                        blocks.append(((i, target_j), score))

            # Deterministic sorting
            blocks.sort(key=lambda b: (b[1], -b[0][0], -b[0][1]), reverse=True)

            stego_chunk_float = chunk.copy()
            B = len(blocks)
            needed_bits = B * BITS_PER_BLOCK
            bits = streamer.get_bits(needed_bits)
            actual_bits = len(bits)

            if actual_bits > 0:
                if actual_bits < needed_bits: bits = bits + [0] * (needed_bits - actual_bits)
                bits_t = torch.tensor(bits, dtype=torch.int32, device=device).view(B, 3, -1)
                X_np = np.zeros((B, 3, 8, 8), dtype=np.float32)
                r_coords = np.array([blk[0][0] for blk in blocks])
                c_coords = np.array([blk[0][1] for blk in blocks])

                for i in range(B): X_np[i] = stego_chunk_float[r_coords[i]:r_coords[i]+8, c_coords[i]:c_coords[i]+8].transpose(2, 0, 1) * 255.0
                X_t = torch.tensor(X_np, device=device)
                Y_t = C_mat @ X_t @ C_mat_T

                pos_r = [p[0] for p in COEFF_POSITIONS]
                pos_c = [p[1] for p in COEFF_POSITIONS]
                coeffs = Y_t[:, :, pos_r, pos_c]
                q_val = torch.round(coeffs / QUANTIZATION_STEP).to(torch.int32)
                new_coeffs = ((q_val & ~1) | bits_t) * QUANTIZATION_STEP
                Y_t[:, :, pos_r, pos_c] = new_coeffs.to(torch.float32)

                X_new_t = C_mat_T @ Y_t @ C_mat
                X_new_np = X_new_t.cpu().numpy().transpose(0, 2, 3, 1)
                for i in range(B): stego_chunk_float[r_coords[i]:r_coords[i]+8, c_coords[i]:c_coords[i]+8] = X_new_np[i] / 255.0

            # Direct save. No padding slice required!
            out_mmap[y:y+chunk_h, x:x+chunk_w] = stego_chunk_float
            out_mmap.flush()

            del chunk, control_crop, stego_chunk_float, inp, fmap, heat, heat_resized, blocks
            if actual_bits > 0: del X_np, X_t, Y_t, X_new_t, X_new_np
            gc.collect()
            torch.cuda.empty_cache()

    print(f"Finished part {part}. Bits embedded so far: {streamer.bits_read}")
    part += 1

streamer.file.close()
del cover_np
gc.collect()
print("\n ENCODER COMPLETE! Run the Decoder next.")