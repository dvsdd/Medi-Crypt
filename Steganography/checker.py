import hashlib
import os

def get_sha256(file_path):
    """Calculates the SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in chunks to prevent memory overload
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# Ensure these files exist in your Colab directory
original_file = "enc_ima.bin"
extracted_file = "extracted_enc_ima.bin"

if not os.path.exists(original_file) or not os.path.exists(extracted_file):
    print("❌ ERROR: Could not find one of the binary files to compare.")
else:
    print(" CRYPTOGRAPHIC INTEGRITY CHECK ")
    print("-" * 65)

    hash_orig = get_sha256(original_file)
    hash_extr = get_sha256(extracted_file)

    print(f"Original Hash:  {hash_orig}")
    print(f"Extracted Hash: {hash_extr}")
    print("-" * 65)

    if hash_orig == hash_extr:
        print(" STATUS: PERFECT MATCH (0% Error Rate)")
        print("Conclusion: The steganographic payload was recovered with absolute integrity.")
    else:
        print(" STATUS: MISMATCH DETECTED (Bit corruption occurred).")