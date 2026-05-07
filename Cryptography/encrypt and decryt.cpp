#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <openssl/evp.h>
#include <openssl/err.h>
#include <openssl/rand.h>
#include <openssl/sha.h>

// --- CONFIGURATION ---
const int CHUNK_SIZE = 64 * 1024;   // 64KB chunks
const int SALT_SIZE  = 16;          // 16 bytes for Salt
const int KEY_SIZE   = 32;          // 32 bytes (256 bits) for Key
const int IV_SIZE    = 16;          // 16 bytes (128 bits) for IV
const int PBKDF2_ITERS = 100000;    // Security Iterations

void handleErrors() {
    unsigned long err = ERR_get_error();
    char buf[256];
    ERR_error_string_n(err, buf, sizeof(buf));
    throw std::runtime_error(std::string("OpenSSL error: ") + buf);
}

// Derive Key & IV from Password + Salt
void derive_key_iv(const std::string& password, const unsigned char* salt, unsigned char* key, unsigned char* iv) {
    unsigned char out[KEY_SIZE + IV_SIZE];
    if (PKCS5_PBKDF2_HMAC(password.c_str(), (int)password.length(), salt, SALT_SIZE, PBKDF2_ITERS, EVP_sha256(), KEY_SIZE + IV_SIZE, out) != 1) {
        handleErrors();
    }
    std::memcpy(key, out, KEY_SIZE);
    std::memcpy(iv,  out + KEY_SIZE, IV_SIZE);
}

void encrypt_file(const std::string& input_path, const std::string& output_path, const std::string& password) {
    std::ifstream in(input_path, std::ios::binary);
    std::ofstream out(output_path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open input file: " + input_path);
    if (!out) throw std::runtime_error("Cannot create output file: " + output_path);

    // 1. Generate and Write Salt
    unsigned char salt[SALT_SIZE];
    if (RAND_bytes(salt, SALT_SIZE) != 1) handleErrors();
    out.write((char*)salt, SALT_SIZE);

    // 2. Derive Key/IV
    unsigned char key[KEY_SIZE];
    unsigned char iv[IV_SIZE];
    derive_key_iv(password, salt, key, iv);

    // 3. Init Encryption
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (1 != EVP_EncryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, iv)) handleErrors();

    // 4. Encrypt Loop
    std::vector<unsigned char> in_buf(CHUNK_SIZE);
    std::vector<unsigned char> out_buf(CHUNK_SIZE + EVP_MAX_BLOCK_LENGTH);
    int out_len;

    while (in.read((char*)in_buf.data(), CHUNK_SIZE) || in.gcount() > 0) {
        int bytes_read = in.gcount();
        if (bytes_read > 0) {
            if (1 != EVP_EncryptUpdate(ctx, out_buf.data(), &out_len, in_buf.data(), bytes_read)) handleErrors();
            out.write((char*)out_buf.data(), out_len);
        }
        if (in.eof()) break;
    }

    // 5. Finalize
    if (1 != EVP_EncryptFinal_ex(ctx, out_buf.data(), &out_len)) handleErrors();
    out.write((char*)out_buf.data(), out_len);

    EVP_CIPHER_CTX_free(ctx);
}

void decrypt_file(const std::string& input_path, const std::string& output_path, const std::string& password) {
    std::ifstream in(input_path, std::ios::binary);
    std::ofstream out(output_path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open input file: " + input_path);
    if (!out) throw std::runtime_error("Cannot create output file: " + output_path);

    // 1. Read Salt
    unsigned char salt[SALT_SIZE];
    in.read((char*)salt, SALT_SIZE);
    if (in.gcount() != SALT_SIZE) {
        throw std::runtime_error("File too small: Missing salt. Is this an encrypted file?");
    }

    // 2. Derive Key/IV
    unsigned char key[KEY_SIZE];
    unsigned char iv[IV_SIZE];
    derive_key_iv(password, salt, key, iv);

    // 3. Init Decryption
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (1 != EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, iv)) handleErrors();

    // 4. Decrypt Loop
    std::vector<unsigned char> in_buf(CHUNK_SIZE);
    std::vector<unsigned char> out_buf(CHUNK_SIZE + EVP_MAX_BLOCK_LENGTH);
    int out_len;

    while (in.read((char*)in_buf.data(), CHUNK_SIZE) || in.gcount() > 0) {
        int bytes_read = in.gcount();
        if (bytes_read > 0) {
            if (1 != EVP_DecryptUpdate(ctx, out_buf.data(), &out_len, in_buf.data(), bytes_read)) handleErrors();
            out.write((char*)out_buf.data(), out_len);
        }
        if (in.eof()) break;
    }

    // 5. Finalize
    if (1 != EVP_DecryptFinal_ex(ctx, out_buf.data(), &out_len)) {
        EVP_CIPHER_CTX_free(ctx);
        throw std::runtime_error("Decryption failed! Incorrect password or corrupted file.");
    }
    out.write((char*)out_buf.data(), out_len);

    EVP_CIPHER_CTX_free(ctx);
}

int main() {
    try {
        int choice;
        std::cout << "1. Encrypt File\n2. Decrypt File\nEnter choice (1 or 2): ";
        if (!(std::cin >> choice)) return 1;
        std::cin.ignore(); // consume newline
        
        std::string input_file, output_file, password;
        
        if (choice == 1) {
            std::cout << "--- ENCRYPTION ---" << std::endl;
            std::cout << "Enter input file path (default: D:\\AAA\\image-00000.dcm): ";
            std::string input_in;
            std::getline(std::cin, input_in);
            input_file = input_in.empty() ? "D:\\AAA\\image-00000.dcm" : input_in;

            std::cout << "Enter output file path (default: D:\\AAA\\image-00000_encrypted.bin): ";
            std::string output_in;
            std::getline(std::cin, output_in);
            output_file = output_in.empty() ? "D:\\AAA\\image-00000_encrypted.bin" : output_in;
            
            std::cout << "Target: " << input_file << std::endl;
            std::cout << "Enter Password to lock this file: ";
            std::getline(std::cin, password);

            encrypt_file(input_file, output_file, password);
            std::cout << "Success! Encrypted file created at: " << output_file << std::endl;
        } else if (choice == 2) {
            std::cout << "--- DECRYPTION ---" << std::endl;
            std::cout << "Enter input file path (default: D:\\AAA\\recovered_image_encrypted.bin): ";
            std::string input_in;
            std::getline(std::cin, input_in);
            input_file = input_in.empty() ? "D:\\AAA\\recovered_image_encrypted.bin" : input_in;

            std::cout << "Enter output file path (default: D:\\AAA\\final_recovery.dcm): ";
            std::string output_in;
            std::getline(std::cin, output_in);
            output_file = output_in.empty() ? "D:\\AAA\\final_recovery.dcm" : output_in;
            
            std::cout << "Target: " << input_file << std::endl;
            std::cout << "Enter Password to unlock this file: ";
            std::getline(std::cin, password);

            decrypt_file(input_file, output_file, password);
            std::cout << "Success! Decrypted file created at: " << output_file << std::endl;
        } else {
            std::cerr << "Invalid choice." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}