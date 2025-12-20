#pragma once

// Simple NumPy .npy format I/O for float32 and int32 tensors
// Format spec: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>

namespace npy {

// Parse shape from NumPy header dict, e.g. "(4, 5120)"
static std::vector<size_t> parse_shape(const std::string & shape_str) {
    std::vector<size_t> shape;
    std::string num;
    for (char c : shape_str) {
        if (c >= '0' && c <= '9') {
            num += c;
        } else if (!num.empty()) {
            shape.push_back(std::stoull(num));
            num.clear();
        }
    }
    if (!num.empty()) {
        shape.push_back(std::stoull(num));
    }
    return shape;
}

// Read float32 array from .npy file
static bool read_f32(const std::string & filename, std::vector<float> & data, std::vector<size_t> & shape) {
    FILE * f = fopen(filename.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "npy::read_f32: failed to open '%s'\n", filename.c_str());
        return false;
    }

    // Read magic and version
    uint8_t magic[6];
    if (fread(magic, 1, 6, f) != 6) {
        fprintf(stderr, "npy::read_f32: failed to read magic\n");
        fclose(f);
        return false;
    }

    // Check magic: \x93NUMPY
    if (magic[0] != 0x93 || magic[1] != 'N' || magic[2] != 'U' ||
        magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') {
        fprintf(stderr, "npy::read_f32: invalid magic number\n");
        fclose(f);
        return false;
    }

    uint8_t version[2];
    if (fread(version, 1, 2, f) != 2) {
        fprintf(stderr, "npy::read_f32: failed to read version\n");
        fclose(f);
        return false;
    }

    // Read header length
    uint32_t header_len = 0;
    if (version[0] == 1) {
        uint16_t len16;
        if (fread(&len16, 1, 2, f) != 2) {
            fclose(f);
            return false;
        }
        header_len = len16;
    } else if (version[0] >= 2) {
        if (fread(&header_len, 1, 4, f) != 4) {
            fclose(f);
            return false;
        }
    } else {
        fprintf(stderr, "npy::read_f32: unsupported version %d.%d\n", version[0], version[1]);
        fclose(f);
        return false;
    }

    // Read header
    std::vector<char> header(header_len + 1);
    if (fread(header.data(), 1, header_len, f) != header_len) {
        fprintf(stderr, "npy::read_f32: failed to read header\n");
        fclose(f);
        return false;
    }
    header[header_len] = '\0';
    std::string header_str(header.data());

    // Parse dtype
    auto dtype_pos = header_str.find("'descr'");
    if (dtype_pos == std::string::npos) {
        dtype_pos = header_str.find("\"descr\"");
    }
    if (dtype_pos == std::string::npos) {
        fprintf(stderr, "npy::read_f32: failed to find descr in header\n");
        fclose(f);
        return false;
    }

    // Check for float32 dtype: '<f4' or '|f4' or '>f4' (we'll handle endianness later if needed)
    bool is_f32 = header_str.find("'<f4'") != std::string::npos ||
                  header_str.find("\"<f4\"") != std::string::npos ||
                  header_str.find("'|f4'") != std::string::npos ||
                  header_str.find("\"|f4\"") != std::string::npos;
    if (!is_f32) {
        fprintf(stderr, "npy::read_f32: expected float32 dtype (<f4), got: %s\n", header_str.c_str());
        fclose(f);
        return false;
    }

    // Check fortran_order
    if (header_str.find("'fortran_order': True") != std::string::npos ||
        header_str.find("\"fortran_order\": True") != std::string::npos) {
        fprintf(stderr, "npy::read_f32: fortran_order not supported\n");
        fclose(f);
        return false;
    }

    // Parse shape
    auto shape_pos = header_str.find("'shape'");
    if (shape_pos == std::string::npos) {
        shape_pos = header_str.find("\"shape\"");
    }
    if (shape_pos == std::string::npos) {
        fprintf(stderr, "npy::read_f32: failed to find shape in header\n");
        fclose(f);
        return false;
    }

    auto paren_start = header_str.find('(', shape_pos);
    auto paren_end = header_str.find(')', paren_start);
    if (paren_start == std::string::npos || paren_end == std::string::npos) {
        fprintf(stderr, "npy::read_f32: failed to parse shape\n");
        fclose(f);
        return false;
    }

    std::string shape_str = header_str.substr(paren_start, paren_end - paren_start + 1);
    shape = parse_shape(shape_str);

    // Calculate total elements
    size_t n_elements = 1;
    for (size_t dim : shape) {
        n_elements *= dim;
    }

    // Read data
    data.resize(n_elements);
    size_t n_read = fread(data.data(), sizeof(float), n_elements, f);
    if (n_read != n_elements) {
        fprintf(stderr, "npy::read_f32: expected %zu elements, got %zu\n", n_elements, n_read);
        fclose(f);
        return false;
    }

    fclose(f);
    return true;
}

// Read int32 array from .npy file
static bool read_i32(const std::string & filename, std::vector<int32_t> & data, std::vector<size_t> & shape) {
    FILE * f = fopen(filename.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "npy::read_i32: failed to open '%s'\n", filename.c_str());
        return false;
    }

    // Read magic and version
    uint8_t magic[6];
    if (fread(magic, 1, 6, f) != 6) {
        fclose(f);
        return false;
    }

    if (magic[0] != 0x93 || magic[1] != 'N' || magic[2] != 'U' ||
        magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') {
        fclose(f);
        return false;
    }

    uint8_t version[2];
    if (fread(version, 1, 2, f) != 2) {
        fclose(f);
        return false;
    }

    uint32_t header_len = 0;
    if (version[0] == 1) {
        uint16_t len16;
        if (fread(&len16, 1, 2, f) != 2) {
            fclose(f);
            return false;
        }
        header_len = len16;
    } else {
        if (fread(&header_len, 1, 4, f) != 4) {
            fclose(f);
            return false;
        }
    }

    std::vector<char> header(header_len + 1);
    if (fread(header.data(), 1, header_len, f) != header_len) {
        fclose(f);
        return false;
    }
    header[header_len] = '\0';
    std::string header_str(header.data());

    // Check for int32 dtype
    bool is_i32 = header_str.find("'<i4'") != std::string::npos ||
                  header_str.find("\"<i4\"") != std::string::npos ||
                  header_str.find("'|i4'") != std::string::npos ||
                  header_str.find("\"|i4\"") != std::string::npos;
    if (!is_i32) {
        fprintf(stderr, "npy::read_i32: expected int32 dtype (<i4), got: %s\n", header_str.c_str());
        fclose(f);
        return false;
    }

    // Parse shape
    auto shape_pos = header_str.find("'shape'");
    if (shape_pos == std::string::npos) {
        shape_pos = header_str.find("\"shape\"");
    }
    auto paren_start = header_str.find('(', shape_pos);
    auto paren_end = header_str.find(')', paren_start);
    std::string shape_str = header_str.substr(paren_start, paren_end - paren_start + 1);
    shape = parse_shape(shape_str);

    size_t n_elements = 1;
    for (size_t dim : shape) {
        n_elements *= dim;
    }

    data.resize(n_elements);
    size_t n_read = fread(data.data(), sizeof(int32_t), n_elements, f);
    if (n_read != n_elements) {
        fclose(f);
        return false;
    }

    fclose(f);
    return true;
}

// Write float32 array to .npy file
static bool write_f32(const std::string & filename, const float * data, const std::vector<size_t> & shape) {
    FILE * f = fopen(filename.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "npy::write_f32: failed to open '%s' for writing\n", filename.c_str());
        return false;
    }

    // Build header dict
    std::string shape_str = "(";
    for (size_t i = 0; i < shape.size(); i++) {
        shape_str += std::to_string(shape[i]);
        if (i < shape.size() - 1) {
            shape_str += ", ";
        } else if (shape.size() == 1) {
            shape_str += ",";  // Tuple with single element needs trailing comma
        }
    }
    shape_str += ")";

    std::string header_dict = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_str + ", }";

    // Pad header to 64-byte alignment (including magic, version, header_len)
    // Total header = 8 (magic + version + header_len) + header_dict + padding + newline
    size_t base_len = 8 + header_dict.size() + 1;  // +1 for newline
    size_t padding = (64 - (base_len % 64)) % 64;
    header_dict += std::string(padding, ' ');
    header_dict += '\n';

    // Write magic
    const uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    fwrite(magic, 1, 6, f);

    // Write version (1.0)
    const uint8_t version[] = {1, 0};
    fwrite(version, 1, 2, f);

    // Write header length (2 bytes for version 1.0)
    uint16_t header_len = (uint16_t)header_dict.size();
    fwrite(&header_len, 1, 2, f);

    // Write header
    fwrite(header_dict.c_str(), 1, header_dict.size(), f);

    // Write data
    size_t n_elements = 1;
    for (size_t dim : shape) {
        n_elements *= dim;
    }
    fwrite(data, sizeof(float), n_elements, f);

    fclose(f);
    return true;
}

} // namespace npy
