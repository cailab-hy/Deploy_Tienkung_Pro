// Copyright (C) 2011  Carl Rogers
// Released under MIT License

#include "cnpy.h"
#include <algorithm>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <regex>

namespace cnpy {

void parse_npy_header(unsigned char* buffer, size_t& word_size, std::vector<size_t>& shape, bool& fortran_order) {
    std::string header(reinterpret_cast<char*>(buffer));

    // fortran order
    size_t loc1 = header.find("fortran_order");
    if (loc1 == std::string::npos)
        throw std::runtime_error("parse_npy_header: failed to find header keyword: 'fortran_order'");
    loc1 += 16;
    fortran_order = (header.substr(loc1, 4) == "True");

    // shape
    loc1 = header.find("(");
    size_t loc2 = header.find(")");
    if (loc1 == std::string::npos || loc2 == std::string::npos)
        throw std::runtime_error("parse_npy_header: failed to find header keyword: '(' or ')'");

    std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
    shape.clear();
    if (!str_shape.empty()) {
        std::istringstream ss(str_shape);
        std::string token;
        while (std::getline(ss, token, ',')) {
            token.erase(std::remove(token.begin(), token.end(), ' '), token.end());
            if (!token.empty()) {
                shape.push_back(std::stoull(token));
            }
        }
    }

    // endian, word size, data type
    loc1 = header.find("descr");
    if (loc1 == std::string::npos)
        throw std::runtime_error("parse_npy_header: failed to find header keyword: 'descr'");
    loc1 += 9;
    bool littleEndian = (header[loc1] == '<' || header[loc1] == '|');
    (void)littleEndian; // assume little-endian on ARM/x86

    std::string str_ws = header.substr(loc1 + 1);
    loc2 = str_ws.find("'");
    word_size = std::stoul(str_ws.substr(0, loc2));
}

void parse_npy_header(FILE* fp, size_t& word_size, std::vector<size_t>& shape, bool& fortran_order) {
    char buffer[256];
    // Read magic (6 bytes) + version (2 bytes) + header_len (2 bytes) = 10 bytes
    size_t res = fread(buffer, sizeof(char), 10, fp);
    if (res != 10) throw std::runtime_error("parse_npy_header: failed fread");

    uint8_t major_version = buffer[6];
    uint32_t header_len;
    if (major_version >= 2) {
        // Version 2+: 4-byte header length (bytes 8-11)
        // We only read 10 bytes, so read 2 more to get the full 4-byte length
        char len_buf[4];
        len_buf[0] = buffer[8];
        len_buf[1] = buffer[9];
        fread(len_buf + 2, 1, 2, fp);
        header_len = *reinterpret_cast<uint32_t*>(len_buf);
    } else {
        // Version 1: 2-byte header length
        header_len = *reinterpret_cast<uint16_t*>(buffer + 8);
    }

    char* header_buffer = new char[header_len + 1];
    res = fread(header_buffer, sizeof(char), header_len, fp);
    if (res != header_len) {
        delete[] header_buffer;
        throw std::runtime_error("parse_npy_header: failed fread");
    }
    header_buffer[header_len] = '\0';
    parse_npy_header(reinterpret_cast<unsigned char*>(header_buffer), word_size, shape, fortran_order);
    delete[] header_buffer;
}

NpyArray npy_load(std::string fname) {
    FILE* fp = fopen(fname.c_str(), "rb");
    if (!fp) throw std::runtime_error("npy_load: Unable to open file " + fname);

    size_t word_size;
    std::vector<size_t> shape;
    bool fortran_order;
    parse_npy_header(fp, word_size, shape, fortran_order);

    NpyArray arr(shape, word_size, fortran_order);
    size_t nread = fread(arr.data<char>(), 1, arr.num_bytes(), fp);
    if (nread != arr.num_bytes()) {
        fclose(fp);
        throw std::runtime_error("npy_load: failed fread");
    }
    fclose(fp);
    return arr;
}

npz_t npz_load(std::string fname) {
    FILE* fp = fopen(fname.c_str(), "rb");
    if (!fp) throw std::runtime_error("npz_load: Unable to open file " + fname);

    npz_t arrays;

    while (1) {
        // Read local file header
        char local_header[30];
        size_t headerres = fread(local_header, 1, 30, fp);
        if (headerres != 30 || local_header[0] != 'P' || local_header[1] != 'K' ||
            local_header[2] != 3 || local_header[3] != 4) {
            break; // End of local file headers
        }

        // Parse header fields
        uint16_t compr_method = *reinterpret_cast<uint16_t*>(&local_header[8]);
        uint32_t compr_bytes = *reinterpret_cast<uint32_t*>(&local_header[18]);
        uint32_t uncompr_bytes = *reinterpret_cast<uint32_t*>(&local_header[22]);
        uint16_t fname_len = *reinterpret_cast<uint16_t*>(&local_header[26]);
        uint16_t extra_field_len = *reinterpret_cast<uint16_t*>(&local_header[28]);

        // Read filename
        char* fname_buf = new char[fname_len + 1];
        fread(fname_buf, 1, fname_len, fp);
        fname_buf[fname_len] = '\0';
        std::string varname(fname_buf);
        delete[] fname_buf;

        // Remove .npy extension
        if (varname.size() >= 4 && varname.substr(varname.size() - 4) == ".npy") {
            varname = varname.substr(0, varname.size() - 4);
        }

        // Skip extra field
        fseek(fp, extra_field_len, SEEK_CUR);

        if (compr_method == 0) {
            // Stored (no compression)
            // Parse npy header from the stored data
            // First read the npy magic + header
            size_t word_size;
            std::vector<size_t> shape;
            bool fortran_order;
            parse_npy_header(fp, word_size, shape, fortran_order);

            NpyArray arr(shape, word_size, fortran_order);
            size_t data_read = fread(arr.data<char>(), 1, arr.num_bytes(), fp);
            if (data_read != arr.num_bytes()) {
                throw std::runtime_error("npz_load: failed to read array data for " + varname);
            }
            arrays[varname] = arr;
        } else {
            // Compressed - use zlib inflate
            std::vector<unsigned char> compr_data(compr_bytes);
            size_t cread = fread(compr_data.data(), 1, compr_bytes, fp);
            if (cread != compr_bytes) {
                throw std::runtime_error("npz_load: failed to read compressed data for " + varname);
            }

            std::vector<unsigned char> uncompr_data(uncompr_bytes);

            z_stream d_stream;
            d_stream.zalloc = Z_NULL;
            d_stream.zfree = Z_NULL;
            d_stream.opaque = Z_NULL;
            d_stream.avail_in = compr_bytes;
            d_stream.next_in = compr_data.data();
            d_stream.avail_out = uncompr_bytes;
            d_stream.next_out = uncompr_data.data();

            int err = inflateInit2(&d_stream, -MAX_WBITS);
            if (err != Z_OK) throw std::runtime_error("npz_load: inflateInit2 failed");
            err = inflate(&d_stream, Z_FINISH);
            if (err != Z_STREAM_END) {
                inflateEnd(&d_stream);
                throw std::runtime_error("npz_load: inflate failed");
            }
            inflateEnd(&d_stream);

            // Parse npy from uncompressed data
            size_t word_size;
            std::vector<size_t> shape;
            bool fortran_order;

            // Skip magic bytes (first 10 bytes: \x93NUMPY + version + header_len)
            unsigned char* buf = uncompr_data.data();
            uint16_t header_len = *reinterpret_cast<uint16_t*>(buf + 8);
            parse_npy_header(buf + 10, word_size, shape, fortran_order);

            NpyArray arr(shape, word_size, fortran_order);
            size_t data_offset = 10 + header_len;
            memcpy(arr.data<char>(), buf + data_offset, arr.num_bytes());
            arrays[varname] = arr;
        }
    }

    fclose(fp);
    return arrays;
}

NpyArray npz_load(std::string fname, std::string varname) {
    npz_t arrays = npz_load(fname);
    auto it = arrays.find(varname);
    if (it == arrays.end()) {
        throw std::runtime_error("npz_load: variable '" + varname + "' not found");
    }
    return it->second;
}

} // namespace cnpy
