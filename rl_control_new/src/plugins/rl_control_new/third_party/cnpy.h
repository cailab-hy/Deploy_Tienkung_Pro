// Copyright (C) 2011  Carl Rogers
// Released under MIT License
// license available in LICENSE file, or at http://www.opensource.org/licenses/mit-license.php

#ifndef LIBCNPY_H_
#define LIBCNPY_H_

#include <string>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <cstdio>
#include <typeinfo>
#include <iostream>
#include <cassert>
#include <zlib.h>
#include <map>
#include <memory>
#include <stdint.h>
#include <numeric>
#include <cstring>

namespace cnpy {

struct NpyArray {
    NpyArray(const std::vector<size_t>& _shape, size_t _word_size, bool _fortran_order)
        : shape(_shape), word_size(_word_size), fortran_order(_fortran_order) {
        num_vals = 1;
        for (size_t i = 0; i < shape.size(); i++) num_vals *= shape[i];
        data_holder = std::shared_ptr<std::vector<char>>(new std::vector<char>(num_vals * word_size));
    }

    NpyArray() : shape(0), word_size(0), fortran_order(false), num_vals(0) {}

    template<typename T>
    T* data() {
        return reinterpret_cast<T*>(&(*data_holder)[0]);
    }

    template<typename T>
    const T* data() const {
        return reinterpret_cast<const T*>(&(*data_holder)[0]);
    }

    template<typename T>
    std::vector<T> as_vec() const {
        const T* p = data<T>();
        return std::vector<T>(p, p + num_vals);
    }

    size_t num_bytes() const {
        return data_holder->size();
    }

    std::shared_ptr<std::vector<char>> data_holder;
    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    size_t num_vals;
};

using npz_t = std::map<std::string, NpyArray>;

npz_t npz_load(std::string fname);
NpyArray npz_load(std::string fname, std::string varname);
NpyArray npy_load(std::string fname);

void parse_npy_header(unsigned char* buffer, size_t& word_size, std::vector<size_t>& shape, bool& fortran_order);
void parse_npy_header(FILE* fp, size_t& word_size, std::vector<size_t>& shape, bool& fortran_order);

} // namespace cnpy

#endif
