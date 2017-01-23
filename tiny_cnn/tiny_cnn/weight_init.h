/*
    Copyright (c) 2015, Taiga Nomi
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY 
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND 
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#include "util.h"

namespace tiny_cnn {
namespace weight_init {

class function {
public:
    virtual void fill(vec_t *weight, layer_size_t fan_in, layer_size_t fan_out) = 0;
    virtual function* clone() const = 0;
};

class scalable : public function {
public:
    scalable(float_t value) : scale_(value) {}

    void scale(float_t value) {
        scale_ = value;
    }
protected:
    float_t scale_;
};

/**
 * Use fan-in and fan-out for scaling
 *
 * X Glorot, Y Bengio,
 * Understanding the difficulty of training deep feedforward neural networks
 * Proc. AISTATS 10, May 2010, vol.9, pp249-256
 **/
class xavier : public scalable {
public:
    xavier() : scalable((float_t)6.0) {}
    explicit xavier(float_t value) : scalable(value) {}

    void fill(vec_t *weight, layer_size_t fan_in, layer_size_t fan_out) override {
        const float_t weight_base = std::sqrt(scale_ / (fan_in + fan_out));

        uniform_rand(weight->begin(), weight->end(), -weight_base, weight_base);     
    }

    virtual xavier* clone() const override { return new xavier(scale_); }
};

/**
 * Use fan-in(number of input weight for each neuron) for scaling
 *
 * Y LeCun, L Bottou, G B Orr, and K Muller,
 * Efficient backprop
 * Neural Networks, Tricks of the Trade, Springer, 1998
 **/
class lecun : public scalable {
public:
    lecun() : scalable((float_t)1.0) {}
    explicit lecun(float_t value) : scalable(value) {}

    void fill(vec_t *weight, layer_size_t fan_in, layer_size_t fan_out) {
        CNN_UNREFERENCED_PARAMETER(fan_out);

        const float_t weight_base = scale_ / std::sqrt(fan_in);

        uniform_rand(weight->begin(), weight->end(), -weight_base, weight_base);
    }

    virtual lecun* clone() const override { return new lecun(scale_); }
};

class constant : public scalable {
public:
    constant() : scalable((float_t)0.0) {}
    explicit constant(float_t value) : scalable(value) {}

    void fill(vec_t *weight, layer_size_t fan_in, layer_size_t fan_out) {
        CNN_UNREFERENCED_PARAMETER(fan_in);
        CNN_UNREFERENCED_PARAMETER(fan_out);

        std::fill(weight->begin(), weight->end(), scale_);
    }

    virtual constant* clone() const override { return new constant(scale_); }
};

} // namespace weight_init
} // namespace tiny_cnn
