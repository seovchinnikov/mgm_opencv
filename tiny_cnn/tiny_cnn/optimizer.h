/*
    Copyright (c) 2013, Taiga Nomi
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
#include <unordered_map>

namespace tiny_cnn {

/**
 * base class of optimizer
 * usesHessian : true if an optimizer uses hessian (2nd order derivative of loss function)
 **/
template <bool usesHessian>
struct optimizer {
    bool requires_hessian() const { return usesHessian; } // vc2012 doesn't support constexpr
    virtual void reset() {} // override to implement pre-learning action

};

// helper class to hold N values for each weight
template <typename value_t, int N, bool usesHessian = false>
struct stateful_optimizer : public optimizer<usesHessian> {
    void reset() override {
        for (auto& e : E_) e.clear();
    }
protected:
    template <int Index>
    std::vector<value_t>& get(const vec_t& key) {
        static_assert(Index < N, "index out of range");
        if (E_[Index][&key].empty())
            E_[Index][&key].resize(key.size(), value_t());
        return E_[Index][&key];
    }
    std::unordered_map<const vec_t*, std::vector<value_t>> E_[N];
};

/**
 * Stochastic Diagonal Levenberg-Marquardt
 *
 * Y LeCun, L Bottou, Y Bengio, and P Haffner,
 * Gradient-based learning applied to document recognition
 * Proceedings of the IEEE, 86, 2278-2324.
 **/
struct gradient_descent_levenberg_marquardt : public optimizer<true> {
public:
    gradient_descent_levenberg_marquardt() : alpha(0.00085), mu(0.02) {}

    void update(const vec_t& dW, const vec_t& Hessian, vec_t& W) {
        for_i(W.size(), [&](int i){
            W[i] = W[i] - (alpha / (Hessian[i] + mu)) * dW[i];
        });
    }

    float_t alpha; // learning rate
    float_t mu; // constant to prevent step size from becoming too large when H is small
};

/**
 * adaptive gradient method
 *
 * J Duchi, E Hazan and Y Singer,
 * Adaptive subgradient methods for online learning and stochastic optimization
 * The Journal of Machine Learning Research, pages 2121-2159, 2011.
 **/
struct adagrad : public stateful_optimizer<float_t, 1, false> {
    adagrad() : alpha(0.01), eps(1e-8) {}

    void update(const vec_t& dW, const vec_t& /*Hessian*/, vec_t &W) {
        vec_t& g = get<0>(W);

        for_i(W.size(), [&](int i) {
            g[i] += dW[i] * dW[i];
            W[i] -= alpha * dW[i] / (std::sqrt(g[i]) + eps);
        });
    }

    float_t alpha; // learning rate
private:
    float_t eps;
};

/**
 * RMSprop
 *
 * T Tieleman, and G E Hinton,
 * Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine Learning (2012)
 **/
struct RMSprop : public stateful_optimizer<float_t, 1, false> {
    RMSprop() : alpha(0.0001), mu(0.99), eps(1e-8) {}

    void update(const vec_t& dW, const vec_t& /*Hessian*/, vec_t& W) {
        vec_t& g = get<0>(W);

        for_i(W.size(), [&](int i)
        {
            g[i] = mu * g[i] + (1 - mu) * dW[i] * dW[i];
            W[i] -= alpha * dW[i] / std::sqrt(g[i] + eps);
        });
    }

    float_t alpha; // learning rate
    float_t mu; // decay term
private:
    float_t eps; // constant value to avoid zero-division
};


/**
 * @brief [a new optimizer (2015)]
 * @details [see Adam: A Method for Stochastic Optimization (Algorithm 1)
 *               http://arxiv.org/abs/1412.6980]
 * 
 */
struct adam : public stateful_optimizer<float_t, 2, false> {
    adam() : alpha(0.001), b1(0.9), b2(0.999) , b1_t(0.9), b2_t(0.999), eps(1e-8) {}

    void update(const vec_t& dW, const vec_t& /*Hessian*/, vec_t& W) {
        vec_t& mt = get<0>(W);
        vec_t& vt = get<1>(W);

        b1_t*=b1;b2_t*=b2;

        for_i(W.size(), [&](int i){
            mt[i] = b1 * mt[i] + (1 - b1) * dW[i];
            vt[i] = b2 * vt[i] + (1 - b2) * dW[i] * dW[i];

            W[i] -= alpha * ( mt[i]/(1-b1_t) ) / std::sqrt( (vt[i]/(1-b2_t)) + eps);
        });
    }

    float_t alpha; // learning rate
    float_t b1; // decay term
    float_t b2; // decay term
    float_t b1_t; // decay term power t
    float_t b2_t; // decay term power t   
private:
    float_t eps; // constant value to avoid zero-division
};



/**
 * SGD without momentum
 *
 * slightly faster than tiny_cnn::momentum
 **/
struct gradient_descent : public optimizer<false> {
    gradient_descent() : alpha(0.01), lambda(0.0) {}

    void update(const vec_t& dW, const vec_t& /*Hessian*/, vec_t& W) {
        for_i(W.size(), [&](int i){
            W[i] = W[i] - alpha * (dW[i] + lambda * W[i]);
        });
    }

    float_t alpha; // learning rate
    float_t lambda; // weight decay
};

/**
 * SGD with momentum
 *
 * B T Polyak,
 * Some methods of speeding up the convergence of iteration methods
 * USSR Computational Mathematics and Mathematical Physics, 4(5):1-17, 1964.
 **/
struct momentum : public stateful_optimizer<float_t, 1, false> {
public:
    momentum() : alpha(0.01), lambda(0.0), mu(0.9) {}

    void update(const vec_t& dW, const vec_t& /*Hessian*/, vec_t& W) {
        vec_t& dWprev = get<0>(W);

        for_i(W.size(), [&](int i){
            float_t V = mu * dWprev[i] - alpha * (dW[i] + W[i] * lambda);
            W[i]      += V;
            dWprev[i] =  V;
        });
    }


    float_t alpha; // learning rate
    float_t lambda; // weight decay
    float_t mu; // momentum
};

} // namespace tiny_cnn
