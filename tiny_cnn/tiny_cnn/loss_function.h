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
#include "product.h"

namespace tiny_cnn {

	// mean-squared-error loss function for regression
	class mse {
	public:
		static float_t f(float_t y, float_t t) {
			return (y - t) * (y - t) / 2;
		}

		static float_t df(float_t y, float_t t) {
			return y - t;
		}
	};


	// just a marker
	class hinge_loss {
	public:
		static float_t f(float_t y, float_t t) {
			return 1;
		}

		static float_t df(float_t y, float_t t) {
			return 1;
		}
	};

	// cross-entropy loss function for (multiple independent) binary classifications
	class cross_entropy {
	public:
		static float_t f(float_t y, float_t t) {
			return -t * std::log(y) - (1.0 - t) * std::log(1.0 - y);
		}

		static float_t df(float_t y, float_t t) {
			return (y - t) / (y * (1 - y));
		}
	};

	// cross-entropy loss function for multi-class classification
	class cross_entropy_multiclass {
	public:
		static float_t f(float_t y, float_t t) {
			return -t * std::log(y);
		}

		static float_t df(float_t y, float_t t) {
			return -t / y;
		}
	};

	template <typename E>
	vec_t gradient(const vec_t& y, const vec_t& t) {
		vec_t grad(y.size());
		assert(y.size() == t.size());

		for (size_t i = 0; i < y.size(); i++)
			grad[i] = E::df(y[i], t[i]);

		return grad;
	}



	static inline float_t cosine(const vec_t& x1, const vec_t& x2, unsigned int size){

		return vectorize::dot(&x1[0], &x2[0], size) / (sqrt(vectorize::dot(&x1[0], &x1[0], size))*sqrt(vectorize::dot(&x2[0], &x2[0], size)));
	}



	static inline void v_zeros(vec_t& x1, unsigned int size){
		std::fill(x1.begin(), x1.end(), 0);
	}

	static inline void fill_derivs(vec_t& out, const vec_t& a, const vec_t& b, unsigned int size, float_t minus){
		float_t a_norm = sqrt(vectorize::dot(&a[0], &a[0], size));
		float_t b_norm = sqrt(vectorize::dot(&b[0], &b[0], size));
		float_t scalar = vectorize::dot(&a[0], &b[0], size);
		float_t a_norm_sqr = a_norm*a_norm;
		float_t a_norm_cub = minus*a_norm_sqr*a_norm*b_norm;
		if (std::abs(a_norm_cub) < 1e-10){
			v_zeros(out, size);
			std::cout << "warning ";
		}
		for (int i = 0; i < size; i++){
			out[i] = (b[i] * a_norm_sqr - scalar*a[i]) / (a_norm_cub);
		}
	}

	static inline void fill_derivs_l2(vec_t& out, const vec_t& a, const vec_t& b, unsigned int size, float_t minus){
		for (int i = 0; i < size; i++){
			out[i] = 2 * minus * (a[i] - b[i]);
		}
	}

	static inline float_t l2_norm(const vec_t& a, const vec_t& b, unsigned int size){
		float_t sum = 0;
		for (int i = 0; i < size; i++){
			sum += (a[i] - b[i])*(a[i] - b[i]);
		}
		return sum;
	}

} // namespace tiny_cnn
