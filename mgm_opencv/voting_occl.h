#ifndef VOTING_OCCL_H_
#define VOTING_OCCL_H_
#include "opencv2/opencv.hpp"
#include <vector>
#include <unordered_map>
#include "helper.h"
#include "img_tools.h"

#define sigma_d (12.*12.)
#define sigma_i (7.*7.)

#define WINDOW_SIZE_X (18)
#define WINDOW_SIZE_Y (12)
#define WINDOW_SIZE_INITIAL_X (20)
#define WINDOW_SIZE_INITIAL_Y (20)
#define ITER (13)

inline float weight_func(int i, int j, int i2, int j2, const cv::Mat& mat){
	float* data = (float*)mat.data;
	return exp(-(pow(i - i2, 2) + pow(j - j2, 2)) / sigma_d -
		(pow(data[i*mat.cols*3 + j*3] - data[i2*mat.cols*3 + j2*3], 2) + pow(data[i*mat.cols*3 + j*3 + 1] - data[i2*mat.cols*3 + j2*3 + 1], 2) +
		pow(data[i*mat.cols*3 + j*3 + 2] - data[i2*mat.cols*3 + j2*3 + 2], 2)) / sigma_i);
}



void initial_support_and_decision(cv::Mat& init_disp, cv::Mat& cur_disp, cv::Mat& cur_probobil, const std::vector<std::pair<int, int>>&  occluded,
	const cv::Mat& mask, const cv::Mat& in, const cv::Mat& detmask){
#pragma omp parallel for
	for (int occl_i = 0; occl_i < occluded.size(); occl_i++){
		const auto& occl = occluded[occl_i];
		int top = std::max(0, occl.first - WINDOW_SIZE_INITIAL_Y / 2);
		int bottom = std::min(in.rows - 1, occl.first + WINDOW_SIZE_INITIAL_Y / 2);
		int left = std::max(0, occl.second - WINDOW_SIZE_INITIAL_X / 2);
		int right = std::min(in.cols - 1, occl.second + WINDOW_SIZE_INITIAL_X / 2);
		std::unordered_map<int, float> probabil;
		for (int i = top; i <= bottom; i++){
			for (int j = left; j <= right; j++){
				if (mask.data[i*mask.cols + j] == 255 && detmask.data[i*detmask.cols + j] != MASK_OUT){
					int disp = round(((float*)init_disp.data)[i*mask.cols + j]);
					probabil[disp] += weight_func(occl.first, occl.second, i, j, in);
				}
			}
		}
		if (probabil.size() == 0){
			continue;
		}

		float currentMax = -INFINITY;
		int arg_max = 0;
		for (auto it = probabil.cbegin(); it != probabil.cend(); ++it)
		{
			if (it->second > currentMax) {
				arg_max = it->first;
				currentMax = it->second;
			}
		}
		if (std::isfinite(currentMax)){
			((float*)cur_probobil.data)[occl.first*cur_probobil.cols + occl.second] = currentMax;
			((float*)cur_disp.data)[occl.first*cur_disp.cols + occl.second] = arg_max;
		}


	}
}

void iter_support_and_decision(const cv::Mat& prev_disp, const cv::Mat& prev_probobil, cv::Mat& cur_disp, cv::Mat& cur_probobil,
	const std::vector<std::pair<int, int>>&  occluded, const cv::Mat& mask, const cv::Mat& in, const cv::Mat& detmask){
#pragma omp parallel for
	for (int occl_i = 0; occl_i < occluded.size(); occl_i++){
		const auto& occl = occluded[occl_i];
		int top = std::max(0, occl.first - WINDOW_SIZE_Y / 2);
		int bottom = std::min(in.rows - 1, occl.first + WINDOW_SIZE_Y / 2);
		int left = std::max(0, occl.second - WINDOW_SIZE_X / 2);
		int right = std::min(in.cols - 1, occl.second + WINDOW_SIZE_X / 2);
		std::unordered_map<int, std::pair<float, float>> probabil;
		for (int i = top; i <= bottom; i++){
			for (int j = left; j <= right; j++){
				if (mask.data[i*mask.cols + j] != 255 && std::isfinite(((float*)prev_disp.data)[i*prev_disp.cols + j])
					&& !(i == occl.first && j == occl.second) && detmask.data[i*detmask.cols + j] != MASK_OUT){
					int disp = round(((float*)prev_disp.data)[i*prev_disp.cols + j]);
					float w = weight_func(occl.first, occl.second, i, j, in);
					probabil[disp].first += w*((float*)prev_probobil.data)[i*prev_probobil.cols + j];
					probabil[disp].second += w;
				}
			}
		}

		if (probabil.size() == 0){
			continue;
		}

		float currentMax = -INFINITY;
		int arg_max = 0;
		for (auto it = probabil.cbegin(); it != probabil.cend(); ++it)
		{
			if (it->second.first > currentMax) {
				arg_max = it->first;
				currentMax = it->second.first;
			}
		}
		if (std::isfinite(currentMax) && probabil[arg_max].second > 0){
			((float*)cur_probobil.data)[occl.first*cur_probobil.cols + occl.second] = currentMax / probabil[arg_max].second;
			((float*)cur_disp.data)[occl.first*cur_disp.cols + occl.second] = arg_max;
		}

	}
}

void voting_occl_refine(cv::Mat& map, const cv::Mat& mask, const cv::Mat& in, const cv::Mat& detmask){
	if (in.channels() != 3){
		std::cout << "smth goes wrong with input img";
	}
	cv::Mat white_mask;
	cv::inRange(mask, 255, 255, white_mask);
	int count = cv::countNonZero(white_mask);

	cv::Mat init_disp = map.clone();
	//cv::Mat init_probobil(map.rows, map.cols, CV_32FC1, cv::Scalar(0));
	std::vector<std::pair<int, int>> occluded;
	occluded.reserve(count);
	for (int i = 0; i < mask.rows; i++){
		for (int j = 0; j < mask.cols; j++){
			if (mask.data[i*mask.cols + j] != 255 && detmask.data[i*detmask.cols + j] != MASK_OUT){
				occluded.push_back(std::move(std::make_pair(i, j)));
				((float*)init_disp.data)[i*init_disp.cols + j] = NAN;
			}
		}
	}

	cv::Mat cur_init_probobil(map.rows, map.cols, CV_32FC1, cv::Scalar(0));
	cv::Mat cur_init_disp = init_disp.clone();
	initial_support_and_decision(init_disp, cur_init_disp, cur_init_probobil, occluded, mask, in, detmask);

	//iio_write_norm_vector_split((std::string)"data2/Motorcycle/before" + std::to_string(rand()) + ".png", cur_init_disp);

	cv::Mat prev_iter_probobil = cur_init_probobil;
	cv::Mat prev_iter_disp = cur_init_disp;
	for (int iter = 0; iter < ITER; iter++){
		cv::Mat cur_iter_probobil = prev_iter_probobil.clone();
		cv::Mat cur_iter_disp = prev_iter_disp.clone();

		iter_support_and_decision(prev_iter_disp, prev_iter_probobil, cur_iter_disp, cur_iter_probobil, occluded, mask, in, detmask);

		prev_iter_probobil = cur_iter_probobil;
		prev_iter_disp = cur_iter_disp;
	}

	map = prev_iter_disp;
}

#endif