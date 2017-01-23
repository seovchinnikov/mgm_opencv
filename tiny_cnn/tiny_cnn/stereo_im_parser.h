#pragma once

#include "util.h"
#include <fstream>
#include <cstdint>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include "pfm.h"
#include <random>
#include <iostream>

#define IMAGE_DEPTH (1)
#define IMAGE_WIDTH (11)
#define IMAGE_HEIGHT (11)
#define IMAGE_AREA (IMAGE_WIDTH*IMAGE_HEIGHT)
#define IMAGE_SIZE (IMAGE_AREA*IMAGE_DEPTH)
#define IMAGE_OUT (64)

#define NEG_LOW (2)
#define NEG_HIGH (6)
#define FREQ (3)


namespace tiny_cnn {

	struct stereoPair{
		std::string left_src;
		std::string right_src;
		std::string disp;
		std::string mask;

		stereoPair(std::string left_src, std::string right_src, std::string disp, std::string mask){
			this->left_src = left_src;
			this->right_src = right_src;
			this->disp = disp;
			this->mask = mask;
		}
	};

	struct trainPair{
		std::string left_dst;
		std::string right_dst;
		std::string labels;
		std::string vec_size;

		trainPair(std::string left_dst, std::string right_dst, std::string labels, std::string vec_size){
			this->left_dst = left_dst;
			this->right_dst = right_dst;
			this->labels = labels;
			this->vec_size = vec_size;
		}
	};

	bool saveArray(const float_t* pdata, size_t length, const std::string& file_path)
	{
		std::ofstream os(file_path.c_str(), std::ios::binary | std::ios::out);
		if (!os.is_open())
			return false;
		os.write(reinterpret_cast<const char*>(pdata), std::streamsize(length*sizeof(float_t)));
		os.close();
		return true;
	}

	bool loadArray(float_t* pdata, size_t length, const std::string& file_path)
	{
		std::ifstream is(file_path.c_str(), std::ios::binary | std::ios::in);
		if (!is.is_open())
			return false;
		is.read(reinterpret_cast<char*>(pdata), std::streamsize(length*sizeof(float_t)));
		is.close();
		return true;
	}

	bool saveArray(const label_t* pdata, size_t length, const std::string& file_path)
	{
		std::ofstream os(file_path.c_str(), std::ios::binary | std::ios::out);
		if (!os.is_open())
			return false;
		os.write(reinterpret_cast<const char*>(pdata), std::streamsize(length*sizeof(label_t)));
		os.close();
		return true;
	}

	bool loadArray(label_t* pdata, size_t length, const std::string& file_path)
	{
		std::ifstream is(file_path.c_str(), std::ios::binary | std::ios::in);
		if (!is.is_open())
			return false;
		is.read(reinterpret_cast<char*>(pdata), std::streamsize(length*sizeof(label_t)));
		is.close();
		return true;
	}

	void read_images(trainPair& pair, std::vector<vec_t>& array_left, std::vector<vec_t>& array_right, std::vector<vec_t>& array_other, std::vector<label_t>& labels){
		size_t size = 0;
		std::ifstream vec_size(pair.vec_size);
		vec_size >> size;

		{
			array_left = std::vector<vec_t>(size / IMAGE_AREA);
			array_right = std::vector<vec_t>(size / IMAGE_AREA);
			array_other = std::vector<vec_t>(size / IMAGE_AREA);
			float_t* temp_l, *temp_r;
			temp_l = new float_t[size];
			temp_r = new float_t[size];
			loadArray(temp_l, size, pair.left_dst);
			loadArray(temp_r, size, pair.right_dst);
			for (int i = 0; i < size / IMAGE_AREA; i++){
				//vec_t subimg_l(IMAGE_AREA), subimg_r(IMAGE_AREA);
				vec_t& subimg_l = array_left.at(i);
				vec_t& subimg_r = array_right.at(i);
				vec_t& subimg_o = array_other.at(i);
				int other_i_l;
				if (i % 2 == 0){
					other_i_l = i + 1;
				}
				else{
					other_i_l = i - 1;
				}
				subimg_l.insert(subimg_l.end(), temp_l + i*IMAGE_AREA, temp_l + (i + 1)*IMAGE_AREA);
				subimg_r.insert(subimg_r.end(), temp_r + i*IMAGE_AREA, temp_r + (i + 1)*IMAGE_AREA);
				subimg_o.insert(subimg_o.end(), temp_r + other_i_l*IMAGE_AREA, temp_r + (other_i_l + 1)*IMAGE_AREA);
				//array_left.push_back(subimg_l);
				//array_right.push_back(subimg_r);

			}
			delete[] temp_l;
			delete[] temp_r;
		}
		labels = std::vector<label_t>(size);
		loadArray(labels.data(), size, pair.labels);
	}

	std::vector<trainPair> create_images(const std::vector<stereoPair>& src,
		float_t scale_min,
		float_t scale_max)
	{
		std::default_random_engine generator;
		std::vector<trainPair> res;
		for (const stereoPair& pair : src){
			cv::Mat left = cv::imread(pair.left_src);
			cv::Mat right = cv::imread(pair.right_src);
			cv::Mat mask = cv::imread(pair.mask, CV_LOAD_IMAGE_GRAYSCALE);
			int gt_w = 0, gt_h = 0;
			float* gt_vec = read_pfm_file(pair.disp.c_str(), &gt_w, &gt_h);
			cv::Mat gt_im = cv::Mat(gt_h, gt_w, CV_32FC1, gt_vec);
			cv::flip(gt_im, gt_im, 0);

			cv::Mat left_g, right_g;
			cv::cvtColor(left, left_g, CV_BGR2GRAY);
			cv::cvtColor(right, right_g, CV_BGR2GRAY);

			cv::Mat left_n, right_n;

			//normalize(left_g, left_n, 0., 255., cv::NORM_MINMAX, CV_64F);
			//normalize(right_g, right_n, 0., 255., cv::NORM_MINMAX, CV_64F);
			left_g.convertTo(left_n, CV_64F);
			right_g.convertTo(right_n, CV_64F);
			double mean1, std1, mean2, std2;
			cv::Scalar mean1s, std1s, mean2s, std2s;
			cv::meanStdDev(left_n, mean1s, std1s);
			cv::meanStdDev(right_n, mean2s, std2s);
			mean1 = mean1s[0], std1 = std1s[0], mean2 = mean2s[0], std2 = std2s[0];
			subtract(left_n, mean1, left_n);
			subtract(right_n, mean2, right_n);
			left_n /= std1;
			right_n /= std2;

			vec_t array_left, array_right;
			std::vector<label_t> labels;

			std::uniform_int_distribution<int> int_x_displ(0, FREQ - 1);
			std::uniform_int_distribution<int> int_y_displ(0, FREQ - 1);
			std::default_random_engine generator2;
			int x_displ = int_x_displ(generator2);
			int y_displ = int_y_displ(generator2);
			for (int y = IMAGE_HEIGHT / 2 + y_displ; y < left_g.rows - IMAGE_HEIGHT / 2; y += FREQ){
				for (int x = IMAGE_WIDTH / 2 + x_displ; x < left_g.cols - IMAGE_WIDTH / 2; x += FREQ){
					if (!std::isfinite(((float*)gt_im.data)[y*left_g.cols + x]) || (mask.data)[y*left_g.cols + x] !=255){
						continue;
					}
					int disp = round(((float*)gt_im.data)[y*left_g.cols + x]);
					int x2 = x - disp;
					if (x2 >= right_g.cols - IMAGE_WIDTH / 2 || x2 < IMAGE_WIDTH / 2){
						continue;
					}

					cv::Rect rect(x - IMAGE_WIDTH / 2, y - IMAGE_HEIGHT / 2, IMAGE_WIDTH, IMAGE_HEIGHT);
					cv::Mat subimg(left_n, rect);
					for (int i = 0; i < subimg.rows; ++i) {
						array_left.insert(array_left.end(), (float_t*)subimg.ptr<uchar>(i), (float_t*)subimg.ptr<uchar>(i)+subimg.cols);
					}
					for (int i = 0; i < subimg.rows; ++i) {
						array_left.insert(array_left.end(), (float_t*)subimg.ptr<uchar>(i), (float_t*)subimg.ptr<uchar>(i)+subimg.cols);
					}

					cv::Rect rect2(x2 - IMAGE_WIDTH / 2, y - IMAGE_HEIGHT / 2, IMAGE_WIDTH, IMAGE_HEIGHT);
					cv::Mat subimg2(right_n, rect2);
					for (int i = 0; i < subimg2.rows; ++i) {
						array_right.insert(array_right.end(), (float_t*)subimg2.ptr<uchar>(i), (float_t*)subimg2.ptr<uchar>(i)+subimg2.cols);
					}
					labels.push_back(1);
					labels.push_back(0);


					int left_false, right_false;
					if (abs(x2 - left_g.cols - IMAGE_WIDTH / 2 - 1) < 20){
						left_false = std::max(IMAGE_WIDTH / 2, x2 - NEG_HIGH);
						right_false = std::max(IMAGE_WIDTH / 2, x2 - NEG_LOW);
					}
					else if (abs(x2 - IMAGE_WIDTH / 2) < 20){
						left_false = std::min(left_g.cols - IMAGE_HEIGHT / 2 - 1, x2 + NEG_LOW);
						right_false = std::min(left_g.cols - IMAGE_HEIGHT / 2 - 1, x2 + NEG_HIGH);
					}
					else
					if (rand() % 2){
						left_false = std::max(IMAGE_WIDTH / 2, x2 - NEG_HIGH);
						right_false = std::max(IMAGE_WIDTH / 2, x2 - NEG_LOW);
					}
					else{
						left_false = std::min(left_g.cols - IMAGE_WIDTH / 2 - 1, x2 + NEG_LOW);
						right_false = std::min(left_g.cols - IMAGE_WIDTH / 2 - 1, x2 + NEG_HIGH);
					}
					std::uniform_int_distribution<int> distribution(left_false, right_false);
					int x3 = distribution(generator);

					cv::Rect rect3(x3 - IMAGE_WIDTH / 2, y - IMAGE_HEIGHT / 2, IMAGE_WIDTH, IMAGE_HEIGHT);
					cv::Mat subimg3(right_n, rect3);
					for (int i = 0; i < subimg3.rows; ++i) {
						array_right.insert(array_right.end(), (float_t*)subimg3.ptr<uchar>(i), (float_t*)subimg3.ptr<uchar>(i)+subimg3.cols);
					}
				}
			}

			trainPair train(pair.left_src + ".byte", pair.right_src + ".byte", pair.disp + ".byte", pair.disp + ".txt");
			saveArray(array_left.data(), array_left.size(), train.left_dst);
			saveArray(array_right.data(), array_right.size(), train.right_dst);
			saveArray(labels.data(), labels.size(), train.labels);
			std::ofstream vec_size(train.vec_size);
			vec_size << array_left.size();
			vec_size.close();
			res.push_back(train);
		}

		return res;
	}

}