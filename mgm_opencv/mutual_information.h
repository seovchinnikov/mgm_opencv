#define _SCL_SECURE_NO_WARNINGS 1

#ifndef MUTUAL_INF_H_
#define MUTUAL_INF_H_

#include "opencv2/opencv.hpp"

static const float sigma = 1.;

/* dont use it */
cv::Mat indiv_probobil_wrong(const cv::Mat& u, int channel){
	cv::Mat prob(1, 256, CV_32FC1);
	int size = u.cols*u.rows;
	int chs = u.channels();
	for (int i = 0; i < size; i++){
		((float*)prob.data)[(u.data)[i*chs + channel]]++;
	}
	for (int i = 0; i < 256; i++){
		((float*)prob.data)[i] /= size;
	}
	return prob;
}


cv::Mat mutual_probobil(const cv::Mat& u, const cv::Mat& map, const cv::Mat& v, int channel, int& corr_pixels, const cv::Mat& maskdet, const cv::Mat& maskdet2){
	cv::Mat prob = cv::Mat::zeros(256, 256, CV_32FC1);
	int size = u.cols*u.rows;
	int chs = u.channels();
	corr_pixels = 0;
	for (int i = 0; i < u.rows; i++){
		for (int j = 0; j < u.cols; j++){
			if (!check_inside_mask(j, i, maskdet)){
				continue;
			}
			int x = j + round(((float*)map.data)[i*map.cols + j]);
			if (!std::isfinite(((float*)map.data)[i*map.cols + j]) || x < 0 || x >= v.cols || !check_inside_mask(x, i, maskdet2)){
				continue;
			}
			corr_pixels++;
			((float*)prob.data)[256 * ((u.data)[i*chs*u.cols + j*chs + channel]) + (v.data)[i*chs*v.cols + x*chs + channel]]++;
		}
	}
	cv::FileStorage file("data2/Motorcycle/cost_mutual_probobil.xml", cv::FileStorage::WRITE);
	file << "Vocabulary" << prob;
	prob /= corr_pixels;

	return prob;
}

cv::Mat mutual_entropy(const cv::Mat& prob, int corr_pixels){
	cv::Mat entropy(256, 256, CV_32FC1);
	int size = entropy.cols*entropy.rows;
	//cv::FileStorage file("data2/Motorcycle/before_gaus_entr1.xml", cv::FileStorage::WRITE);
	//file << "Vocabulary" << prob;
	cv::GaussianBlur(prob, entropy, cv::Size(7, 7), sigma);
	//cv::FileStorage file2("data2/Motorcycle/after_gaus_entr1.xml", cv::FileStorage::WRITE);
	//file2 << "Vocabulary" << entropy;
	for (int i = 0; i < 256 * 256; i++){
		if (((float*)entropy.data)[i] == 0.){
			((float*)entropy.data)[i] = 1e-15;
		}
	}
	cv::log(entropy, entropy);
	//cv::FileStorage file3("data2/Motorcycle/after_log_entr1.xml", cv::FileStorage::WRITE);
	//file3 << "Vocabulary" << entropy;
	cv::GaussianBlur(entropy, entropy, cv::Size(7, 7), 1.0);
	//cv::FileStorage file4("data2/Motorcycle/after_blur2_entr1.xml", cv::FileStorage::WRITE);
	//file4 << "Vocabulary" << entropy;
	entropy = entropy*(-1. / corr_pixels);
	//cv::FileStorage file5("data2/Motorcycle/after_mult_entr1.xml", cv::FileStorage::WRITE);
	//file5 << "Vocabulary" << entropy;
	return entropy;
}

cv::Mat indiv_probobil_cols1(const cv::Mat& mut_probobil){
	cv::Mat prob = cv::Mat::zeros(1, 256, CV_32FC1);
	for (int i = 0; i < 256; i++){
		for (int j = 0; j < 256; j++){
			((float*)prob.data)[i] += ((float*)mut_probobil.data)[i*mut_probobil.cols + j];
		}
	}
	return prob;
}

cv::Mat indiv_probobil_rows2(const cv::Mat& mut_probobil){
	cv::Mat prob = cv::Mat::zeros(1, 256, CV_32FC1);
	for (int i = 0; i < 256; i++){
		for (int j = 0; j < 256; j++){
			((float*)prob.data)[j] += ((float*)mut_probobil.data)[i*mut_probobil.cols + j];
		}
	}
	return prob;
}

cv::Mat indiv_entropy(const cv::Mat& prob, int corr_pixels){
	cv::Mat entropy(1, 256, CV_32FC1);
	int size = entropy.cols*entropy.rows;
	//cv::FileStorage file("data2/Motorcycle/before_gaus_indentr1.xml", cv::FileStorage::WRITE);
	//file << "Vocabulary" << prob;
	cv::GaussianBlur(prob, entropy, cv::Size(7, 1), sigma);
	//cv::FileStorage file2("data2/Motorcycle/after_gaus_indentr1.xml", cv::FileStorage::WRITE);
	//file2 << "Vocabulary" << entropy;
	for (int i = 0; i < 256; i++){
		if (((float*)entropy.data)[i] == 0.){
			((float*)entropy.data)[i] = 1e-15;
		}
	}
	cv::log(entropy, entropy);
	//cv::FileStorage file3("data2/Motorcycle/after_log_indentr1.xml", cv::FileStorage::WRITE);
	//file3 << "Vocabulary" << entropy;
	cv::GaussianBlur(entropy, entropy, cv::Size(7, 1), sigma);
	//cv::FileStorage file4("data2/Motorcycle/after_blur2_indentr1.xml", cv::FileStorage::WRITE);
	//file4 << "Vocabulary" << entropy;
	entropy = entropy*(-1. / corr_pixels);
	//cv::FileStorage file5("data2/Motorcycle/after_mult_indentr1.xml", cv::FileStorage::WRITE);
	//file5 << "Vocabulary" << entropy;
	return entropy;
}

cv::Mat mutual_entropy_1_channel(const cv::Mat& u, const cv::Mat& map, const cv::Mat& v, cv::Mat& prob, int& corr_pixels, const cv::Mat& maskdet, const cv::Mat& maskdet2){
	prob = mutual_probobil(u, map, v, 0, corr_pixels, maskdet, maskdet2);
	return mutual_entropy(prob, corr_pixels);

}

void fill_cost_mat(cv::Mat& costs, const cv::Mat& u2, const cv::Mat& prev_out, const cv::Mat& v2, const cv::Mat& maskdet, const cv::Mat& maskdet2){
	int corr_pixels = 0;
	cv::Mat mutual_prob;
	cv::Mat mutual_entr1 = mutual_entropy_1_channel(u2, prev_out, v2, mutual_prob, corr_pixels, maskdet, maskdet2);
	cv::Mat indiv_entr1 = indiv_entropy(indiv_probobil_cols1(mutual_prob), corr_pixels);
	cv::Mat indiv_entr2 = indiv_entropy(indiv_probobil_rows2(mutual_prob), corr_pixels);
	//cv::FileStorage file("data2/Motorcycle/mutual_entr1.xml", cv::FileStorage::WRITE);
	//file << "Vocabulary" << mutual_entr1;
	//cv::FileStorage file2("data2/Motorcycle/indiv_entr1.xml", cv::FileStorage::WRITE);
	//file2 << "Vocabulary" << indiv_entr1;
	//cv::FileStorage file3("data2/Motorcycle/indiv_entr2.xml", cv::FileStorage::WRITE);
	//file3 << "Vocabulary" << indiv_entr2;
	for (int i = 0; i < 256; i++){
		for (int j = 0; j < 256; j++){
			((float*)costs.data)[i * 256 + j] = ((float*)mutual_entr1.data)[i * 256 + j] -
				((float*)indiv_entr1.data)[i] - ((float*)indiv_entr2.data)[j];
		}
	}
}

int partition(float* input, int p, int r)
{
	float pivot = input[r];
	while (p < r)
	{
		while (input[p] < pivot)
			p++;

		while (input[r] > pivot)
			r--;

		if (input[p] == input[r])
			p++;
		else if (p < r) {
			float tmp = input[p];
			input[p] = input[r];
			input[r] = tmp;
		}
	}

	return r;
}

float quick_select(float* input, int p, int r, int k)
{
	if (p == r) return input[p];
	int j = partition(input, p, r);
	int length = j - p + 1;
	if (length == k) return input[j];
	else if (k < length) return quick_select(input, p, j - 1, k);
	else  return quick_select(input, j + 1, r, k - length);
}


void winsorize_mat(cv::Mat& costs){
	int big_quantile = 256 * 256 * 0.95;
	int small_quantile = 256 * 256 * 0.05;
	float* inp1 = new float[256 * 256]; float* inp2 = new float[256 * 256];
	std::copy(((float*)costs.data), ((float*)costs.data) + 256*256, inp1);
	std::copy(((float*)costs.data), ((float*)costs.data) + 256*256, inp2);
	float smallest_val = quick_select(inp1, 0, 256 * 256, small_quantile);
	float biggest_val = quick_select(inp2, 0, 256 * 256, big_quantile);
	delete[] inp1;
	delete[] inp2;
	int k = 0;
	for (int i = 0; i < 256 * 256; i++){
		if (((float*)costs.data)[i] < smallest_val){
			((float*)costs.data)[i] = smallest_val;
		}
		if (((float*)costs.data)[i] > biggest_val){
			((float*)costs.data)[i] = biggest_val;
		}
	}
}



#endif