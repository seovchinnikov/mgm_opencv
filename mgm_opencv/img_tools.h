/* Copyright (C) 2015, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>,
 *                     Carlo de Franchis <carlo.de-franchis@ens-cachan.fr>,
 *                     Enric Meinhardt <enric.meinhardt@cmla.ens-cachan.fr>*/
//// a structure to wrap images 
#ifndef IMG_TOOLS_H_
#define IMG_TOOLS_H_

#include <algorithm>
#include "point.h"
#include "opencv2/opencv.hpp"
#include "pfm.h"
#include "helper.h"

/************ IMG IO  **************/

cv::Mat iio_read_vector_split(char *nm)
{
	cv::Mat out;
	cv::Mat in = cv::imread(nm);
	in.convertTo(out, CV_32F);
	return out;
}


cv::Mat conj_masks(const cv::Mat& mask1, const cv::Mat& mask2){
	cv::Mat res = mask1.clone();
	if (!(mask1.channels() == 1 && mask2.channels() == 1)){
		fprintf(stderr, "invalid params - channels number");
		return res;
	}
	if (!(mask1.isContinuous() && mask2.isContinuous())){
		fprintf(stderr, "invalid params - not continuous");
		return res;
	}

	for (int i = 0; i < mask1.rows*mask1.cols; i++){
		if ((mask2.data)[i] == 0){
			(res.data)[i] = 0;
		}
		else if ((mask2.data)[i] != 255 && (mask1.data)[i] != 0){
			(res.data)[i] = (mask2.data)[i];
		}
	}
	return res;
}

cv::Mat refine_mask(const cv::Mat& mask){
	cv::Mat res = mask.clone();
	uchar* res_data = res.data;
	uchar* data = mask.data;

	for (int i = 0; i < mask.rows*mask.cols; i++){
		if ((i - 1 >= 0 && data[i - 1] != 255 && i % mask.cols != 0) || (i + 1 < mask.rows*mask.cols && data[i + 1] != 255 && (i + 1) % mask.cols != 0)
			|| (i - mask.cols >= 0 && data[i - mask.cols] != 255) || (i + mask.cols < mask.rows*mask.cols && data[i + mask.cols] != 255)){
			res_data[i] = 122;
		}

	}
	return res;
}

void iio_write_vector_split(char *nm, const cv::Mat& out)
{
	//cv::Mat in;
	//out.convertTo(in, CV_8U);
	cv::imwrite(nm, out);
}

void iio_write_pfm(std::string nm, const cv::Mat& out)
{
	//cv::Mat in;
	//out.convertTo(in, CV_8U);
	cv::Mat temp;
	cv::flip(out, temp, 0);
	temp *= -1.;
	write_pfm_file(nm.c_str(), (float*)temp.data, temp.cols, temp.rows);
}

void iio_write_norm_vector_split(std::string nm, const cv::Mat& out)
{
	//cv::Mat in;
	//out.convertTo(in, CV_8U);
	cv::Mat res;
	//write_pfm_file(nm, (float*)out.data, out.cols, out.rows);
	cv::normalize(out, res, 0, 255, CV_MINMAX, -1);
	cv::imwrite(nm, res);
}


void remove_nonfinite_values_Img(cv::Mat &u, float newval)
{
	float* data = (float*)u.data;
	for (int i = 0; i < u.total()*u.channels(); i++)
	if (!isfinite(data[i]))
		data[i] = newval;
}


/************ IMG ACCESS **************/

inline float val(const cv::Mat& u, const Point p, int ch) {
	int x = p.x;
	int y = p.y;
	return ((float*)u.data)[x*u.channels() + u.cols*y*u.channels() + ch];
}

inline float val(const cv::Mat& u, int x, int y, int ch) {
	return ((float*)u.data)[x*u.channels() + u.cols*y*u.channels() + ch];
}

//inline float val(const float* d, int x, int y, int ch, const cv::Mat& u) {
//	return d[x*u.channels() + u.cols*y*u.channels() + ch];
//}

inline void setVal(cv::Mat& u, int x, int y, int ch, float val) {
	((float*)u.data)[x*u.channels() + u.cols*y*u.channels() + ch] = val;
}

//inline void setVal(float* d, int x, int y, int ch, cv::Mat& u, float val) {
//	d[x*u.channels() + u.cols*y*u.channels() + ch] = val;
//}

inline int check_inside_image(const Point p, const cv::Mat& u) {
	int nx = u.cols;
	int ny = u.rows;
	float x = p.x;
	float y = p.y;
	if (x >= 0 && y >= 0 && x < nx && y < ny) return 1;
	else return 0;
}

inline int check_inside_mask(const Point p, const cv::Mat& mask) {

	int x = p.x;
	int y = p.y;
	if ((mask.data)[mask.cols*y + x] != MASK_OUT){
		return 1;
	}
	else{
		return 0;
	}
}

inline int check_inside_mask(int x, int y, const cv::Mat& mask) {

	if ((mask.data)[mask.cols*y + x] != MASK_OUT){
		return 1;
	}
	else{
		return 0;
	}
}

inline float valnan(const cv::Mat& u, const Point p, int ch)
{
	return check_inside_image(p, u) ? val(u, p, ch) : NAN;
}

inline float valzero(const cv::Mat& u, const Point p, int ch)
{
	return check_inside_image(p, u) ? val(u, p.x, p.y, ch) : 0;
}

inline float valzero(const cv::Mat& u, const int x, const int y, int ch)
{
	return check_inside_image(Point(x, y), u) ? val(u, x, y, ch) : 0;
}

inline float valneumann(const cv::Mat& u, const int x, const int y, int ch)
{
	int xx = x, yy = y;
	xx = x >= 0 ? xx : 0;
	xx = x < u.cols ? xx : u.cols - 1;
	yy = y >= 0 ? yy : 0;
	yy = y < u.rows ? yy : u.rows - 1;
	return val(u, xx, yy, ch);
}



/************ IMG PROC **************/

cv::Mat compute_insensity_image(const cv::Mat& u) {
	int nx = u.cols;
	int ny = u.rows;
	int nch = u.channels();
	cv::Mat Intensity = cv::Mat::zeros(ny, nx, CV_32FC1);
	//cv::cvtColor(u, Intensity, CV_RGB2GRAY);

	float* dst = (float*)Intensity.data;

	//for (int i = 0; i < nx*ny; i++) dst[i] = 0;

	for (int j = 0; j < ny; j++){
		for (int i = 0; i < nx; i++){
			for (int c = 0; c < nch; c++){
				dst[i + j*nx] += val(u, i, j, c);
			}
			dst[i + j*nx] /= nch;
		}
	}

	return Intensity;
}

cv::Mat apply_filter(const cv::Mat& u, const cv::Mat& filter) {
	cv::Mat out = u.clone();
	int hfnx = filter.cols / 2;
	int hfny = filter.rows / 2;
	int hfnch = filter.channels() / 2;

	int channels = filter.channels();

	for (int c = 0; c < u.channels(); c++)
	for (int j = 0; j < u.rows; j++)
	for (int i = 0; i < u.cols; i++) {
		float v = 0;
		for (int jj = 0; jj < filter.rows; jj++)
		for (int ii = 0; ii < filter.cols; ii++)
		for (int cc = 0; cc < channels; cc++){
			v += valneumann(u, i + ii - hfnx,
				j + jj - hfny,
				c + cc - hfnch) *
				val(filter, ii, jj, cc);
		}
		setVal(out, i, j, c, v);
	}

	return out;
}

cv::Mat apply_filter(const cv::Mat& u, float ff[], int fnx, int fny, int fnc) {
	cv::Mat f(fnx, fny, fnc);
	float* fdata = (float*)f.data;
	//struct Img f(fnx, fny, fnc);

	for (int i = 0; i < fnx*fny*fnc; i++) fdata[i] = ff[i];
	return apply_filter(u, f);
}

//struct Img sobel_x(struct Img &u) {
//   struct Img f(3,3,1);
//   float ff[] = {-1,0,1, -1,0,1, -1,0,1};
//   for(int i=0;i<9;i++) f[i] = ff[i];
//   return apply_filter(u,f);
//}



static float inline unnormalized_gaussian_function(float sigma, float x) {
	return exp(-x*x / (2 * sigma*sigma));
}

#define KWMAX 39
static inline int gaussian_kernel_width(float sigma) {
	float radius = 3 * fabs(sigma);
	int r = ceil(1 + 2 * radius);
	if (r < 1) r = 1;
	if (r > KWMAX) r = KWMAX;
	return r;
}

static void fill_gaussian_kernel(float *k, int w, int h, float s) {
	int cw = (w - 1) / 2;
	int ch = (h - 1) / 2;
	float m = 0;
	for (int j = 0; j < h; j++)
	for (int i = 0; i < w; i++) {
		float v = unnormalized_gaussian_function(s, hypot(i - cw, j - ch));
		k[j*w + i] = v;
		m += v;
	}
	for (int i = 0; i < w*h; i++)  {
		k[i] /= m;
	}
}

cv::Mat gblur_truncated(const cv::Mat& u, float sigma) {
	// determine the size of the kernel
	int rad = gaussian_kernel_width(sigma);
	//struct Img fx(rad, 1, 1), fy(1, rad, 1);
	cv::Mat fx(1, rad, CV_32FC1), fy(rad, 1, CV_32FC1);
	fill_gaussian_kernel((float*)fx.data, rad, 1, sigma);
	fill_gaussian_kernel((float*)fy.data, 1, rad, sigma);
	cv::Mat tmp = apply_filter(u, fx);
	return apply_filter(tmp, fy);
}


std::pair<float, float> image_minmax(const cv::Mat &u){
	// returns global (finite) min and max of an image
	int nx = u.cols;
	int ny = u.rows;
	int nch = u.channels();
	float gmin = INFINITY; float gmax = -INFINITY;


	for (int j = 0; j < ny; j++)
	for (int i = 0; i < nx; i++)
	for (int c = 0; c < nch; c++){
		float v = val(u, i, j, c);
		if (isfinite(v)) {
			if (v < gmin) gmin = v;
			if (v > gmax) gmax = v;
		}
	}

	return std::pair<float, float>(gmin, gmax);
}

/// Median filter
cv::Mat median_filter(const cv::Mat &u, int radius) {
	cv::Mat M = u.clone();
	//int channels = u.channels();
	//struct Img M(u);
	int size = 2 * radius + 1;
	size *= size;
	std::vector<float> v(size);
	for (int y = 0; y < M.rows; y++)
	for (int x = 0; x < M.cols; x++)
	for (int k = 0; k < M.channels(); k++)
	{
		int n = 0;
		for (int j = -radius; j <= radius; j++){
			if (0 <= j + y && j + y < u.rows){
				for (int i = -radius; i <= radius; i++)
				if (0 <= i + x && i + x < u.cols && std::isfinite(val(M, i + x, j + y, k)))
					v[n++] = val(M, i + x, j + y, k);
			}
		}
		if (n == 0){
			v[n++] = std::numeric_limits<float>::quiet_NaN();
		}
		std::nth_element(v.begin(), v.begin() + n / 2, v.end());
		setVal(M, x, y, k, v[n / 2]);
	}
	return M;
}

void fill_small_holes(cv::Mat& in_mask){
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	cv::morphologyEx(in_mask, in_mask, cv::MORPH_OPEN, kernel);
	cv::Mat temp;
	cv::threshold(in_mask, temp, 254, 255, cv::THRESH_BINARY_INV);
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(temp, contours, hierarchy, cv::RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	for (int i = 0; i < contours.size(); i++) // iterate through each contour. 
	{
		// has no parent
		if (hierarchy[i][3] == -1 ) {
			//cv::drawContours(in_mask, contours, i, cv::Scalar(0), CV_FILLED, 8);
			double outer = contourArea(contours[i], false);
			int first_child = hierarchy[i][2];
			for (int child = first_child; child != -1; child = hierarchy[child][0]){
				double inner = contourArea(contours[child], false);
				if (outer / inner>140 && cv::arcLength(contours[child], false) < 100){
					cv::drawContours(in_mask, contours, child, cv::Scalar(122), CV_FILLED, 8);
					//std::cout << 33333;
				}

			}
		}
	}
}

#endif // IMG_TOOLS_H_
