#ifndef OCCL_TOOLS_H_
#define OCCL_TOOLS_H_
#include <vector>
#include "opencv2/opencv.hpp"
#include "img_tools.h"
#include "point.h"
#include "occl_tools.h"
#include "helper.h"

SMART_PARAMETER_INT(OCCL_MISM_LINE_DETECTION, 1)

enum PixelType { OCCL, MISM };

enum InterpMainDir { BT, LR };

bool inline isnotfinite(float x){
	return !std::isfinite(x);
}

inline static int point_to_coord(Point p, int cols){
	return ((int)p.y)*cols + (int)p.x;
}

inline static int point_to_nd_coord(Point p, int cols, int ch, int nd){
	return nd*((int)p.y)*cols + nd * (int)p.x + ch;
}

inline static int point_to_nd_coord(int x, int y, int cols, int ch, int nd){
	return nd*y*cols + nd * x + ch;
}

bool inline len_is_not_too_long(Point cur, Point first, int max_len_x, int max_len_y){
	if (abs(cur.x - first.x) > max_len_x || abs(cur.y - first.y) > max_len_y){
		return false;
	}
	return true;
}

void fill_interp_matrix(cv::Mat& map, cv::Mat& interpolation, const cv::Mat& mask, const cv::Mat& detmask, InterpMainDir main_dir){
	std::vector<Point> dirs;
	if (main_dir == BT){
		dirs = { Point(0, 1), Point(0, -1), Point(1, 0), Point(-1, 0), Point(1, 1), Point(-1, 1), Point(1, -1), Point(-1, -1) };
	}
	else{
		dirs = { Point(1, 0), Point(-1, 0), Point(0, 1), Point(0, -1), Point(1, 1), Point(-1, 1), Point(1, -1), Point(-1, -1) };
	}
	int max_len_x = map.cols / 5;
	int max_len_y;
	if (main_dir == BT){
		max_len_y = 35;
	}
	else{
		max_len_y = map.rows / 5;
	}

	for (int i = 0; i < mask.total(); i++){
		float maxDist = std::numeric_limits<float>::infinity();
		for (int dir_counter = 0; dir_counter < dirs.size(); dir_counter++){
			const Point& dir = dirs[dir_counter];
			float propVal = NAN;
			if ((mask.data)[i] != 255 && (detmask.data)[i] != MASK_OUT){
				int len = 0;
				Point first(i % mask.cols, i / mask.cols);
				Point next = first + dir;
				float dist = 0;
				while (check_inside_image(next, mask) && len_is_not_too_long(next, first, max_len_x, max_len_y) && (dir_counter < 2 || dist < maxDist)){
					if ((mask.data)[point_to_coord(next, mask.cols)] == 255){
						break;
					}
					dist++;
					next = next + dir;
				}
				if (check_inside_image(next, mask) && len_is_not_too_long(next, first, max_len_x, max_len_y) && (dir_counter < 2 || dist < maxDist)){
					propVal = ((float*)map.data)[point_to_coord(next, mask.cols)];
				}
				if (dir_counter < 2 && std::isfinite(propVal)){
					maxDist = std::isfinite(maxDist) ? std::fmax(maxDist, 1.35*dist) : 1.35*dist;
				}
				Point opp_dir = Point(0, 0) - dir;
				next += opp_dir;
				Point end = first + opp_dir;
				while (next != end){
					/*if (!std::isfinite(propVal)){
						((float*)interpolation.data)[point_to_nd_coord(next, mask.cols, dir_counter, 8)] = ((float*)map.data)[point_to_coord(next, mask.cols)];
						}
						else{*/
					((float*)interpolation.data)[point_to_nd_coord(next, mask.cols, dir_counter, 8)] = propVal;
					//}
					//std::cout << ((float*)interpolation.data)[point_to_nd_coord(next, mask.cols, dir_counter, 8)];
					next += opp_dir;
				}
			}
		}
	}
}


float inline median(float* data, int size){
	//if (size % 2 == 0){
	//	return 0.5*(data[size / 2 - 1] + data[size / 2]);
	//}
	//else{
	return data[size / 2];
	//}
}

static inline bool fun_comp(int i, int j)
{
	return abs(i) > abs(j);
}

// 8 pixels
void inline refine_pixel(cv::Mat& map, cv::Mat& interpolation, PixelType p_type, int x, int y){
	float* data = ((float*)interpolation.data) + point_to_nd_coord(x, y, map.cols, 0, 8);
	float* partition = std::partition(data, data + 8, static_cast<bool(*)(float)>(isnotfinite));
	int size = data + 8 - partition;
	std::sort(partition, data + 8, fun_comp);
	//if (x == 223 && y == 551){
	//	std::cout << std::endl;
	//	for (int i = 0; i < 8; i++){
	//		std::cout << data[i];
	//	}
	//	std::cout << "size:" << size << " " << partition[size - 1 - (size>4)];
	//}
	if (p_type == OCCL){
		((float*)map.data)[y*map.cols + x] = partition[size - 1 - (size > 2)];
	}
	else{
		((float*)map.data)[y*map.cols + x] = median(partition, size);
	}
}

// left right
void inline refine_pixel2(cv::Mat& map, cv::Mat& interpolation, PixelType p_type, int x, int y){
	float* data = ((float*)interpolation.data) + point_to_nd_coord(x, y, map.cols, 0, 8);
	/*if (size < 2){
		return;
		}
		std::sort(partition, data + 8);*/
	/*if (x == 223 && y == 551){
		std::cout << std::endl;
		for (int i = 0; i < 8; i++){
		std::cout << data[i];
		}
		std::cout << "size:" << size << " " << partition[size - 1 - (size>4)];
		}*/
	if (p_type == OCCL){
		if (std::isfinite(data[0]) && !std::isfinite(data[1])){
			((float*)map.data)[y*map.cols + x] = data[0];
		}
		else if (std::isfinite(data[1]) && !std::isfinite(data[0])){
			((float*)map.data)[y*map.cols + x] = data[1];
		}
		else {
			((float*)map.data)[y*map.cols + x] = std::abs(data[0]) > std::abs(data[1]) ? data[1] : data[0];
		}
	}
	else{
		float* partition = std::partition(data, data + 8, static_cast<bool(*)(float)>(isnotfinite));
		int size = data + 8 - partition;
		std::sort(partition, data + 8, fun_comp);
		((float*)map.data)[y*map.cols + x] = median(partition, size);
	}
}


// left right
void inline refine_pixel2_ver(cv::Mat& map, cv::Mat& interpolation, PixelType p_type, int x, int y){
	float* data = ((float*)interpolation.data) + point_to_nd_coord(x, y, map.cols, 0, 8);
	if (p_type == OCCL){
		if (std::isfinite(data[0]) && !std::isfinite(data[1])){
			((float*)map.data)[y*map.cols + x] = data[0];
		}
		else if (std::isfinite(data[1]) && !std::isfinite(data[0])){
			((float*)map.data)[y*map.cols + x] = data[1];
		}
		else if (std::isfinite(data[1]) && std::isfinite(data[0])){
			((float*)map.data)[y*map.cols + x] = std::abs(data[0]) > std::abs(data[1]) ? data[1] : data[0];
		}
		else{
			((float*)map.data)[y*map.cols + x] = NAN;
		}
	}
	else{
		float* partition = std::partition(data, data + 8, static_cast<bool(*)(float)>(isnotfinite));
		int size = data + 8 - partition;
		std::sort(partition, data + 8, fun_comp);
		((float*)map.data)[y*map.cols + x] = median(partition, size);
	}
}

// left right and one more
void inline refine_pixel3(cv::Mat& map, cv::Mat& interpolation, PixelType p_type, int x, int y){
	float* data = ((float*)interpolation.data) + point_to_nd_coord(x, y, map.cols, 0, 8);
	/*if (size < 2){
	return;
	}
	std::sort(partition, data + 8);*/
	/*if (x == 223 && y == 551){
	std::cout << std::endl;
	for (int i = 0; i < 8; i++){
	std::cout << data[i];
	}
	std::cout << "size:" << size << " " << partition[size - 1 - (size>4)];
	}*/
	if (p_type == OCCL){
		float min_from_l_and_r = std::numeric_limits<float>::quiet_NaN();
		if (std::isfinite(data[0]) && !std::isfinite(data[1])){
			min_from_l_and_r = data[0];
		}
		else if (std::isfinite(data[1]) && !std::isfinite(data[0])){
			min_from_l_and_r = data[1];
		}
		else {
			min_from_l_and_r = std::abs(data[0]) > std::abs(data[1]) ? data[1] : data[0];
		}
		float* partition = std::partition(data + 2, data + 8, static_cast<bool(*)(float)>(isnotfinite));
		int size = data + 8 - partition;
		std::sort(partition, data + 8, fun_comp);
		if (std::isfinite(partition[size - 2]) && (abs(partition[size - 2]) < abs(min_from_l_and_r) || !std::isfinite(min_from_l_and_r))){
			min_from_l_and_r = partition[size - 2];
		}

		((float*)map.data)[y*map.cols + x] = min_from_l_and_r;
	}
	else{
		float* partition = std::partition(data, data + 8, static_cast<bool(*)(float)>(isnotfinite));
		int size = data + 8 - partition;
		std::sort(partition, data + 8, fun_comp);
		((float*)map.data)[y*map.cols + x] = median(partition, size);
	}
}

inline bool neighbor_is_occl2_hor(const cv::Mat& map, int i, int j){
	uchar* data = map.data;
	if (/*i > 0 && data[(i - 1)*map.cols + j] == OCCL || i + 1<map.rows && data[(i + 1)*map.cols + j] == OCCL ||*/
		j > 0 && data[i*map.cols + j - 1] == OCCL || j + 1< map.cols && data[i*map.cols + j + 1] == OCCL /*||
		j > 0 && i > 0 && data[(i - 1)*map.cols + j - 1] == OCCL || j + 1 < map.cols && i + 1< map.rows && data[(i + 1)*map.cols + j + 1] == OCCL ||
		j > 0 && i + 1 < map.rows && data[(i + 1)*map.cols + j - 1] == OCCL || j + 1 < map.cols && i > 0 && data[(i - 1)*map.cols + j + 1] == OCCL*/){
		return true;
	}
	return false;
}

inline bool neighbor_is_occl2_ver(const cv::Mat& map, int i, int j){
	uchar* data = map.data;
	if (i > 0 && data[(i - 1)*map.cols + j] == OCCL || i + 1<map.rows && data[(i + 1)*map.cols + j] == OCCL){
		return true;
	}
	return false;
}

inline bool neighbor_is_occl8(const cv::Mat& map, int i, int j){
	uchar* data = map.data;
	if (i > 0 && data[(i - 1)*map.cols + j] == OCCL || i + 1<map.rows && data[(i + 1)*map.cols + j] == OCCL ||
		j > 0 && data[i*map.cols + j - 1] == OCCL || j + 1< map.cols && data[i*map.cols + j + 1] == OCCL ||
		j > 0 && i > 0 && data[(i - 1)*map.cols + j - 1] == OCCL || j + 1 < map.cols && i + 1< map.rows && data[(i + 1)*map.cols + j + 1] == OCCL ||
		j > 0 && i + 1 < map.rows && data[(i + 1)*map.cols + j - 1] == OCCL || j + 1 < map.cols && i > 0 && data[(i - 1)*map.cols + j + 1] == OCCL){
		return true;
	}
	return false;
}

//that's not correct method :(
void occl_mism_detect_point(cv::Mat& occl_mism_map, cv::Mat& map, const cv::Mat& refined_mask, const cv::Mat& right_map, const cv::Mat& detmask){
	for (int i = 0; i < refined_mask.rows; i++){
		for (int j = 0; j < refined_mask.cols; j++){
			if ((refined_mask.data)[i*refined_mask.cols + j] != 255 && (detmask.data)[i*detmask.cols + j] != MASK_OUT){
				PixelType p_type = OCCL;
				float disp1 = ((float*)map.data)[i*map.cols + j];
				int x2 = j + round(disp1);
				if (!std::isfinite(disp1) || !(x2 < right_map.cols && x2 >= 0)){
					(occl_mism_map.data)[i*refined_mask.cols + j] = OCCL;
					//refine_pixel0(map, interpolation, OCCL, j, i);
					continue;
				}
				float disp2 = ((float*)right_map.data)[i*right_map.cols + x2];
				float disp_dif = disp2 + disp1;
				int inters_x = round(x2 - disp_dif);
				if (!std::isfinite(disp_dif) || !(inters_x < right_map.cols && inters_x >= 0)){
					(occl_mism_map.data)[i*refined_mask.cols + j] = OCCL;
					//refine_pixel0(map, interpolation, OCCL, j, i);
					continue;
				}
				float inters_d = ((float*)right_map.data)[i*right_map.cols + inters_x];
				if (abs(inters_d - disp2) <= 1.){
					p_type = MISM;
					//mism++;
				}
				(occl_mism_map.data)[i*refined_mask.cols + j] = p_type;
				//refine_pixel0(map, interpolation, p_type, j, i);
			}
		}

	}
}

void occl_mism_detect_line(cv::Mat& occl_mism_map, cv::Mat& map, const cv::Mat& refined_mask, const cv::Mat& right_map, const cv::Mat& detmask){
	for (int i = 0; i < right_map.rows; i++){
		for (int j = 0; j < right_map.cols; j++){
			float disp2 = ((float*)right_map.data)[i*right_map.cols + j];
			int x = trunc(j + disp2);
			if (std::isfinite(disp2) && (x < map.cols && x >= 0) && (refined_mask.data)[i*refined_mask.cols + x] != 255 && (detmask.data)[i*detmask.cols + x] != MASK_OUT){
				(occl_mism_map.data)[i*refined_mask.cols + x] = MISM;
			}
			x += 1;
			if (std::isfinite(disp2) && (x < map.cols && x >= 0) && (refined_mask.data)[i*refined_mask.cols + x] != 255 && (detmask.data)[i*detmask.cols + x] != MASK_OUT){
				(occl_mism_map.data)[i*refined_mask.cols + x] = MISM;
			}
		}
	}

	for (int i = 0; i < refined_mask.rows; i++){
		for (int j = 0; j < refined_mask.cols; j++){
			if ((occl_mism_map.data)[i*occl_mism_map.cols + j] != MISM && (refined_mask.data)[i*refined_mask.cols + j] != 255 && (detmask.data)[i*detmask.cols + j] != MASK_OUT){
				(occl_mism_map.data)[i*refined_mask.cols + j] = OCCL;
			}
		}

	}
}

cv::Mat occl_mismatches_refine(cv::Mat& map, const cv::Mat& mask, const cv::Mat& right_map, const cv::Mat& detmask, InterpMainDir main_dir){
	cv::Mat refined_mask = mask;
	cv::Mat interpolation(refined_mask.size(), CV_MAKETYPE(CV_32FC1, 8), cv::Scalar(0));
	fill_interp_matrix(map, interpolation, refined_mask, detmask, main_dir);
	//int mism = 0;
	cv::Mat occl_mism_map(refined_mask.size(), CV_8UC1, cv::Scalar(255));

	if (OCCL_MISM_LINE_DETECTION()){
		occl_mism_detect_line(occl_mism_map, map, refined_mask, right_map, detmask);
	}
	else{
		occl_mism_detect_point(occl_mism_map, map, refined_mask, right_map, detmask);
	}

	int mism = 0;
	bool(*neighbor_is_occl_f)(const cv::Mat&, int, int);
	void(*refine_pixel_f)(cv::Mat& map, cv::Mat& interpolation, PixelType p_type, int x, int y);

	if (main_dir == BT){
		neighbor_is_occl_f = neighbor_is_occl2_ver;
		refine_pixel_f = refine_pixel2_ver;
	}
	else{
		neighbor_is_occl_f = neighbor_is_occl2_hor;
		refine_pixel_f = refine_pixel3;
	}
	for (int i = 0; i < refined_mask.rows; i++){
		for (int j = 0; j < refined_mask.cols; j++){
			if ((occl_mism_map.data)[i*refined_mask.cols + j] != 255 && (detmask.data)[i*detmask.cols + j] != MASK_OUT){
				if (neighbor_is_occl_f(occl_mism_map, i, j)){
					(occl_mism_map.data)[i*refined_mask.cols + j] = OCCL;
				}
				if ((occl_mism_map.data)[i*refined_mask.cols + j] == MISM){
					mism++;
				}
				refine_pixel_f(map, interpolation, (PixelType)((occl_mism_map.data)[i*refined_mask.cols + j]), j, i);
			}
		}
	}
	std::cout << "mism" << mism;
	return occl_mism_map;
}

void erode_dilate(const cv::Mat& src, cv::Mat& dst, int erosion_size, int dil_size){
	cv::Mat elem_dil = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * dil_size + 1, 2 * dil_size + 1), cv::Point(dil_size, dil_size));
	cv::Mat elem_erode = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1), cv::Point(erosion_size, erosion_size));
	cv::dilate(src, dst, elem_dil);
	cv::erode(dst, dst, elem_erode);
}
#endif