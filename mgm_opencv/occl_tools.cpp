#include <vector>
#include "opencv2/opencv.hpp"
#include "img_tools.h"
#include "point.h"
#include "occl_tools.h"

inline static int point_to_coord(Point p, int cols){
	return ((int)p.y)*cols + (int)p.x;
}

inline static int point_to_nd_coord(Point p, int cols, int ch, int nd){
	return nd*((int)p.y)*cols + nd * (int)p.x + ch;
}

void occl_mismatches_refine(cv::Mat& map, const cv::Mat& mask){
	cv::Mat refined_mask = refine_mask(mask);
	cv::Mat interpolation(mask.size(), CV_MAKETYPE(CV_32FC1, 8), cv::Scalar(0));
	std::vector<Point> dirs{ Point(0, 1), Point(0, -1), Point(1, 0), Point(-1, 0), Point(1, 1), Point(-1, 1), Point(1, -1), Point(-1, -1) };

	for (int dir_counter = 0; dir_counter < dirs.size(); dir_counter++){
		const Point& dir = dirs[dir_counter];
		for (int i = 0; i < mask.total(); i++){
			float propVal = INFINITY;
			if ((mask.data)[i] != 255){
				Point first(i % mask.cols, i / mask.cols);
				Point next = first + dir;
				while (check_inside_image(next, mask)){
					if ((mask.data)[point_to_coord(next, mask.cols)] == 255){
						break;
					}
					next = next + dir;
				}
				if (check_inside_image(next, mask)){
					propVal = ((float*)map.data)[point_to_coord(next, mask.cols)];
				}
				Point opp_dir = Point(0, 0) - dir;
				next += opp_dir;
				Point end = first + opp_dir;
				while (next != end){
					if (!std::isfinite(propVal)){
						((float*)interpolation.data)[point_to_nd_coord(next, mask.cols, dir_counter, 8)] = ((float*)map.data)[point_to_coord(next, mask.cols)];
					}
					else{
						((float*)interpolation.data)[point_to_nd_coord(next, mask.cols, dir_counter, 8)] = propVal; 
					}
				}
			}
		}
	}
}