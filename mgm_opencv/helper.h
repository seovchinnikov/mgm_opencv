#include "opencv2/opencv.hpp"
#include <limits>

#ifndef HELPER_H_
#define HELPER_H_

unsigned int typeConst(unsigned int channels){
	return CV_MAT_DEPTH(CV_32F) + ((channels - 1) << CV_CN_SHIFT);
}

float atoff(const char* str){
	if (_strcmpi(str, "inf") == 0){
		return std::numeric_limits<float>::infinity();
	}
	else{
		return atof(str);
	}
}

#define MASK_OUT  0

#endif // HELPER_H_