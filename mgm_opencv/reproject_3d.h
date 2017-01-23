#ifndef REPROJECT_3D_
#define REPROJECT_3D_

#include "opencv2\opencv.hpp"
#include <math.h>
#include <iostream>
#include "img_tools.h"

static const char SEP = '\t';

#define MAX_DIST (10.)

struct d3Point{
	int x;
	int y;
	int disp;
	int z;
	int im_source;
	int index;
	int it_index;

	d3Point(int x, int y, int z, int im_source, int index, int it_index){
		this->x = x;
		this->y = y;
		this->z = z;
		this->im_source = im_source;
		this->index = index;
		this->it_index = it_index;
	}

};

struct Triangle{
	int i;
	int j;
	int k;
};

void project_to_new_plane(cv::Mat& new_left_proj, const cv::Mat& src_d, int new_img_offset, bool inverse = false){
	for (int y = 0; y < src_d.rows; y++){
		for (int x = 0; x < src_d.cols; x++){
			if (!std::isfinite(((float*)src_d.data)[y*src_d.cols + x])){
				continue;
			}
			int right_offs = inverse ? round(((float*)src_d.data)[y*src_d.cols + x]) : 0;
			((float*)new_left_proj.data)[y*new_left_proj.cols + x + new_img_offset + right_offs] = ((float*)src_d.data)[y*src_d.cols + x];
		}
	}
}

int project_to_new_plane_points(const cv::Mat& src_d, std::vector<d3Point>& out, int new_img_offset, int new_img_width, float& min, float& max, int& it_index, bool inverse = false){
	int pts_cntr = 0;
	for (int y = 0; y < src_d.rows; y++){
		for (int x = 0; x < src_d.cols; x++){
			int right_offs = inverse ? round(((float*)src_d.data)[y*src_d.cols + x]) : 0;
			float disp = ((float*)src_d.data)[y*src_d.cols + x];
			if (inverse){
				disp *= -1;
			}
			d3Point pnt(x + new_img_offset + right_offs, y, disp, inverse, y*src_d.cols + x, it_index);
			if (!std::isfinite(((float*)src_d.data)[y*src_d.cols + x])){
				pnt.it_index = -1;
			}
			else{
				if (disp<min){
					min = disp;
				}
				if (disp>max){
					max = disp;
				}
				it_index++;
				pts_cntr++;
			}
			out.push_back(pnt);
		}
	}
	return pts_cntr;
}

std::pair<int, int> compute_common_plane(const cv::Mat& left_d, const cv::Mat& right_d){
	int left = 0;
	int right = left_d.cols;

	for (int y = 0; y < right_d.rows; y++){
		for (int x = 0; x < right_d.cols; x++){
			if (!std::isfinite(((float*)right_d.data)[y*right_d.cols + x]))
				continue;
			int x2 = x + round(((float*)right_d.data)[y*right_d.cols + x]);
			if (x2<left){
				left = x2;
			}
			if (x2>right){
				right = x2;
			}
		}
	}
	std::pair<int, int> res(left, right);
	std::cout << std::endl << left << " " << right << std::endl;
	return res;
}

void calib_mats_norm(cv::Mat& calib1_l, cv::Mat& calib2_r, int w, int h, float max, float min){
	calib1_l /= calib1_l.at<float>(2, 2);
	calib2_r /= calib2_r.at<float>(2, 2);
	calib1_l = calib1_l.inv();
	/// Check consistency
	cv::Mat v = (calib2_r*calib1_l).row(0);
	float min2 = +FLT_MAX;
	float max2 = -FLT_MAX;

	float x = 0, y = 0, z;
	//z = x - (v(0)*x + v(1)*y + v(2));
	z = x - (v.at<float>(0, 0)*x + v.at<float>(0, 1)*y + v.at<float>(0, 2));
	if (z<min2) min2 = z; if (z>max2) max2 = z;

	x = (float)w, y = 0;
	z = x - (v.at<float>(0, 0)*x + v.at<float>(0, 1)*y + v.at<float>(0, 2));
	if (z<min2) min2 = z; if (z>max2) max2 = z;

	x = 0, y = (float)h;
	z = x - (v.at<float>(0, 0)*x + v.at<float>(0, 1)*y + v.at<float>(0, 2));
	if (z<min2) min2 = z; if (z>max2) max2 = z;

	x = (float)w, y = (float)h;
	z = x - (v.at<float>(0, 0)*x + v.at<float>(0, 1)*y + v.at<float>(0, 2));
	if (z<min2) min2 = z; if (z>max2) max2 = z;

	if (max + max2 < 0) {
		std::cout << "change signes" << std::endl;
		calib1_l = -calib1_l;
		calib2_r = -calib2_r;
	}
	else if (min + min2 > 0) {
		std::cout << "nothing to do" << std::endl;
	}
	else {
		std::cout << "Warning: problem with calibration matrices" << std::endl;
	}
}

/// Write header of PLY file
static void write_ply_header(std::ostream& out, size_t npts, int triang) {
	out << "ply" << std::endl;
	out << "format ascii 1.0" << std::endl;
	out << "comment mesh" << std::endl;
	out << "element vertex " << npts << std::endl;
	out << "property float x" << std::endl;
	out << "property float y" << std::endl;
	out << "property float z" << std::endl;
	out << "property uchar red" << std::endl;
	out << "property uchar green" << std::endl;
	out << "property uchar blue" << std::endl;
	if (triang){
		out << "element face " << triang << std::endl;
		out << "property list uchar int vertex_index" << std::endl;
	}
	out << "end_header" << std::endl;
}

/// Write body of PLY file
static void write_ply_body(const cv::Mat& calib1_l, const cv::Mat& calib2_r, const cv::Mat& left_src, const cv::Mat& right_src,
	const std::vector<d3Point>& pts, std::ostream& out, int w, int h, float max, float min, std::vector<Triangle>& triangles) {
	std::vector<d3Point>::const_iterator it = pts.begin();
	for (; it != pts.end(); ++it){
		if (it->it_index == -1){
			continue;
		}
		float x, y, z;
		cv::Mat v(3, 1, CV_32FC1);
		v.at<float>(0, 0) = it->x; v.at<float>(1, 0) = it->y; v.at<float>(2, 0) = 1.0f;
		v = calib1_l*v;
		z = calib2_r.at<float>(0, 0) / (it->x + it->z - ((cv::Mat)(calib2_r*v)).at<float>(0, 0));
		x = z*v.at<float>(0, 0);
		y = z*v.at<float>(1, 0);
		size_t i = it->index;
		out << x << SEP << y << SEP << z << SEP;
		float* color;
		if (it->im_source == 0){
			color = (float*)left_src.data;
		}
		else{
			color = (float*)right_src.data;
		}
		if (left_src.channels() == 3){
			out << (int)color[i*left_src.channels() + 2] << SEP << (int)color[i*left_src.channels() + 1] << SEP << (int)color[i*left_src.channels()] << std::endl;
		}
		else{
			out << (int)color[i] << std::endl;
		}

	}

	std::vector<Triangle>::const_iterator itt = triangles.begin();
	for (; itt != triangles.end(); ++itt){
		out << 3 << SEP << itt->i << SEP << itt->j << SEP << itt->k << std::endl;
	}
}


static void triangulate_f(std::vector<d3Point>::const_iterator p_begin, std::vector<Triangle>& triangles, int nx, int ny) {
	for (size_t y = 0; y + 1 < ny; y++) {
		std::vector<d3Point>::const_iterator it = p_begin + y*nx;
		std::vector<d3Point>::const_iterator it2 = it + nx;
		float max_d = MAX_DIST;
		for (size_t x = 0; x + 1 < nx; ++x, ++it, ++it2) {
			int fir = 1, sec = 1, thi = 1, fou = 1;
			if (it->it_index == -1){
				fou = fir = 0;
			}
			if ((it + 1)->it_index == -1){
				fir = sec = 0;
			}
			if ((it2 + 1)->it_index == -1){
				sec = thi = 0;
			}
			if (it2->it_index == -1){
				thi = fou = 0;
			}
			if (fir && std::abs(it->z - (it + 1)->z) > max_d){
				fir = 0;
			}
			if (sec && std::abs((it + 1)->z - (it2 + 1)->z) > max_d){
				sec = 0;
			}
			if (thi && std::abs(it2 ->z - (it2 + 1)->z) > max_d){
				thi = 0;
			}
			if (fou && std::abs(it->z - it2->z) > max_d){
				fou = 0;
			}
			// 2 triangles
			if (fir && sec && thi && fou){
				Triangle tr;
				if (std::abs(it->z - (it2 + 1)->z) < std::abs(it2->z - (it + 1)->z)){
					// 1,2
					tr.i = it->it_index;
					tr.j = (it + 1)->it_index;
					tr.k = (it2 + 1)->it_index;
					triangles.push_back(tr);
					// 3,4
					tr.i = (it2 + 1)->it_index;
					tr.j = it2->it_index;
					tr.k = it->it_index;
					triangles.push_back(tr);

				}
				else{
					// 1,4
					tr.i = it2->it_index;
					tr.j = it->it_index;
					tr.k = (it + 1)->it_index;
					triangles.push_back(tr);
					// 2,3
					tr.i = (it + 1)->it_index;
					tr.j = (it2 + 1)->it_index;
					tr.k = it2->it_index;
					triangles.push_back(tr);
				}
			}
			else if (fir + sec + thi + fou >= 2 ){
				// 1 triangle or 0
				Triangle tr;
				if (fir && sec){
					tr.i = it->it_index;
					tr.j = (it + 1)->it_index;
					tr.k = (it2+1)->it_index;
					triangles.push_back(tr);
				} else if (sec && thi){
					tr.i = (it+1)->it_index;
					tr.j = (it2 + 1)->it_index;
					tr.k = it2->it_index;
					triangles.push_back(tr);
				}
				else if (thi && fou){
					tr.i = (it2 + 1)->it_index;
					tr.j = it2->it_index;
					tr.k = it->it_index;
					triangles.push_back(tr);
				}
				else if (fou && fir){
					tr.i = it2->it_index;
					tr.j = it->it_index;
					tr.k = (it + 1)->it_index;
					triangles.push_back(tr);
				}

			}
			else{
				// no triangles
			}
		}
	}
}

void reproject_to_3d(const cv::Mat& left_src, const cv::Mat& right_src, const cv::Mat& left_d, const cv::Mat& right_d, const cv::Mat& calib1, const cv::Mat& calib2, const std::string& out3d, bool triangulate){

	cv::Mat& calib1_l = calib1.clone();
	cv::Mat& calib2_l = calib2.clone();


	std::pair<int, int> new_size = compute_common_plane(left_d, right_d);
	int new_w = new_size.second - new_size.first;
	int max_left_offs = new_size.first;
	int max_right_offs = new_size.second - left_d.cols;
	std::cout << std::endl << new_w << " " << max_left_offs << " " << max_right_offs << std::endl;

	std::vector<d3Point> points;


	//cv::Mat new_left_proj(left_d.rows, new_w, CV_32FC1, cv::Scalar(NAN));
	//project_to_new_plane(new_left_proj, left_d, max_left_offs, false);
	//iio_write_norm_vector_split((std::string)"data2/faces/new_left_proj.png", new_left_proj);
	float max = FLT_MIN;
	float min = FLT_MAX;
	int it_index = 0;
	int pts_cntr1 = project_to_new_plane_points(left_d, points, max_left_offs, new_w, max, min, it_index, false);
	std::vector<Triangle> triangles;
	if (triangulate){
		triangulate_f(points.begin(), triangles, new_w, left_d.rows);
	}

	//cv::Mat new_right_proj(right_d.rows, new_w, CV_32FC1, cv::Scalar(NAN));
	//project_to_new_plane(new_right_proj, right_d, max_left_offs, true);
	//iio_write_norm_vector_split((std::string)"data2/faces/new_right_proj.png", new_right_proj);
	int temp_size = points.size();
	int pts_cntr2 = project_to_new_plane_points(right_d, points, max_left_offs, new_w, max, min, it_index, true);
	if (triangulate){
		triangulate_f(points.begin() + temp_size, triangles, new_w, left_d.rows);
	}

	// Initialize the ply file by writing the ply header
	std::ofstream out(out3d);
	write_ply_header(out, pts_cntr1 + pts_cntr2, triangles.size());
	calib_mats_norm(calib1_l, calib2_l, new_w, left_d.rows, max, min);
	write_ply_body(calib1_l, calib2_l, left_src, right_src, points, out, new_w, left_d.rows, max, min, triangles);
	out.close();
}


#endif /* REPROJECT_3D_ */