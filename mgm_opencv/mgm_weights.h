/* Copyright (C) 2015, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>,
 *                     Carlo de Franchis <carlo.de-franchis@ens-cachan.fr>,
 *                     Enric Meinhardt <enric.meinhardt@cmla.ens-cachan.fr>*/

#include "opencv2/opencv.hpp"
#include "helper.h"


#define __max(a,b)  (((a) > (b)) ? (a) : (b))
#define __min(a,b)  (((a) < (b)) ? (a) : (b))
// fast alternatives to: __min(a,__min(b,c)) 
// fastestest ?
#define fmin3_(x, y, z) \
	(((x) < (y)) ? (((z) < (x)) ? (z) : (x)) : (((z) < (y)) ? (z) : (y)))
// fast and easy to understand
//static inline float fmin3(float a, float b, float c)
//{
//   float m = a;
//   if (m > b) m = b;
//   if (m > c) m = c;
//   return m;
//}

inline float fastexp(float x) {
	int result = static_cast<int>(12102203 * x) + 1065353216;
	result *= result > 0;
	std::memcpy(&x, &result, sizeof(result));
	return x;
}

inline float deltaImage(const cv::Mat& u, const Point p, const Point q)
{
	float d = 0;
	for (int c = 0; c < u.channels(); c++){
		float diff = val(u, p, c) - val(u, q, c);
		//      d += diff > 0 ? diff : -diff;
		d += diff * diff;
		//      d = __max(d, fabs(val(u, p, c) - val(u, q, c))) ;
	}
	return d / u.channels();
}

inline float deltaImageGray(const cv::Mat& u, const Point p, const Point q)
{
	float diff = std::abs((u.data)[(int)p.x + u.cols*(int)p.y] - (u.data)[(int)q.x + u.cols*(int)q.y]);
	//      d += diff > 0 ? diff : -diff;
	//      d = __max(d, fabs(val(u, p, c) - val(u, q, c))) ;
	return diff;
}

inline float ws2(float DeltaI, float aP3, float Thresh) {
	return 1. / (1 + DeltaI / Thresh);
}

inline float ws(float DeltaI, float aP3, float Thresh) {

	if (fabs(DeltaI) < Thresh*Thresh) return aP3;
	else return 1;

	//// implements weights from "simple but effective tree structure for dynamic programming-based stereo matching" Blayer, Gelautz
	//   float T=30; 
	//   float P3=4;
	//   if (fabs(DeltaI) < T) return P3;
	//   else return 1;
	//
	//// implements image adaptive weights from formula 1 of "Efficient High-Resolution Stereo Matching using Local Plane Sweeps"
	//   float sigmaI=128;
	//   float alpha=10;
	//   return (8 + alpha * fastexp( - fabs(DeltaI) / sigmaI))/32.0;
	//   return 8.0/32.0 + 1.0/__max(fabs(DeltaI),0.0001);
	//
	//   return 1.;
}




// For a pixel p each image of the stack correspond the weights to 
// int neighbouring pixels: W, E, S, N, (NW, NE, SE, SW)
cv::Mat compute_mgm_weights(const cv::Mat& u, float aP, float aThresh, cv::Mat& tmp_filtered_u2, int dep_w = 0)
{
	int nx = u.cols;
	int ny = u.rows;

	// load the edge weights (or initialize them to 1: TODO)
	//struct Img w(nx, ny, 8);
	float(*wss)(float, float, float);
	if (dep_w){
		wss = ws2;
	}
	else{
		cv::Mat mat(ny, nx, typeConst(8));
		for (int i = 0; i < nx*ny * 8; i++){
			((float*)mat.data)[i] = 1;
		}
		return mat;
	}
	const int CHANNELS = 8;
	cv::Mat w(ny, nx, typeConst(CHANNELS));

	if (tmp_filtered_u2.empty()){
		u.convertTo(tmp_filtered_u2, CV_8UC3);
		cv::cvtColor(tmp_filtered_u2, tmp_filtered_u2, CV_BGR2GRAY);
	}

	Point scans[] = { Point(-1, 0), Point(1, 0), Point(0, 1), Point(0, -1), Point(-1, -1), Point(1, -1), Point(1, 1), Point(-1, 1) };
	float* wdata = (float*)w.data;
	for (int j = 0; j < ny; j++)
	for (int i = 0; i < nx; i++)
	for (int o = 0; o < 8; o++)
	{
		float wvalue = 1.0;
		Point p(i, j);				   // current point
		Point pr = p + scans[o];   // neighbor
		if (check_inside_image(pr, tmp_filtered_u2))  {
			float Delta = deltaImageGray(tmp_filtered_u2, p, pr);
			wvalue = wss(Delta, aP, aThresh);
		}
		//if (wvalue!=1){
		//	std::cout << "d";
		//}
		wdata[i*CHANNELS + tmp_filtered_u2.cols*j*CHANNELS + o] = wvalue;
		//x*u.channels() + u.cols*y*u.channels() + ch
	}
	return w;
}
