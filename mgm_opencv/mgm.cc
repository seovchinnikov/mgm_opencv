/* Copyright (C) 2015, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>,
 *                     Carlo de Franchis <carlo.de-franchis@ens-cachan.fr>,
 *                     Enric Meinhardt <enric.meinhardt@cmla.ens-cachan.fr>*/
#define _CRT_SECURE_NO_WARNINGS 1
#define DVEC_ALLOCATION_HACK 1

#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "math.h"
#include <numeric>
#include <algorithm>
#include <vector>
#include <cstring>
#include "assert.h"
#include <fstream>
#include <omp.h>
#include "helper.h"
#include "voting_occl.h"

#include "smartparameter.h"

//// a structure to wrap images 
#include "opencv2/opencv.hpp"
#include "point.h"

#include "img_tools.h"
#include "occl_tools.h"
#include "reproject_3d.h"
// // not used here but generally useful 
// typedef std::vector<float> FloatVector;


SMART_PARAMETER(TSGM_DEBUG, 0)


/********************** COSTVOLUME *****************************/

#include "mgm_costvolume.h"
struct costvolume_t allocate_and_fill_sgm_costvolume(const cv::Mat& in_u, // source (reference) image                      
	const cv::Mat& in_v, // destination (match) image                  
	const cv::Mat& dminI,// per pixel max&min disparity
	const cv::Mat& dmaxI,
	char* prefilter,        // none, sobel, census(WxW)
	char* distance,         // census, l1, l2, ncc(WxW), btl1, btl2
	float truncDist,		// truncated differences
	cv::Mat& tmp_filtered_u, cv::Mat& tmp_filtered_v,
	const cv::Mat& prev_out, bool prev_out_exists,
	cv::Mat& u2, cv::Mat& v2, const cv::Mat& maskdet, const cv::Mat& maskdet2,
	char *cnnweights, char **params, bool direct);

/********************** MGM *****************************/

#include "mgm_core.cc"
struct costvolume_t mgm(struct costvolume_t CC, const cv::Mat& in_w,
	const cv::Mat& dminI, const cv::Mat& dmaxI,
	cv::Mat& out, cv::Mat& outcost, const cv::Mat& maskdet,
	const float P1, const float P2, const int NDIR, const int MGM,
	const int USE_FELZENSZWALB_POTENTIALS, // USE SGM(0) or FELZENSZWALB(1) POTENTIALS
	int SGM_FIX_OVERCOUNT, const cv::Mat* occl_blayer, int LET_OCCL_PROPAGATE);                // fix the overcounting in SGM following (Drory etal. 2014)

#include "mgm_weights.h"
cv::Mat compute_mgm_weights(const cv::Mat& u, float aP, float aThresh);

/********** SOLUTION REFINEMENT AND ENERGY COMPUTATION ***********/

#include "mgm_refine.h"
void subpixel_refinement_sgm(struct costvolume_t &S,       // modifies out and outcost
	float* out,
	float* outcost,
	char *refinement, int N); //none, vfit, parabola, cubic, parabolaOCV

#include "mgm_print_energy.h"
void print_solution_energy(const cv::Mat& in_u, const float* disp, struct costvolume_t& CC, float P1, float P2);

/********************** OTHERSTUFF  *****************************/


cv::Mat leftright_test(cv::Mat& dx, const cv::Mat& Rdx, float threshold = 1, bool maskOnly = 0)
{
	int nc = dx.cols;
	int nr = dx.rows;
	int Rnc = Rdx.cols;
	int Rnr = Rdx.rows;
	cv::Mat occL = cv::Mat(nr, nc, CV_8UC1, cv::Scalar(255));

	for (int y = 0; y < nr; y++)
	for (int x = 0; x < nc; x++) {
		int i = x + y*nc;
		int Lx, Rx;
		Lx = round(x + ((float*)dx.data)[i]);
		if (std::isfinite(((float*)dx.data)[i]) && (Lx)<Rnc && (Lx) >= 0){
			int Lidx = Lx + y*Rnc;
			float Rx = Lx + ((float*)Rdx.data)[Lidx];
			if (!std::isfinite(((float*)Rdx.data)[Lidx]) || fabs(Rx - x) > threshold) {
				if (!maskOnly){
					((float*)dx.data)[i] = NAN;
				}
				(occL.data)[i] = 122;
			}
		}
		else {
			if (!maskOnly){
				((float*)dx.data)[i] = NAN;
			}
			(occL.data)[i] = 122;
		}
	}
	return occL;

}

cv::Mat leftright_test_bleyer(cv::Mat& dx, const cv::Mat& Rdx, bool maskOnly = 0)
// warps the pixels of the right image to the left, if no pixel in the
// left image receives a contribution then it is marked as occluded
{
	int nc = dx.cols;
	int nr = dx.rows;
	int Rnc = Rdx.cols;
	int Rnr = Rdx.rows;

	//struct Img occL(nc, nr);
	cv::Mat occL = cv::Mat(nr, nc, CV_8UC1, cv::Scalar(122));
	//for (int i = 0; i < nr*nc; i++) ((float*)occL.data)[i] = 0;

	for (int y = 0; y < Rnr; y++)
	for (int x = 0; x < Rnc; x++) {
		int i = x + y*Rnc;
		int Lx = round(x + ((float*)Rdx.data)[i]);
		if (std::isfinite(((float*)Rdx.data)[i]) && (Lx) < nc && (Lx) >= 0){
			(occL.data)[Lx + y*nc] = 255;
		}
	}
	if (!maskOnly){
		for (int i = 0; i < nr*nc; i++)
		if ((occL.data)[i] != 255) ((float*)dx.data)[i] = NAN;
	}
	return occL;
}

cv::Mat leftright_test_bleyer_test_only(const cv::Mat& Rdx, int nc, int nr)
// warps the pixels of the right image to the left, if no pixel in the
// left image receives a contribution then it is marked as occluded
{

	int Rnc = Rdx.cols;
	int Rnr = Rdx.rows;

	//struct Img occL(nc, nr);
	cv::Mat occL = cv::Mat(nr, nc, CV_8UC1, cv::Scalar(122));
	//for (int i = 0; i < nr*nc; i++) ((float*)occL.data)[i] = 0;

	for (int y = 0; y < Rnr; y++)
	for (int x = 0; x < Rnc; x++) {
		int i = x + y*Rnc;
		int Lx = round(x + ((float*)Rdx.data)[i]);
		if (std::isfinite(((float*)Rdx.data)[i]) && (Lx) < nc && (Lx) >= 0){
			(occL.data)[Lx + y*nc] = 255;
		}
	}
	return occL;
}

// seems like it's filter in da neighborhood
std::pair<float, float> update_dmin_dmax(const cv::Mat& outoff, cv::Mat& dminI, cv::Mat& dmaxI, int slack = 3, int radius = 2) {
	//struct Img dminI2(*dminI);
	//struct Img dmaxI2(*dmaxI);
	cv::Mat dminI2(dminI);
	cv::Mat dmaxI2(dmaxI);
	int nx = outoff.cols;
	int ny = outoff.rows;

	// global (finite) min and max
	std::pair<float, float>gminmax = image_minmax(outoff);
	float gmin = gminmax.first; float gmax = gminmax.second;

	if (slack < 0) slack = -slack;
	int r = radius;

	for (int j = 0; j < ny; j++)
	for (int i = 0; i < nx; i++)
	{
		float dmin = INFINITY; float dmax = -INFINITY;
		for (int dj = -r; dj <= r; dj++)
		for (int di = -r; di <= r; di++)
		{
			float v = valneumann(outoff, i + di, j + dj, 0);
			if (isfinite(v)) {
				dmin = fmin(dmin, v - slack);
				dmax = fmax(dmax, v + slack);
			}
			else {
				dmin = fmin(dmin, gmin - slack);
				dmax = fmax(dmax, gmax + slack);
			}
		}
		if (isfinite(dmin)) {
			((float*)dminI2.data)[i + j*nx] = dmin; ((float*)dmaxI2.data)[i + j*nx] = dmax;
		}

	}

	dminI = dminI2;
	dmaxI = dmaxI2;
	return std::pair<float, float>(gmin, gmax);
}


// c: pointer to original argc
// v: pointer to original argv
// o: option name after hyphen
// d: default value (if NULL, the option takes no argument)
char *pick_option(int *c, char ***v, char *o, char *d)
{
	int argc = *c;
	char **argv = *v;
	int id = d ? 1 : 0;
	for (int i = 0; i < argc - id; i++)
	if (argv[i][0] == '-' && 0 == strcmp(argv[i] + 1, o)) {
		char *r = argv[i + id] + 1 - id;
		*c -= id + 1;
		for (int j = i; j < argc - id; j++)
			(*v)[j] = (*v)[j + id + 1];
		return r;
	}
	return d;
}



/*MGM*/


SMART_PARAMETER(TSGM, 2);
SMART_PARAMETER(TSGM_FIX_OVERCOUNT, 1);
SMART_PARAMETER(TSGM_2LMIN, 0);
SMART_PARAMETER(USE_TRUNCATED_LINEAR_POTENTIALS, 0);

SMART_PARAMETER(WITH_MGM2, 0);

SMART_PARAMETER_INT(TSGM_ITER, 1)
SMART_PARAMETER_INT(TESTLRRL, 1)
SMART_PARAMETER_INT(MEDIAN, 0)
SMART_PARAMETER_INT(STATS, 0)
SMART_PARAMETER_INT(BOTH_IMAGES, 0)
SMART_PARAMETER_INT(DEP_WEIGHTS, 0)
SMART_PARAMETER_INT(LET_OCCL_PROP, 0)
SMART_PARAMETER_INT(MAIN_DIR, 1)
SMART_PARAMETER_INT(REPROJ_TO_3D, 0)
SMART_PARAMETER_INT(TRIANGULATE, 0)
SMART_PARAMETER_INT(VOTING_OCCL, 1)
int main(int argc, char* argv[])
{
	/* patameter parsing - parameters*/
	_putenv("MEDIAN=0");
	_putenv("CENSUS_NCC_WIN=5");
	_putenv("USE_TRUNCATED_LINEAR_POTENTIALS=0");
	_putenv("TSGM=3");
	_putenv("TESTLRRL=0");
	_putenv("TSGM_DEBUG=1");
	_putenv("TSGM_FIX_OVERCOUNT=1");
	_putenv("WITH_MGM2=0");
	_putenv("STATS=1");
	_putenv("BOTH_IMAGES=1");
	_putenv("TSGM_ITER=2");
	_putenv("COMMON_COST_UPPER_BOUND=255");
	_putenv("DEP_WEIGHTS=1");
	_putenv("LET_OCCL_PROP=0");
	_putenv("MAIN_DIR=1");
	_putenv("REPROJ_TO_3D=0");
	_putenv("TRIANGULATE=0");
	_putenv("OCCL_MISM_LINE_DETECTION=1");
	_putenv("VOTING_OCCL=0");

	if (argc < 4)
	{
		fprintf(stderr, "too few parameters\n");
		fprintf(stderr, "   usage: %s  [-r dmin -R dmax] [-m dminImg -M dmaxImg] [-O NDIR: 2, (4), 8, 16] u v out [gt mask] [cost]\n", argv[0]);
		fprintf(stderr, "        [-P1 (8) -P2 (32)]: sgm regularization parameters P1 and P2\n");
		fprintf(stderr, "        [-p PREFILT(none)]: prefilter = {none|census|sobelx|gblur} (census is WxW)\n");
		fprintf(stderr, "        [-t      DIST(ad)]: distance = {census|ad|sd|ncc|btad|btsd}  (ncc is WxW, bt is Birchfield&Tomasi)\n");
		fprintf(stderr, "        [-truncDist (inf)]: truncate distances at nch*truncDist  (default INFINITY)\n");
		fprintf(stderr, "        [-s  SUBPIX(none)]: subpixel refinement = {none|vfit|parabola|cubic}\n");
		fprintf(stderr, "        [-aP1         (1)]: multiplier factors of P1 and P2 when\n");
		fprintf(stderr, "        [-aP2         (1)]:    \\sum |I1 - I2|^2 < nch*aThresh^2\n");
		fprintf(stderr, "        [-aThresh     (5)]: Threshold for the multiplier factor (default 5)\n");
		fprintf(stderr, "        [-l   FILE (none)]:  disparity file with LR test (default none)\n");
		fprintf(stderr, "        [-epsilon     (5)]: epsilon for computation of the error rate\n");
		fprintf(stderr, "        [-occlusions  (0)]:apply occlusions fix\n");


		fprintf(stderr, "        [-dmaskl  (none)]: left image mask file with pixels we dont want to consider masked 0\n");
		fprintf(stderr, "        [-dmaskr  (none)]: right image mask file with pixels we dont want to consider masked 0\n");
		fprintf(stderr, "        [-folder  (none)]: folder with dataset\n");
		fprintf(stderr, "        [-calib  (none)]: calibration file for left camera for 3D rendering in opencv format\n");
		fprintf(stderr, "        [-out3d  (none)]: output 3D ply file for 3D rendering\n");
		fprintf(stderr, "        [-cnnweights  (../LeNet-weights)]: weights file for convolution neural network cost function\n");
		fprintf(stderr, "        [-statfile  (none)]: file for stat output\n");

		fprintf(stderr, "        ENV: CENSUS_NCC_WIN=3   : size of the window for census and NCC\n");
		fprintf(stderr, "        ENV: TESTLRRL=1   : lrrl\n");
		fprintf(stderr, "        ENV: MEDIAN=0     : radius of the median filter postprocess\n");
		fprintf(stderr, "        ENV: TSGM=4       : regularity level\n");
		fprintf(stderr, "        ENV: TSGM_ITER=1  : iterations\n");
		fprintf(stderr, "        ENV: TSGM_FIX_OVERCOUNT=1   : fix overcounting of the data term in the energy\n");
		fprintf(stderr, "        ENV: TSGM_DEBUG=0 : prints debug informtion\n");
		fprintf(stderr, "        ENV: TSGM_2LMIN=0 : use the improved TSGM cost only for TSGM=2. Overrides TSGM value\n");
		fprintf(stderr, "        ENV: USE_TRUNCATED_LINEAR_POTENTIALS=0 : use the Felzenszwalb-Huttenlocher\n");
		fprintf(stderr, "                          : truncated linear potential (when=1). P1 and P2 change meaning\n");
		fprintf(stderr, "                          : The potential they describe becomes:  V(p,q) = min(P2,  P1*|p-q|)\n");

		fprintf(stderr, "        ENV: STATS=0   : print statistics\n");
		fprintf(stderr, "        ENV: DEP_WEIGHTS=0   : dependent weights\n");
		fprintf(stderr, "        ENV: LET_OCCL_PROP=0   : if 0, occluded pixels wont propagate its wrong value to its neighbors\n");
		fprintf(stderr, "        ENV: MAIN_DIR=1   : main extrapolation direction : 1 - L to R, 0 - B to T\n");
		fprintf(stderr, "        ENV: REPROJ_TO_3D=0   : need to render 3D model\n");
		fprintf(stderr, "        ENV: TRIANGULATE=0   : make triangulation for 3D model\n");
		fprintf(stderr, "        ENV: COMMON_COST_UPPER_BOUND=CENSUS_NCC_WIN()*CENSUS_NCC_WIN()   :  upper bound for cost function's value\n");
		fprintf(stderr, "        ENV: OCCL_MISM_LINE_DETECTION=1   :  fix of checking pixel type between occl/mism\n");
		return 1;
	}


	//read the parameters

	int i = 1;
	char *in_min_disp_file = pick_option(&argc, &argv, (char*) "m", (char*) "");
	char *in_max_disp_file = pick_option(&argc, &argv, (char*) "M", (char*) "");
	int dmin = atoi(pick_option(&argc, &argv, (char*) "r", (char*) "-30"));
	int dmax = atoi(pick_option(&argc, &argv, (char*) "R", (char*) "30"));
	int NDIR = atoi(pick_option(&argc, &argv, (char*) "O", (char*) "4"));
	int occlusions_fix = atoi(pick_option(&argc, &argv, (char*) "occlusions", (char*) "0"));
	float P1 = atof(pick_option(&argc, &argv, (char*) "P1", (char*) "8"));
	float P2 = atof(pick_option(&argc, &argv, (char*) "P2", (char*) "32"));
	float aP1 = atof(pick_option(&argc, &argv, (char*) "aP1", (char*) "1"));
	float aP2 = atof(pick_option(&argc, &argv, (char*) "aP2", (char*) "1"));
	float aThresh = atof(pick_option(&argc, &argv, (char*) "aTresh", (char*) "50"));
	float epsilon = atof(pick_option(&argc, &argv, (char*) "epsilon", (char*) "1"));

	char* distance = pick_option(&argc, &argv, (char*) "t", (char*) "ad");   //{census|ad|sd|ncc|btad|btsd}
	char* prefilter = pick_option(&argc, &argv, (char*) "p", (char*) "none"); //{none|census|sobelx}
	char* refine = pick_option(&argc, &argv, (char*) "s", (char*) "none"); //{none|vfit|parabola|cubic}
	float truncDist = atoff(pick_option(&argc, &argv, (char*) "truncDist", (char*) "inf"));
	char *lr_disp_file = pick_option(&argc, &argv, (char*) "l", (char*) "");
	char *detect_maskl = pick_option(&argc, &argv, (char*) "dmaskl", (char*) "");
	char *detect_maskr = pick_option(&argc, &argv, (char*) "dmaskr", (char*) "");
	char *folder = pick_option(&argc, &argv, (char*) "folder", (char*) "test");
	char *calib = pick_option(&argc, &argv, (char*) "calib", (char*) "");
	char *out3d = pick_option(&argc, &argv, (char*) "out3d", (char*) "");
	char *cnnweights = pick_option(&argc, &argv, (char*) "cnnweights", (char*) "../LeNet-weights");
	char *statfile = pick_option(&argc, &argv, (char*) "statfile", (char*) "");

	char* f_u = (argc > i) ? argv[i] : NULL;      i++;
	char* f_v = (argc > i) ? argv[i] : NULL;      i++;
	char* f_out = (argc > i) ? argv[i] : NULL;      i++;
	char* gt = (argc > i) ? argv[i] : NULL;      i++;
	char* mask = (argc > i) ? argv[i] : NULL;      i++;
	char* f_cost = (argc > i) ? argv[i] : NULL;      i++;

	printf("%d %d\n", dmin, dmax);


	// read input
	cv::Mat u = iio_read_vector_split(f_u);
	cv::Mat v = iio_read_vector_split(f_v);

	remove_nonfinite_values_Img(u, 0);
	remove_nonfinite_values_Img(v, 0);

	cv::Mat dminI(u.rows, u.cols, CV_32FC1, cv::Scalar(dmin));
	cv::Mat dmaxI(u.rows, u.cols, CV_32FC1, cv::Scalar(dmax));


	//for (int i = 0; i < u.npix; i++) { dminI[i] = dmin; dmaxI[i] = dmax; }

	if (strcmp(in_min_disp_file, "") != 0){
		dminI = iio_read_vector_split(in_min_disp_file);
		dmaxI = iio_read_vector_split(in_max_disp_file);
		// sanity check for nans
		remove_nonfinite_values_Img(dminI, dmin);
		remove_nonfinite_values_Img(dmaxI, dmax);

		// more hacks to prevent produce due to bad inputs (min>=max)
		float* dmaxIdata = (float*)dmaxI.data;
		float* dminIdata = (float*)dminI.data;
		for (int i = 0; i < u.cols*u.rows; i++) {
			if (dmaxIdata[i] < dminIdata[i] + 1) dmaxIdata[i] = ceil(dminIdata[i] + 1);
		}
	}


	//P1 = P1*u.channels(); //8
	//P2 = P2*u.channels(); //32

	// call
	cv::Mat outoff(u.rows, u.cols, CV_32FC1);
	cv::Mat outcost(u.rows, u.cols, CV_32FC1);


	// variables for LR
	cv::Mat outoffR(v.rows, v.cols, CV_32FC1);
	cv::Mat outcostR(v.rows, v.cols, CV_32FC1);
	cv::Mat dminRI(v.rows, v.cols, CV_32FC1, cv::Scalar(-dmax));
	cv::Mat dmaxRI(v.rows, v.cols, CV_32FC1, cv::Scalar(-dmin));

	// vars for tests
	cv::Mat occlLR, occlLR_right;
	cv::Mat occl_blayer, occl_blayer_right;
	cv::Mat occl_fix_mask, occl_fix_mask_right;
	cv::Mat after_lr;
	cv::Mat occl_mism_map;

	bool iter = strcmp(distance, "mic") == 0 || strcmp(distance, "mi") == 0;
	bool mi = strcmp(distance, "mic") == 0 || strcmp(distance, "mi") == 0;

	cv::Mat tmp_filtered_u2, tmp_filtered_v2;
	cv::Mat u_w = compute_mgm_weights(u, aP2, aThresh, tmp_filtered_u2, DEP_WEIGHTS()); // missing aP1 !! TODO
	cv::Mat v_w = compute_mgm_weights(v, aP2, aThresh, tmp_filtered_v2, DEP_WEIGHTS());

	cv::Mat maskdet_l = cv::Mat::ones(u.size(), CV_8UC1), maskdet_r = cv::Mat::ones(v.size(), CV_8UC1);
	if (strcmp(detect_maskl, "") != 0){
		maskdet_l = cv::imread(detect_maskl, CV_LOAD_IMAGE_GRAYSCALE);
	}
	if (strcmp(detect_maskr, "") != 0){
		maskdet_r = cv::imread(detect_maskr, CV_LOAD_IMAGE_GRAYSCALE);
	}

	cv::Mat tmp_filtered_u, tmp_filtered_v;
	cv::Mat prev_out_left, prev_out_right;

	bool prev_out_left_exists = false, prev_out_right_exists = false;
	for (int i = 0; i < (iter ? TSGM_ITER() : 1); i++) {
		char* cur_distance = mi && i == 0 ? "census" : distance;
		if (TESTLRRL() || occlusions_fix || STATS() || BOTH_IMAGES()) {
			struct costvolume_t CC = allocate_and_fill_sgm_costvolume(v, u, dminRI, dmaxRI, prefilter, cur_distance, truncDist,
				tmp_filtered_v, tmp_filtered_u, prev_out_right, prev_out_right_exists, tmp_filtered_v2, tmp_filtered_u2, maskdet_r, maskdet_l, cnnweights, argv, false);
			cv::Mat* occl_param_right = (occlusions_fix && i>0) ? &occl_blayer_right : NULL;
			struct costvolume_t S = WITH_MGM2() ?
				mgm_naive_parallelism(CC, v_w, dminRI, dmaxRI, outoffR, outcostR, maskdet_r, P1, P2,
				NDIR, TSGM(), USE_TRUNCATED_LINEAR_POTENTIALS(), TSGM_FIX_OVERCOUNT(), occl_param_right, LET_OCCL_PROP()) :
				mgm(CC, v_w, dminRI, dmaxRI, outoffR, outcostR, maskdet_r, P1, P2,
				NDIR, TSGM(), USE_TRUNCATED_LINEAR_POTENTIALS(), TSGM_FIX_OVERCOUNT(), occl_param_right, LET_OCCL_PROP());
			// call subpixel refinement  (modifies out and outcost)
			subpixel_refinement_sgm(S, (float*)outoffR.data, (float*)outcostR.data, refine, outoffR.cols*outcostR.rows);
			//std::pair<float, float>gminmax = update_dmin_dmax(outoffR, dminRI, dmaxRI);
			//remove_nonfinite_values_Img(dminRI, gminmax.first);
			//remove_nonfinite_values_Img(dmaxRI, gminmax.second);

			if (MEDIAN()) outoffR = median_filter(outoffR, MEDIAN());

		}

		if (STATS() || occlusions_fix){
			occl_blayer = leftright_test_bleyer_test_only(outoffR, u.cols, u.rows);
			erode_dilate(occl_blayer, occl_blayer, 1, 1);
			cv::imwrite((std::string)"data2/" + folder + "/occl_blayer_left" + std::to_string(i) + ".png", occl_blayer);
		}

		{
			struct costvolume_t CC = allocate_and_fill_sgm_costvolume(u, v, dminI, dmaxI, prefilter, cur_distance, truncDist,
				tmp_filtered_u, tmp_filtered_v, prev_out_left, prev_out_left_exists, tmp_filtered_u2, tmp_filtered_v2, maskdet_l, maskdet_r,cnnweights, argv, true);
			cv::Mat* occl_param = occlusions_fix ? &occl_blayer : NULL;
			struct costvolume_t S = WITH_MGM2() ?
				mgm_naive_parallelism(CC, u_w, dminI, dmaxI, outoff, outcost, maskdet_l, P1, P2,
				NDIR, TSGM(), USE_TRUNCATED_LINEAR_POTENTIALS(), TSGM_FIX_OVERCOUNT(), occl_param, LET_OCCL_PROP()) :
				mgm(CC, u_w, dminI, dmaxI, outoff, outcost, maskdet_l, P1, P2,
				NDIR, TSGM(), USE_TRUNCATED_LINEAR_POTENTIALS(), TSGM_FIX_OVERCOUNT(), occl_param, LET_OCCL_PROP());
			// call subpixel refinement  (modifies out and outcost)
			subpixel_refinement_sgm(S, (float*)outoff.data, (float*)outcost.data, refine, outoff.cols*outoff.rows);
			// for the next iteration i think
			//std::pair<float, float>gminmax = update_dmin_dmax(outoff, dminI, dmaxI);
			//remove_nonfinite_values_Img(dminI, gminmax.first);
			//remove_nonfinite_values_Img(dmaxI, gminmax.second);


			if (MEDIAN()) outoff = median_filter(outoff, MEDIAN());

		}

		/**
			LEFT IMAGE
		*/

		if (TESTLRRL() || STATS() || occlusions_fix) {
			// original images stay untouched
			cv::Mat tmpL = outoff.clone();
			occlLR = leftright_test(tmpL, outoffR, epsilon, !TESTLRRL());  // L-R
			cv::imwrite((std::string)"data2/" + folder + "/lr_left" + std::to_string(i) + ".png", occlLR);
			after_lr = tmpL;
		}

		iio_write_norm_vector_split((std::string)"data2/" + folder + "/beforefix_left" + std::to_string(i) + ".png", outoff);
		if (occlusions_fix){
			if (!LET_OCCL_PROP()){
				occl_fix_mask = conj_masks(occlLR, refine_mask(occl_blayer));
			}
			else{
				occl_fix_mask = occlLR.clone();
			}
			fill_small_holes(occl_fix_mask);
			cv::imwrite((std::string)"data2/" + folder + "/occl_fix_mask_left" + std::to_string(i) + ".png", occl_fix_mask);
			if (VOTING_OCCL()){
				voting_occl_refine(outoff, occl_fix_mask, u, maskdet_l);
			} else {
				occl_mism_map = occl_mismatches_refine(outoff, occl_fix_mask, outoffR, maskdet_l, (InterpMainDir)MAIN_DIR());
			}
		}
		prev_out_left = outoff;
		prev_out_left_exists = true;

		/**
			RIGHT IMAGE

		*/

		// right image blayer
		if (BOTH_IMAGES() && (STATS() || occlusions_fix)){
			occl_blayer_right = leftright_test_bleyer_test_only(outoff, v.cols, v.rows);
			erode_dilate(occl_blayer_right, occl_blayer_right, 1, 1);
			cv::imwrite((std::string)"data2/" + folder + "/occl_blayer_right" + std::to_string(i) + ".png", occl_blayer_right);
		}
		// right image LR
		if (BOTH_IMAGES() && (TESTLRRL() || STATS() || occlusions_fix)) {
			// original images stay untouched
			cv::Mat tmpR = outoffR.clone();
			occlLR_right = leftright_test(tmpR, outoff, epsilon, !TESTLRRL()); // R-L
			cv::imwrite((std::string)"data2/" + folder + "/lr_right" + std::to_string(i) + ".png", occlLR_right);
		}
		// right image occl refinement
		if (BOTH_IMAGES() && occlusions_fix){
			iio_write_norm_vector_split((std::string)"data2/" + folder + "/beforefix_right" + std::to_string(i) + ".png", outoffR);
			if (!LET_OCCL_PROP()){
				occl_fix_mask_right = conj_masks(occlLR_right, refine_mask(occl_blayer_right));
			} else{
				occl_fix_mask_right = occlLR_right.clone();
			}
			fill_small_holes(occl_fix_mask_right);
			cv::imwrite((std::string)"data2/" + folder + "/occl_fix_mask_right" + std::to_string(i) + ".png", occl_fix_mask_right);
			if (VOTING_OCCL()){
				voting_occl_refine(outoffR, occl_fix_mask_right, v, maskdet_r);
			}
			else{
				occl_mismatches_refine(outoffR, occl_fix_mask_right, outoff, maskdet_r, (InterpMainDir)MAIN_DIR());
			}
		}
		prev_out_right = outoffR;
		prev_out_right_exists = true;
	}
	// save the disparity with LR
	if (TESTLRRL() && !strcmp(lr_disp_file, "") == 0)
		iio_write_vector_split(lr_disp_file, after_lr);


	//cv::FileStorage file5("data2/faces/disp.yml", cv::FileStorage::WRITE);
	//file5 << "Vocabulary" << outoff;
	//std::ofstream outs("data2/faces/disp.mat");
	//outs << format(outoff, "MATLAB");
	iio_write_norm_vector_split(f_out, outoff);
	iio_write_pfm(std::string(f_out) + ".pfm", outoff);

	if (TESTLRRL() || occlusions_fix || STATS() || BOTH_IMAGES()){
		iio_write_norm_vector_split((std::string)"data2/" + folder + "/disp_right.png", outoffR);
	}
	if (f_cost) iio_write_vector_split(f_cost, outcost);

	if (REPROJ_TO_3D() && strcmp(calib, "") != 0 && strcmp(out3d, "") !=0){
		cv::FileStorage calib_mats(calib, cv::FileStorage::READ);

		cv::Mat calib1, calib2;
		calib_mats["mat1"] >> calib1;
		calib_mats["mat2"] >> calib2;
		reproject_to_3d(u, v, outoff, outoffR, calib1, calib2, out3d, TRIANGULATE());
	}

	/****
	STATS STATS STATS
	*/

	// save the disparity
	cv::Mat out = outoff.clone();
	cv::Mat mask_im;
	cv::Mat gt_im;
	mask_im = mask ? cv::imread(mask, CV_LOAD_IMAGE_GRAYSCALE) : cv::Mat(u.rows, u.cols, CV_8UC1, cv::Scalar(255));
	//occl_blayer = refine_mask(occl_blayer);
	if (STATS() && gt && mask){
		int gt_w = 0, gt_h = 0;
		float* gt_vec = read_pfm_file(gt, &gt_w, &gt_h);
		gt_im = cv::Mat(gt_h, gt_w, CV_32FC1, gt_vec);
		cv::flip(gt_im, gt_im, 0);
		print_error_rate(out, gt_im, mask_im, epsilon, 1, statfile);
	}

	if (STATS() && gt){
		print_error_rate(out, gt_im, occlLR, epsilon, 1);
	}
	if (STATS() && gt){
		print_error_rate(out, gt_im, occl_blayer, epsilon, 1);
	}
	if (STATS() && gt && occlusions_fix){
		std::cout << std::endl << "error rate as in occl_fix mask";
		print_error_rate(out, gt_im, occl_fix_mask, epsilon, 1);
	}
	cv::Mat conjug_masks;
	if (mask && STATS()){
		conjug_masks = conj_masks(occlLR, occl_blayer);

		std::cout << std::endl << "LR vs gt";
		compare_occl_lr_gt(occlLR, mask_im);
		std::cout << std::endl << "blayer vs gt";
		compare_occl_lr_gt(occl_blayer, mask_im);
		std::cout << std::endl << "LR vs blayer";
		compare_occl_lr_gt(occlLR, occl_blayer);
		std::cout << std::endl << "conj vs gt";
		compare_occl_lr_gt(occl_fix_mask, mask_im);
	}

	if (mask && STATS() && gt){
		std::cout << std::endl << "mismatches not in masks' common regions: lr";
		compare_occl_lr_gt_mismatches_rate(occlLR, mask_im, out, gt_im, epsilon, 1);
		std::cout << std::endl << "mismatches not in masks' common regions: blayer";
		compare_occl_lr_gt_mismatches_rate(occl_blayer, mask_im, out, gt_im, epsilon, 1);
	}

	if (STATS() && gt && mask){
		std::cout << std::endl << "gt mask avg error:";
		print_aver_error(out, gt_im, mask_im, std::abs(dmin) + std::abs(dmax), 1., statfile);
	}

	if (STATS() && mask && !occl_mism_map.empty()){
		std::cout << std::endl << "wrong mismatch class appointed:";
		mismatches_class_wrong_rate(occl_mism_map, mask_im);
	}
	/*if (STATS() && gt){
		std::cout << std::endl << "occlLR mask avg error:";
		print_aver_error(out, gt_im, occlLR, std::abs(dmin) + std::abs(dmax), 1.);
	}*/
	if (!gt_im.empty()){
		delete[] (float*)gt_im.data;
	}
	if (strcmp(statfile, "")!=0){
		std::ofstream outfile;
		outfile.open(statfile, std::ios_base::app);
		outfile << std::endl;
	}

	return 0;
}
