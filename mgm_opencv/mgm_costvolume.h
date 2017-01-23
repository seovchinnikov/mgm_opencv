/* Copyright (C) 2015, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>,
 *                     Carlo de Franchis <carlo.de-franchis@ens-cachan.fr>,
 *                     Enric Meinhardt <enric.meinhardt@cmla.ens-cachan.fr>*/


#ifndef COSTVOLUME_H_
#define COSTVOLUME_H_

#include "point.h"
#include "img_tools.h"
#include "census_tools.cc"
#include "smartparameter.h"
//include only once in main file
#include "opencv2/opencv.hpp"
#include "mutual_information.h"
#include "helper.h"
#include "tiny_cnn.h"
#include "stereo_im_parser.h"

#define __max(a,b)  (((a) > (b)) ? (a) : (b))
#define __min(a,b)  (((a) < (b)) ? (a) : (b))

//double TSGM_DEBUG(void);

char *pick_option(int *c, char ***v, char *o, char *d);

// the type of a cost function
typedef float(*cost_t)(Point, Point, const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&, const std::vector<tiny_cnn::vec_t>&, const std::vector<tiny_cnn::vec_t>&);

inline float computeC_AD(Point p, Point q, const cv::Mat& u, const cv::Mat& v, const cv::Mat& u2, const cv::Mat& v2, const cv::Mat& cost_mat, 
	const std::vector<tiny_cnn::vec_t>& u_cnn, const std::vector<tiny_cnn::vec_t>& v_cnn) {
	if (!check_inside_image(p, u)) return INFINITY;
	if (!check_inside_image(q, v)) return INFINITY;
	float tmp = 0;
	for (int t = 0; t < u.channels(); t++) {
		float x = val(u, p, t) - val(v, q, t);
		x = __max(x, -x);
		tmp += x;
	}
	return tmp;
}
inline float computeC_SD(Point p, Point q, const cv::Mat& u, const cv::Mat& v, const cv::Mat& u2, const cv::Mat& v2, const cv::Mat& cost_mat,
	const std::vector<tiny_cnn::vec_t>& u_cnn, const std::vector<tiny_cnn::vec_t>& v_cnn) {
	if (!check_inside_image(p, u)) return INFINITY;
	if (!check_inside_image(q, v)) return INFINITY;
	float tmp = 0;
	for (int t = 0; t < u.channels(); t++) {
		float x = val(u, p, t) - val(v, q, t);
		x = __max(x, -x);
		tmp += x * x;
	}
	return tmp;
}


// SMART_PARAMETER(LAMBDA,0.9);

// inline float computeC_COMBI(Point p, Point q, const struct Img &u, const struct Img &v)
// {
// 	float a = computeC_AD(p, q, u, v);
// 	float b = computeC_SC(p, q, u, v);
// 	float l = LAMBDA();
// 	float r = (1 - l) * a + l * b;
// 	return r;
// }


// this macro defines the function CENSUS_NCC_WIN() that reads
// the environment variable with the same name
SMART_PARAMETER(CENSUS_NCC_WIN, 5);
SMART_PARAMETER(COMMON_COST_UPPER_BOUND, CENSUS_NCC_WIN()*CENSUS_NCC_WIN());

// fast census (input images must pre-processed by census transform)
inline float computeC_census_on_preprocessed_images(Point p, Point q, const cv::Mat& u, const cv::Mat& v, const cv::Mat& u2, const cv::Mat& v2, const cv::Mat& cost_mat,
	const std::vector<tiny_cnn::vec_t>& u_cnn, const std::vector<tiny_cnn::vec_t>& v_cnn)
{
	float r = 0;
	for (int t = 0; t < u.channels(); t++)
	{
		float vp = valzero(u, p, t);
		float vq = valzero(v, q, t);

		r += compute_census_distance_array((uint8_t*)(&vp), (uint8_t*)(&vq), sizeof(float));
	}
	// magic factor each channel contrinutes to r  4 bytes (32bits) 
	// to guarantee that the total cost is below 256 we take r*8 / nch
	return r *COMMON_COST_UPPER_BOUND() / (CENSUS_NCC_WIN()*CENSUS_NCC_WIN()) / u.channels(); // magic factor
}

// mic
inline float compute_mic_preprocessed_images(Point p, Point q, const cv::Mat& u, const cv::Mat& v, const cv::Mat& u2, const cv::Mat& v2, const cv::Mat& cost_mat,
	const std::vector<tiny_cnn::vec_t>& u_cnn, const std::vector<tiny_cnn::vec_t>& v_cnn){
	float census = 0.6*computeC_census_on_preprocessed_images(p, q, u, v, u2, v2, cost_mat, u_cnn, v_cnn);
	float mi = 0.4*((float*)cost_mat.data)[256 * ((u2.data)[(int)p.y*u2.cols + (int)p.x]) + (v2.data)[(int)q.y*v2.cols + (int)q.x]];
	return census + mi;
}

// mi
inline float compute_mi_preprocessed_images(Point p, Point q, const cv::Mat& u, const cv::Mat& v, const cv::Mat& u2, const cv::Mat& v2, const cv::Mat& cost_mat,
	const std::vector<tiny_cnn::vec_t>& u_cnn, const std::vector<tiny_cnn::vec_t>& v_cnn){
	float mi = ((float*)cost_mat.data)[256 * ((u2.data)[(int)p.y*u2.cols + (int)p.x]) + (v2.data)[(int)q.y*v2.cols + (int)q.x]];
	return mi;
}

// cnn
inline float compute_cnn(Point p, Point q, const cv::Mat& u, const cv::Mat& v, const cv::Mat& u2, const cv::Mat& v2, const cv::Mat& cost_mat,
	const std::vector<tiny_cnn::vec_t>& u_cnn, const std::vector<tiny_cnn::vec_t>& v_cnn){
	float cosin = tiny_cnn::cosine(u_cnn[p.y*u.cols + p.x], v_cnn[q.y*v.cols + q.x], IMAGE_OUT);
	cosin = ((-cosin) + 1)*COMMON_COST_UPPER_BOUND()/2.;
	return cosin;
}

// birchfield and tomasi absolute differences
inline float BTAD(Point p, Point q, int channel, const cv::Mat& u, const cv::Mat& v, const cv::Mat& u2, const cv::Mat& v2, const cv::Mat& cost_mat,
	const std::vector<tiny_cnn::vec_t>& u_cnn, const std::vector<tiny_cnn::vec_t>& v_cnn) {
	if (!check_inside_image(p, u)) return INFINITY;
	if (!check_inside_image(q, v)) return INFINITY;
#define min3(a,b,c) (((a)<(b))? (((a)<(c))?(a):(c)) : (((c)<(b))?(c):(b)) )
#define max3(a,b,c) (((a)>(b))? (((a)>(c))?(a):(c)) : (((c)>(b))?(c):(b)) )

	int t = channel;
	float IL = val(u, p, t);
	float ILp = IL, ILm = IL;
	if (p.x < u.cols - 1) ILp = (IL + val(u, p + Point(1, 0), t)) / 2.0;
	if (p.x >= 1) ILm = (IL + val(u, p + Point(-1, 0), t)) / 2.0;

	float IR = val(v, q, t);
	float IRp = IR, IRm = IR;
	if (q.x < v.cols - 1) IRp = (IR + val(v, q + Point(1, 0), t)) / 2.0;
	if (q.x >= 1) IRm = (IR + val(v, q + Point(-1, 0), t)) / 2.0;

	float IminR = min3(IRm, IRp, IR);
	float ImaxR = max3(IRm, IRp, IR);

	float IminL = min3(ILm, ILp, IL);
	float ImaxL = max3(ILm, ILp, IL);

	float dLR = max3(0, IL - ImaxR, IminR - IL);
	float dRL = max3(0, IR - ImaxL, IminL - IR);

	float BT = __min(dLR, dRL);
	return fabs(BT);
}


// birchfield and tomasi absolute differences
inline float computeC_BTAD(Point p, Point q, const cv::Mat& u, const cv::Mat& v, const cv::Mat& u2, const cv::Mat& v2, const cv::Mat& cost_mat,
	const std::vector<tiny_cnn::vec_t>& u_cnn, const std::vector<tiny_cnn::vec_t>& v_cnn) {
	if (!check_inside_image(p, u)) return INFINITY;
	if (!check_inside_image(q, v)) return INFINITY;
	float val = 0;
	for (int t = 0; t < u.channels(); t++)  {
		val += BTAD(p, q, t, u, v, u2, v2, cost_mat, u_cnn, v_cnn);
	}
	return val;
}
// birchfield and tomasi squared differences
inline float computeC_BTSD(Point p, Point q, const cv::Mat& u, const cv::Mat& v, const cv::Mat& u2, const cv::Mat& v2, const cv::Mat& cost_mat,
	const std::vector<tiny_cnn::vec_t>& u_cnn, const std::vector<tiny_cnn::vec_t>& v_cnn) {
	if (!check_inside_image(p, u)) return INFINITY;
	if (!check_inside_image(q, v)) return INFINITY;
	float val = 0;
	for (int t = 0; t < u.channels(); t++)  {
		float x = BTAD(p, q, t, u, v, u2, v2, cost_mat, u_cnn, v_cnn);
		val += x*x;
	}
	return val;
}

// Clipped NCC
// NOTE: window size = 3x3
inline float computeC_clippedNCC(Point p, Point q, const cv::Mat& u, const cv::Mat& v, const cv::Mat& u2, const cv::Mat& v2, const cv::Mat& cost_mat,
	const std::vector<tiny_cnn::vec_t>& u_cnn, const std::vector<tiny_cnn::vec_t>& v_cnn)
{
	int hwindow = CENSUS_NCC_WIN() / 2;
	float r = 0;
	float NCC = 0;
	for (int t = 0; t < u.channels(); t++)
	{
		float mu1 = 0; float mu2 = 0;
		float s1 = 0; float s2 = 0;
		float prod = 0;
		int n = 0;
		for (int i = -hwindow; i <= hwindow; i++)
		for (int j = -hwindow; j <= hwindow; j++)
		{
			float v1 = valnan(u, p + Point(i, j), t);
			float v2 = valnan(v, q + Point(i, j), t);
			if (isnan(v1) || isnan(v2)) return INFINITY;
			mu1 += v1;    mu2 += v2;
			s1 += v1*v1; s2 += v2*v2; prod += v1*v2;
			n++;
		}
		mu1 /= n; mu2 /= n;
		s1 /= n;  s2 /= n; prod /= n;

		NCC += (prod - mu1*mu2) / sqrt(__max(0.0000001, (s1 - mu1*mu1)*(s2 - mu2*mu2)));
	}
	float clippedNCC = u.channels() - __max(0, __min(NCC, u.channels()));
	return clippedNCC * 64;
}



//// global table of all the cost functions
struct distance_functions{
	cost_t f;
	const char *name;
} global_table_of_distance_functions[] = {
#define REGISTER_FUNCTIONN(x,xn) {x, xn}
	REGISTER_FUNCTIONN(computeC_AD, "ad"),
	REGISTER_FUNCTIONN(computeC_SD, "sd"),
	REGISTER_FUNCTIONN(computeC_census_on_preprocessed_images, "census"),
	REGISTER_FUNCTIONN(computeC_clippedNCC, "ncc"),
	REGISTER_FUNCTIONN(computeC_BTAD, "btad"),
	REGISTER_FUNCTIONN(computeC_BTSD, "btsd"),
	REGISTER_FUNCTIONN(compute_mic_preprocessed_images, "mic"),
	REGISTER_FUNCTIONN(compute_mi_preprocessed_images, "mi"),
	REGISTER_FUNCTIONN(compute_cnn, "cnn"),
#undef REGISTER_FUNCTIONN
	{ NULL, "" },
};
int get_distance_index(const char *name) {
	int r = 0; // default cost function is computeC_AD (first in table distance_functions)
	for (int i = 0; global_table_of_distance_functions[i].f; i++)
	if (strcmp(name, global_table_of_distance_functions[i].name) == 0)
		r = i;
	return r;
}


//// global table of the prefilter names
const char* global_table_of_prefilters[] = {
	"none",
	"census",
	"sobelx",
	"gblur",
	NULL,
};
int get_prefilter_index(const char *name) {
	int r = 0;
	for (int i = 0; global_table_of_prefilters[i]; i++)
	if (strcmp(name, global_table_of_prefilters[i]) == 0)
		r = i;
	return r;
}

static inline std::shared_ptr<tiny_cnn::network<tiny_cnn::hinge_loss, tiny_cnn::RMSprop>> getCNNInstance(const std::string& weights) {
	using namespace tiny_cnn;
	using namespace tiny_cnn::activation;
	static std::shared_ptr<network<hinge_loss, RMSprop>> nn;
	if (!nn) {
		nn = std::make_shared<network<hinge_loss, RMSprop>>();
		*nn
			<< convolutional_layer<relu>(IMAGE_WIDTH, IMAGE_HEIGHT, 3, 1, 64)
			<< convolutional_layer<relu>(IMAGE_WIDTH - 2, IMAGE_HEIGHT - 2, 3, 64, 64)
			<< convolutional_layer<relu>(IMAGE_WIDTH - 4, IMAGE_HEIGHT - 4, 3, 64, 64)
			<< convolutional_layer<relu>(IMAGE_WIDTH - 6, IMAGE_HEIGHT - 6, 3, 64, 64)
			<< convolutional_layer<>(IMAGE_WIDTH - 8, IMAGE_HEIGHT - 8, 3, 64, 64);

		std::ifstream input_weights(weights);
		if (input_weights.good()){
			input_weights >> *nn;
		}
	}
	return nn;
}

void compute_cnn_vectors(const cv::Mat& in_im, std::vector<tiny_cnn::vec_t>& out, const std::string& cnn_weights, const cv::Mat& mask){
	using namespace tiny_cnn;
	std::shared_ptr<tiny_cnn::network<tiny_cnn::hinge_loss, tiny_cnn::RMSprop>> net = getCNNInstance(cnn_weights);
	//std::vector<std::vector<int>> v(10, std::vector<int>(10, 1));
	cv::Mat padded;
	cv::Mat gray_float_u, temp;
 	cv::cvtColor(in_im, gray_float_u, CV_BGR2GRAY);

	double mean1, std1;
	cv::Scalar mean1s, std1s;
	cv::meanStdDev(gray_float_u, mean1s, std1s, mask);
	mean1 = mean1s[0], std1 = std1s[0];
	subtract(gray_float_u, mean1, gray_float_u);
	gray_float_u /= std1;


	cv::copyMakeBorder(gray_float_u, padded, IMAGE_HEIGHT / 2, IMAGE_HEIGHT / 2, IMAGE_WIDTH / 2, IMAGE_WIDTH / 2, cv::BORDER_CONSTANT, cv::Scalar(0));
	padded.convertTo(temp, CV_64F);
	omp_set_num_threads(CNN_TASK_SIZE);
#pragma omp parallel for
	for (int i = 0; i < in_im.rows; i++){
		for (int j = 0; j < in_im.cols; j++){
			if (mask.data[i*in_im.cols + j] == MASK_OUT){
				continue;
			}
			int tid = omp_get_thread_num();
			cv::Rect rect(j, i, IMAGE_WIDTH, IMAGE_HEIGHT);
			cv::Mat subimg(temp, rect);
			vec_t net_in;
			for (int k = 0; k < subimg.rows; ++k) {
				net_in.insert(net_in.end(), (tiny_cnn::float_t*)subimg.ptr<uchar>(k), (tiny_cnn::float_t*)subimg.ptr<uchar>(k)+subimg.cols);
			}
			net->fprop_siam_res(net_in, out[i*in_im.cols + j], tid);
		}
	}
	std::cout << "added";
}

void fill_cnn_vector(const cv::Mat& in_im, std::vector<tiny_cnn::vec_t>& u_cnn, const std::string& cache, const std::string& cnn_weights, const cv::Mat& mask){
	using namespace tiny_cnn;
	std::ifstream cache_vec_in(cache, std::ios::binary | std::ios::in);
	if (cache_vec_in.good()){
		for (vec_t& vec : u_cnn){
			cache_vec_in.read(reinterpret_cast<char*>(&vec[0]), std::streamsize(IMAGE_OUT*sizeof(tiny_cnn::float_t)));
		}
		cache_vec_in.close();
	} else{
		cache_vec_in.close();
		compute_cnn_vectors(in_im, u_cnn, cnn_weights, mask);
		std::ofstream cache_vec_out(cache, std::ios::binary | std::ios::out);
		for (vec_t& vec : u_cnn){
			cache_vec_out.write(reinterpret_cast<const char*>(&vec[0]), std::streamsize(IMAGE_OUT*sizeof(tiny_cnn::float_t)));
		}
		cache_vec_out.close();
	}
}




//////////////////////////////////////////////
//////////////////////////////////////////////
//////////////////////////////////////////////
//////////////////////////////////////////////
//////////////////////////////////////////////
// costvolume_t contains a large vectors of Dvec (vectors)
// This complex data structure is managed transparently thanks
// to std::vectors wihtout caring aboyt copy or deallocation.
// But allocation is quite expensive as all vectors are 
// initialized upon creation. 
// DVEC_ALLOCATION_HACK implements a more efficient memory 
// management (using malloc and free) while exposing the 
// same interface as costvolume_t. 
// However the code is less elegant and maybe even buggy. 

#ifdef DVEC_ALLOCATION_HACK

#include "dvec2.cc"

struct costvolume_t {
	int npix;
	int ndata;
	struct Dvec *vectors;
	float *alldata;


	inline Dvec  operator[](int i) const  {
		return this->vectors[i];
	}
	inline Dvec& operator[](int i)        {
		return this->vectors[i];
	}

	costvolume_t() {
		int npix = 0;
		int ndata = 0;
		this->vectors = NULL;
		this->alldata = NULL;
	}

	costvolume_t(const struct costvolume_t &src)
	{
		npix = src.npix;
		ndata = src.ndata;
		vectors = (struct Dvec*) malloc(sizeof(struct Dvec)*npix);
		alldata = (float*)calloc(ndata, sizeof(float));
		float *baseptr = alldata;
		memcpy(vectors, src.vectors, sizeof(struct Dvec)*npix);
		memcpy(alldata, src.alldata, sizeof(float)*ndata);

		int size = 0;
		for (int i = 0; i < npix; i++)  {
			vectors[i].data = baseptr + size;
			size += (int)((int)vectors[i].max - (int)vectors[i].min + 1);
		}
	}

	costvolume_t(costvolume_t&& original)
		: npix(original.npix), ndata(original.ndata), vectors(original.vectors), alldata(original.alldata)
	{
		original.vectors = nullptr;
		original.alldata = nullptr;
	}

	~costvolume_t(void)
	{
		if (this->vectors != NULL) free(this->vectors);
		if (this->alldata != NULL) free(this->alldata);
	}

};

struct costvolume_t allocate_costvolume(const cv::Mat& min, const cv::Mat& max, const cv::Mat& maskdet)
{
	int npix = min.cols*min.rows;
	int size = 0;
	std::vector< int > pos(npix);
	float* maxdata = (float*)max.data;
	float* mindata = (float*)min.data;
	for (int i = 0; i < npix; i++)  {
		pos[i] = size;
		if ((maskdet.data)[i] == MASK_OUT){
			mindata[i] = 0;
			maxdata[i] = 0;
		}
		size += (int)((int)maxdata[i] - (int)mindata[i] + 1); //???? todo
	}

	struct costvolume_t cv;
	cv.npix = npix;
	cv.ndata = size;
	cv.vectors = (struct Dvec*) malloc(sizeof(struct Dvec)*npix); // here is the trick
	cv.alldata = (float*)calloc(size, sizeof(float));
	float *baseptr = cv.alldata;

#pragma omp parallel for
	for (int i = 0; i < npix; i++) {
		cv.vectors[i].init(mindata[i], maxdata[i], baseptr + pos[i]);
	}

	return cv;
}



//////////////////////////////////////////////
//////////////////////////////////////////////
#else // DVEC_ALLOCATION_HACK
//////////////////////////////////////////////
////////    USING VECTORS     ////////////////

#include "dvec.cc"

struct costvolume_t {
	std::vector< Dvec > vectors;
	inline Dvec  operator[](int i) const  { return this->vectors[i]; }
	inline Dvec& operator[](int i)        { return this->vectors[i]; }
};


struct costvolume_t allocate_costvolume(const cv::Mat& min, const cv::Mat& max, const cv::Mat& maskdet)
{
	struct costvolume_t cv;
	cv.vectors = std::vector< Dvec >(min.total()); // so bad - default initial-on
	float* datamin = (float*)min.data;
	float* datamax = (float*)max.data;

	for (int i = 0; i < min.total(); i++) {
		if ((maskdet.data)[i] == MASK_OUT){
			datamin[i] = 0;
			datamax[i] = 0;
		}
		cv[i].init(datamin[i], datamax[i]);
	}

	return cv;
}


#endif // DVEC_ALLOCATION_HACK
//////////////////////////////////////////////
//////////////////////////////////////////////
//////////////////////////////////////////////
//////////////////////////////////////////////
//////////////////////////////////////////////

struct costvolume_t allocate_and_fill_sgm_costvolume(const cv::Mat& in_u, // source (reference) image                      
	const cv::Mat& in_v, // destination (match) image                  
	const cv::Mat& dminI,// per pixel max&min disparity
	const cv::Mat& dmaxI,
	char* prefilter,        // none, sobel, census(WxW)
	char* distance,         // census, l1, l2, ncc(WxW), btl1, btl2
	float truncDist, // truncated differences
	cv::Mat& u, cv::Mat& v,
	const cv::Mat& prev_out, bool prev_out_exists,
	cv::Mat& u2, cv::Mat& v2, const cv::Mat& maskdet, const cv::Mat& maskdet2,
	char *cnnweights, char **params, bool direct)
{
	int nx = in_u.cols;
	int ny = in_u.rows;
	int nch = in_u.channels();

	//cv::Mat u = in_u.clone();
	//cv::Mat v = in_v.clone();

	// 0. pick the prefilter and cost functions
	int distance_index = get_distance_index(distance);
	int prefilter_index = get_prefilter_index(prefilter);
	//cost_t cost = global_table_of_distance_functions[distance_index].f;

	// 1. parameter consistency check
	if (distance_index == get_distance_index("census") || prefilter_index == get_prefilter_index("census")) {
		if (TSGM_DEBUG()) printf("costvolume: changing both distance and prefilter to CENSUS\n");
		distance_index = get_distance_index("census");
		prefilter_index = get_prefilter_index("census");
	}
	cost_t cost = global_table_of_distance_functions[distance_index].f;
	if (TSGM_DEBUG()) printf("costvolume: selecting distance  %s\n", global_table_of_distance_functions[distance_index].name);
	if (TSGM_DEBUG()) printf("costvolume: selecting prefilter %s\n", global_table_of_prefilters[prefilter_index]);
	if (TSGM_DEBUG()) printf("costvolume: truncate distances at %f\n", truncDist);

	// 2. apply prefilters if needed
	if (prefilter_index == get_prefilter_index("census") && (u.empty() || v.empty())) {
		int winradius = CENSUS_NCC_WIN() / 2;
		if (TSGM_DEBUG()) printf("costvolume: applying census with window of size %d\n", winradius * 2 + 1);
		if (u.empty()){
			u = census_transform(in_u, winradius);
		}
		if (v.empty()){
			v = census_transform(in_v, winradius);
		}
	}
	if (prefilter_index == get_prefilter_index("sobelx") && (u.empty() || v.empty())) {
		if (TSGM_DEBUG()) printf("costvolume: applying sobel filter\n");
		float sobel_x[] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
		if (u.empty()){
			u = apply_filter(in_u, sobel_x, 3, 3, 1);
		}
		if (v.empty()){
			v = apply_filter(in_v, sobel_x, 3, 3, 1);
		}
	}
	if (prefilter_index == get_prefilter_index("gblur") && (u.empty() || v.empty())) {
		if (TSGM_DEBUG()) printf("costvolume: applying gblur(s=1) filter\n");
		if (u.empty()){
			u = gblur_truncated(in_u, 1.0);
		}
		if (v.empty()){
			v = gblur_truncated(in_v, 1.0);
		}
	}


	cv::Mat cost_mat(256, 256, CV_32FC1);
	if (prev_out_exists && (distance_index == get_distance_index("mic") || distance_index == get_distance_index("mi"))){
		if (u2.empty()){
			in_u.convertTo(u2, CV_8UC3);
			cv::cvtColor(u2, u2, CV_BGR2GRAY);
		}
		if (v2.empty()){
			in_v.convertTo(v2, CV_8UC3);
			cv::cvtColor(v2, v2, CV_BGR2GRAY);
		}
		fill_cost_mat(cost_mat, u2, prev_out, v2, maskdet, maskdet2);
		//cv::FileStorage file("data2/Adirondack/cost_mat.xml", cv::FileStorage::WRITE);
		//file << "Vocabulary" << cost_mat;
		winsorize_mat(cost_mat);
		normalize(cost_mat, cost_mat, 0., COMMON_COST_UPPER_BOUND(), cv::NORM_MINMAX);
		//std::cout << "!!!" << in_u.isContinuous();
		//cv::FileStorage file("data2/Motorcycle/cost_mat.xml", cv::FileStorage::WRITE);
		//file << "Vocabulary" << cost_mat;
		//mutual_entr1 = -1 * (indiv_entr1 + indiv_entr2 - mutual_entr1);
	}

	std::vector<tiny_cnn::vec_t> v_cnn;
	std::vector<tiny_cnn::vec_t> u_cnn;
	if (distance_index == get_distance_index("cnn")){
		std::string u_cnn_cache = direct ? params[1] : params[2]; u_cnn_cache += ".cnn";
		std::string v_cnn_cache = direct ? params[2] : params[1]; v_cnn_cache += ".cnn";
		u = in_u;
		v = in_v;
		
		if (strcmp(cnnweights, "") == 0){
			throw "no cnn_weights";
		}

		u_cnn = std::vector<tiny_cnn::vec_t>(in_u.rows*in_u.cols, tiny_cnn::vec_t(IMAGE_OUT, 0));
		fill_cnn_vector(in_u, u_cnn, u_cnn_cache, cnnweights, maskdet);

		v_cnn = std::vector<tiny_cnn::vec_t>(in_v.rows*in_v.cols, tiny_cnn::vec_t(IMAGE_OUT, 0));
		fill_cnn_vector(in_v, v_cnn, v_cnn_cache, cnnweights, maskdet2);

	}

	// 3. allocate the cost volume 5
	struct costvolume_t CC = allocate_costvolume(dminI, dmaxI, maskdet);

	// 4. apply it 
#pragma omp parallel for
	for (int jj = 0; jj < ny; jj++)
	for (int ii = 0; ii < nx; ii++)
	{
		int pidx = (ii + jj*nx);
		int allinvalid = 1;
		if ((maskdet.data)[jj*nx + ii] == MASK_OUT){
			continue;
		}

		for (int o = CC[pidx].min; o <= CC[pidx].max; o++)
		{
			Point p(ii, jj);      // current point on left image
			Point q = p + Point(o, 0); // other point on right image
			// 4.1 compute the cost 
			float e = truncDist * u.channels();
			if (check_inside_image(q, in_v) && check_inside_mask(q, maskdet2))
				e = cost(p, q, u, v, u2, v2, cost_mat, u_cnn, v_cnn);
			// 4.2 truncate the cost (if needed)
			e = __min(e, truncDist * u.channels());
			// 4.3 store it in the costvolume
			CC[pidx].set_nolock(o, e); // no pragma omp critic inside set
			if (isfinite(e)) allinvalid = 0;
		}
		// SAFETY MEASURE: If there are no valid hypotheses for this pixel 
		// (ie all hypotheses fall outside the target image or are invalid in some way)
		// then the cost must be set to 0, for all the available hypotheses
		// Leaving inf would be propagated and invalidate the entire solution 
		if (allinvalid) {
			for (int o = CC[pidx].min; o <= CC[pidx].max; o++)
			{
				Point p(ii, jj);      // current point on left image
				Point q = p + Point(o, 0); // other point on right image
				CC[pidx].set_nolock(o, 0); // no pragma omp critic inside set
			}
		}
	}
	return CC;
}


#endif //COSTVOLUME_H_
