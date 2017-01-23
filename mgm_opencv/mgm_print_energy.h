/* Copyright (C) 2015, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>,
 *                     Carlo de Franchis <carlo.de-franchis@ens-cachan.fr>,
 *                     Enric Meinhardt <enric.meinhardt@cmla.ens-cachan.fr>*/
#ifndef MGM_PRINT_ENERGY_H_
#define MGM_PRINT_ENERGY_H_
/******************** PRINT ENERGY ********************/

#include "img_tools.h"
#include "point.h"

// EVALUATE THE ENERGY OF THE CURRENT SOLUTION
// type==0 : truncated L1
// type==1 : L1s
// type==2 : L2
float evaluate_energy_4connected(const cv::Mat &u, const float* outdisp, struct costvolume_t &CC, cv::Mat& E_out, float P1, float P2, int type) {
	int nx = u.cols;
	int ny = u.rows;

	// EVALUATE THE ENERGY OF THE CURRENT SOLUTION
	float ENERGY = 0;
	float ENERGYL2 = 0;
	float ENERGYtrunc = 0;

	cv::Mat E = cv::Mat::zeros(ny, nx, CV_32FC1);
	cv::Mat EL2 = cv::Mat::zeros(ny, nx, CV_32FC1);
	cv::Mat Etrunc = cv::Mat::zeros(ny, nx, CV_32FC1);
	//struct Img E(nx, ny);
	//struct Img EL2(nx, ny);
	//struct Img Etrunc(nx, ny);

	for (int y = 0; y < ny; y++)
	for (int x = 0; x < nx; x++)
	{
		Point p(x, y);				   // current point
		int pidx = (p.x + p.y *nx);

		//E[pidx] = 0;
		//EL2[pidx] = 0;
		//Etrunc[pidx] = 0;

		float G = 0; // contribution for the current point 
		float GL2 = 0; // contribution for the current point 
		float Gtrunc = 0; // contribution for the current point 

		// DATA TERM
		int o = outdisp[pidx];
		G += CC[pidx][o];
		GL2 += CC[pidx][o];
		Gtrunc += CC[pidx][o];

		// EDGE POTENTIALS
		Point directions[] = { Point(-1, 0), Point(0, 1),
			Point(1, 0), Point(0, -1),
			Point(-1, 0) };

		int N = 4;
		for (int t = 0; t < N; t++) {
			Point pr = p + directions[t];
			Point pq = p + directions[t + 1]; // needed for L2
			if (!check_inside_image(pr, u)) continue;
			if (!check_inside_image(pq, u)) continue;
			int pridx = (pr.x + pr.y*nx);
			int pqidx = (pq.x + pq.y*nx);
			int oor = outdisp[pridx];
			int ooq = outdisp[pqidx];

			G += fabs((float)oor - o) / N;

			GL2 += sqrt((oor - o)*(oor - o) + (ooq - o)*(ooq - o)) / N;

			if (fabs((float)oor - o) <= 1) Gtrunc += P1 / N;
			else Gtrunc += P2 / N;
		}

		ENERGY += G;
		((float*)E.data)[pidx] = G;

		ENERGYL2 += GL2;
		((float*)EL2.data)[pidx] = GL2;

		ENERGYtrunc += Gtrunc;
		((float*)Etrunc.data)[pidx] = G;

	}
	if (type == 1) {
		E_out = E;
		return ENERGY;
	}
	if (type == 2) {
		E_out = EL2;
		return ENERGYL2;
	}
	E_out = Etrunc;
	return ENERGYtrunc;


}



void print_solution_energy(const cv::Mat& in_u, const float* disp, struct costvolume_t &CC, float P1, float P2) {
	if (TSGM_DEBUG()) {
		// DEBUG INFORMATION
		cv::Mat E;
		printf(" ENERGY L1trunc: %.9e\t", evaluate_energy_4connected(in_u, disp, CC, E, P1, P2, 0));
		iio_write_vector_split((char*)"/tmp/ENERGY_L1trunc.tif", E);
		printf("L1: %.9e\t", evaluate_energy_4connected(in_u, disp, CC, E, P1, P2, 1));
		printf("L2: %.9e\n", evaluate_energy_4connected(in_u, disp, CC, E, P1, P2, 2));
	}
	else {
		printf("\n");
	}
}

void print_error_rate(const cv::Mat& in_u, const cv::Mat& gt, const cv::Mat& mask, float epsilon, float reversed = 1., std::string file = ""){
	if (!(in_u.channels() == 1 && gt.channels() == 1 && mask.channels() == 1)){
		fprintf(stderr, "invalid params - channels number");
		return;
	}
	if (!(in_u.isContinuous() && gt.isContinuous() && mask.isContinuous())){
		fprintf(stderr, "invalid params - not continuous in print error rate");
		return;
	}

	uint32_t positive = 0, total = 0, hidden = 0, positive_hidden = 0, others = 0;
	//std::cout << "fd" << mask.rows*mask.cols;
	//std::ofstream file;
	//file.open("Numbers.txt", std::ios::out);
	float* in_u_d = (float*)in_u.data;
	float* gt_d = (float*)gt.data;
	for (int i = 0; i < in_u.rows*in_u.cols; i++){
		float val = abs(in_u_d[i] + reversed*gt_d[i]);
		if (mask.data[i] == 255){
			total++;
			if ((!std::isfinite(in_u_d[i]) && !std::isfinite(gt_d[i])) || (val <= epsilon && std::isfinite(val))){
				positive++;
			}
			//file << abs(((float*)in_u.data)[i] + reversed*((float*)gt.data)[i]) << " ";
		}
		else if (mask.data[i] != 0){
			hidden++;
			if ((!std::isfinite(in_u_d[i]) && !std::isfinite(gt_d[i])) || (val <= epsilon && std::isfinite(val))){
				positive_hidden++;
				//file << in_u_d[i] << ", " << gt_d[i] << ", " << abs(in_u_d[i] + reversed*gt_d[i]) << "\n";
			}
		}
		else{
			others++;
		}
	}
	printf("\n%u/%u ~ %f masked \n%u/%u ~ %f unmasked (%u others) - %u/%u~ %f ovr\n", positive, total, (float)positive / total,
		positive_hidden, hidden, (float)positive_hidden / hidden, others,
		positive + positive_hidden, total + hidden, ((float)positive + positive_hidden) / (total + hidden));
	if (!file.empty()){
		std::ofstream outfile;
		outfile.open(file, std::ios_base::app);
		outfile << positive + positive_hidden << " ";
		outfile << total + hidden << " ";
	}
}

void compare_occl_lr_gt(const cv::Mat& lr, const cv::Mat& gt){
	if (!(lr.channels() == 1 && gt.channels() == 1)){
		fprintf(stderr, "invalid params - channels number");
		return;
	}
	if (!(lr.isContinuous() && gt.isContinuous())){
		fprintf(stderr, "invalid params - not continuous in compare occl lr gt", lr.isContinuous(), gt.isContinuous());
		return;
	}

	int lr_total = 0, gt_total = 0, both = 0;
	for (int i = 0; i < lr.rows*lr.cols; i++){
		if ((lr.data)[i] != 255 && (gt.data)[i] != 0){
			lr_total++;
			if ((gt.data)[i] != 255){
				both++;
			}
		}
		if ((gt.data)[i] != 255 && (gt.data)[i] != 0){
			gt_total++;
		}
	}

	printf("\nlr_total %u, gt_total %u, both %u \n", lr_total, gt_total, both);
}

void compare_occl_lr_gt_mismatches_rate(const cv::Mat& lr, const cv::Mat& gt, const cv::Mat& in_u, const cv::Mat& gt_im, float epsilon, float reversed = 1.){
	if (!(lr.channels() == 1 && gt.channels() == 1)){
		fprintf(stderr, "invalid params - channels number");
		return;
	}
	if (!(lr.isContinuous() && gt.isContinuous())){
		fprintf(stderr, "invalid params - not continuous in compare occl lr gt mism");
		return;
	}

	float* in_u_d = (float*)in_u.data;
	float* gt_d = (float*)gt_im.data;
	int total = 0, mismatches = 0;
	for (int i = 0; i < lr.rows*lr.cols; i++){
		if ((lr.data)[i] != 255 && (gt.data)[i] != 0){
			if ((gt.data)[i] == 255){
				total++;
				float val = abs(in_u_d[i] + reversed*gt_d[i]);
				if (!((!std::isfinite(in_u_d[i]) && !std::isfinite(gt_d[i])) || (val <= epsilon && std::isfinite(val)))){
					mismatches++;
				}
			}
		}
	}

	printf("\ntotal %u , mismatches %u \n", total, mismatches);
}

void mismatches_class_wrong_rate(const cv::Mat& map_type, const cv::Mat& gt_mask){


	int total = 0, wrong_mismatches = 0;
	for (int i = 0; i < map_type.rows*map_type.cols; i++){
		if ((map_type.data)[i] != 255 && (gt_mask.data)[i] != 0){
			if ((gt_mask.data)[i] != 255){
				total++;
				if ((map_type.data)[i] == MISM){
					wrong_mismatches++;
				}
			}
		}
	}

	printf("\ntotal %u , wrong_mismatches %u \n", total, wrong_mismatches);
}


void print_aver_error(const cv::Mat& in_u, const cv::Mat& gt, const cv::Mat& mask, float limit = 100., float reversed = 1., std::string file = ""){
	if (!(in_u.channels() == 1 && gt.channels() == 1 && mask.channels() == 1)){
		fprintf(stderr, "invalid params - channels number");
		return;
	}
	if (!(in_u.isContinuous() && gt.isContinuous() && mask.isContinuous())){
		fprintf(stderr, "invalid params - not continuous in avg error");
		return;
	}

	uint32_t total = 0, hidden = 0;
	//std::cout << "fd" << mask.rows*mask.cols;
	//std::ofstream file;
	//file.open("Numbers.txt", std::ios::out);
	float* in_u_d = (float*)in_u.data;
	float* gt_d = (float*)gt.data;
	float sum_total = 0, sum_hidden = 0;

	for (int i = 0; i < in_u.rows*in_u.cols; i++){
		float val = abs(in_u_d[i] + reversed*gt_d[i]);
		if (mask.data[i] == 255){
			if (std::isfinite(val) && val < limit){
				total++;
				sum_total += val;
			}
			//file << abs(((float*)in_u.data)[i] + reversed*((float*)gt.data)[i]) << " ";
		}
		else if (mask.data[i] != 0){
			if (std::isfinite(val) && val < limit){
				hidden++;
				sum_hidden += val;
				//file << in_u_d[i] << ", " << gt_d[i] << ", " << abs(in_u_d[i] + reversed*gt_d[i]) << "\n";
			}
		}

	}

	float aver_total = sum_total / total;
	float aver_hidden = sum_hidden / hidden;
	float aver_all = (sum_total + sum_hidden) / (total + hidden);
	float disp_total = 0, disp_hidden = 0;
	for (int i = 0; i < in_u.rows*in_u.cols; i++){
		float val = abs(in_u_d[i] + reversed*gt_d[i]);
		if (mask.data[i] == 255){
			if (std::isfinite(val) && val < limit){
				disp_total += (val - aver_total)*(val - aver_total);
			}
			//file << abs(((float*)in_u.data)[i] + reversed*((float*)gt.data)[i]) << " ";
		}
		else if (mask.data[i] != 0){
			if (std::isfinite(val) && val < limit){
				disp_hidden += (val - aver_hidden)*(val - aver_hidden);
				//file << in_u_d[i] << ", " << gt_d[i] << ", " << abs(in_u_d[i] + reversed*gt_d[i]) << "\n";
			}
		}

	}

	printf("\n%f masked \n %f unmasked (total %f)\n", aver_total, aver_hidden, aver_all);
	printf("\n%f/%d masked \n %f/%d unmasked (total %f/%d)\n", sum_total , total, sum_hidden , hidden, (sum_total + sum_hidden) , (total + hidden));
	printf("disp \n%f masked \n %f unmasked \n", sqrt(disp_total / total), sqrt(disp_hidden / hidden));
	if (!file.empty()){
		std::ofstream outfile;
		outfile.open(file, std::ios_base::app);
		outfile << (int)(sum_total + sum_hidden) << " ";
	}
}
#endif //MGM_PRINT_ENERGY_H_
