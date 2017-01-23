/*
	Copyright (c) 2013, Taiga Nomi
	All rights reserved.

	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions are met:
	* Redistributions of source code must retain the above copyright
	notice, this list of conditions and the following disclaimer.
	* Redistributions in binary form must reproduce the above copyright
	notice, this list of conditions and the following disclaimer in the
	documentation and/or other materials provided with the distribution.
	* Neither the name of the <organization> nor the
	names of its contributors may be used to endorse or promote products
	derived from this software without specific prior written permission.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
	EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
	DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
	(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
	LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
	ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
	*/
#include <iostream>
//#include <boost/timer.hpp>
//#include <boost/progress.hpp>

#include "tiny_cnn.h"
//#define NOMINMAX
//#include "imdebug.h"
#include "stereo_im_parser.h"

void my_test();
void my_test_test();
void create_half_size();

using namespace tiny_cnn;
using namespace tiny_cnn::activation;

int main(void) {
	// evaluating
	my_test_test();
	// training
	//my_test();
	system("pause");
	return 0;
}

// training
void my_test(void) {
	// construct LeNet-5 architecture
	network<hinge_loss, RMSprop> nn;
	network<hinge_loss, RMSprop> nn2;

	nn
		<< convolutional_layer<relu>(11, 11, 3, 1, 64)
		<< convolutional_layer<relu>(9, 9, 3, 64, 64)
		<< convolutional_layer<relu>(7, 7, 3, 64, 64)
		<< convolutional_layer<relu>(5, 5, 3, 64, 64)
		<< convolutional_layer<>(3, 3, 3, 64, 64);

	nn2
		<< convolutional_layer<relu>(11, 11, 3, 1, 64)
		<< convolutional_layer<relu>(9, 9, 3, 64, 64)
		<< convolutional_layer<relu>(7, 7, 3, 64, 64)
		<< convolutional_layer<relu>(5, 5, 3, 64, 64)
		<< convolutional_layer<>(3, 3, 3, 64, 64);

	int cntr = 0;
	auto on_enumerate_minibatch = [&](){
		//disp += minibatch_size; 
		cntr++;
		if (cntr % 10 == 0){
			//std::cout << cntr << std::endl;
		}
	};

	auto on_enumerate_epoch = [&](){
		//nn.test_siam_all(array_left, array_right, array_other, labels);
		//nn.optimizer().alpha *= 0.85; // decay learning rate
		//nn.optimizer().alpha = std::max(0.00001, nn.optimizer().alpha);
		//nn2.optimizer().alpha *= 0.85; // decay learning rate
		//nn2.optimizer().alpha = std::max(0.00001, nn2.optimizer().alpha);
	};

	int epoch_number = 15;
	int minibatch_size = 128;
	int start_fold = 0;
	int start_ep = 0;
	tiny_cnn::float_t initial_l_rate = 0.0001;
	tiny_cnn::float_t epoch_l_decr = 0.75;
	tiny_cnn::float_t cur_l_rate = initial_l_rate*pow(epoch_l_decr, start_ep);
	int need_weights_init = 1;
	std::string folder_path = "..";
	std::ifstream input_weights(folder_path + "/data/LeNet-weights");
	if (input_weights.good()){
		input_weights >> nn;
		need_weights_init = 0;
		std::cout << "weights are given in file" << folder_path << "/data/LeNet-weights"  << std::endl;
		input_weights.close();
	}

	nn.optimizer().alpha = cur_l_rate;
	nn2.optimizer().alpha = cur_l_rate;
	std::cout << "load models..." << std::endl;
	//std::vector<std::string> folders = { "Adirondack", "Jadeplant", "Motorcycle", "Piano", "Pipes", "Playroom", "Playtable", "Recycle", "Shelves" };
	std::vector<std::string> folders = { "Backpack", "Bicycle1", "Cable", "Classroom1", "Couch", "Flowers", "Mask", "Sticks", "Storage" };
	//std::vector<std::string> folders = { "Motorcycle" };
	int iter = 0;
	for (int ep = start_ep; ep < epoch_number; ep++){
		for (int fold = start_fold; fold < folders.size(); fold++){
			std::string folder = folders.at(fold);
			iter++;
			std::vector<stereoPair> stereo_p;
			stereo_p.push_back(stereoPair(folder_path + "/data/" + folder + "/im0.png", folder_path + "/data/" + folder + "/im1.png", folder_path + "/data/" + folder + "/disp.pfm", folder_path + "/data/" + folder + "/mask0nocc.png"));
			std::vector<trainPair> pairs = create_images(stereo_p, -1., 1.);
			std::vector<vec_t> array_left, array_right, array_other;
			std::vector<label_t> labels;
			read_images(pairs.at(0), array_left, array_right, array_other, labels);


			//nn.test_siam_all(array_left, array_right, array_other, labels);
			std::cout << "start learning folder " << fold << " " << folder << " with l_r " << cur_l_rate << " on epoch " << ep << std::endl;
			nn.train_siam(nn2, array_left, array_right, array_other, labels, minibatch_size, 1, on_enumerate_minibatch, on_enumerate_epoch, need_weights_init, CNN_TASK_SIZE, iter == 1);
			std::cout << "end training " << iter << " " << folder << std::endl;
			nn.test_siam_all(array_left, array_right, array_other, labels);

			std::remove(pairs.at(0).left_dst.c_str());
			std::remove(pairs.at(0).right_dst.c_str());
			std::remove(pairs.at(0).labels.c_str());

			// save networks
			std::ofstream ofs(folder_path + "/data/LeNet-weights" + std::to_string(iter));
			ofs << nn;
			ofs.close();
		}
		//cur_l_rate *= epoch_l_decr;
		nn.optimizer().alpha = cur_l_rate;
		nn2.optimizer().alpha = cur_l_rate;
	}


	// test and show results
	//nn.test(test_images, test_labels).print_detail(std::cout);

}

void create_half_size(){
	std::string folder_path = "..";
	//std::vector<std::string> folders = { "Backpack", "Bicycle1", "Cable", "Classroom1", "Couch", "Flowers", "Mask" };
	std::vector<std::string> folders = { "Sticks", "Storage" };
	for (int fold = 0; fold < folders.size(); fold++){
		cv::Mat im0 = cv::imread(folder_path + "/data/" + folders[fold] + "/im02.png");
		cv::Mat im1 = cv::imread(folder_path + "/data/" + folders[fold] + "/im12.png");
		cv::Mat mask0nocc = cv::imread(folder_path + "/data/" + folders[fold] + "/mask0nocc2.png");
		int gt_w, gt_h;
		float* gt_vec = read_pfm_file(std::string(folder_path + "/data/" + folders[fold] + "/disp2.pfm").c_str(), &gt_w, &gt_h);
		cv::Mat gt_im = cv::Mat(gt_h, gt_w, CV_32FC1, gt_vec);
		gt_im /= 2.;
		//cv::flip(gt_im, gt_im, 0);
		cv::resize(im0, im0, cv::Size(im0.cols/2, im0.rows/2));
		cv::resize(im1, im1, cv::Size(im1.cols / 2, im1.rows / 2));
		cv::resize(mask0nocc, mask0nocc, cv::Size(mask0nocc.cols / 2, mask0nocc.rows / 2));
		cv::resize(gt_im, gt_im, cv::Size(gt_im.cols / 2, gt_im.rows / 2));
		write_pfm_file(std::string(folder_path + "/data/" + folders[fold] + "/disp.pfm").c_str(), (float*)gt_im.data, gt_im.cols, gt_im.rows);
		cv::imwrite(folder_path + "/data/" + folders[fold] + "/im0.png", im0);
		cv::imwrite(folder_path + "/data/" + folders[fold] + "/im1.png", im1);
		cv::imwrite(folder_path + "/data/" + folders[fold] + "/mask0nocc.png", mask0nocc);
	}
}

// evaluating
void my_test_test(void) {
	// construct LeNet-5 architecture
	network<hinge_loss, RMSprop> nn;
	network<hinge_loss, RMSprop> nn2;

	nn
		<< convolutional_layer<relu>(11, 11, 3, 1, 64)
		<< convolutional_layer<relu>(9, 9, 3, 64, 64)
		<< convolutional_layer<relu>(7, 7, 3, 64, 64)
		<< convolutional_layer<relu>(5, 5, 3, 64, 64)
		<< convolutional_layer<>(3, 3, 3, 64, 64);


	int cntr = 0;
	auto on_enumerate_minibatch = [&](){
		//disp += minibatch_size; 
		cntr++;
		if (cntr % 10 == 0){
			//std::cout << cntr << std::endl;
		}
	};

	auto on_enumerate_epoch = [&](){
	};

	int epoch_number = 15;
	int minibatch_size = 128;
	int start_fold = 0;
	int start_ep = 0;
	tiny_cnn::float_t initial_l_rate = 0.0001;
	tiny_cnn::float_t epoch_l_decr = 0.75;
	tiny_cnn::float_t cur_l_rate = initial_l_rate*pow(epoch_l_decr, start_ep);
	int need_weights_init = 1;
	std::string folder_path = "..";
	std::ifstream input_weights(folder_path + "/data/LeNet-weights");
	if (input_weights.good()){
		input_weights >> nn;
		need_weights_init = 0;
		std::cout << "weights are given in file" << folder_path << "/data/LeNet-weights" << std::endl;
		input_weights.close();
	}

	nn.optimizer().alpha = cur_l_rate;
	std::vector<std::string> folders = { "Motorcycle" };//{ "Adirondack", "Jadeplant", "Motorcycle", "Piano", "Pipes", "Playroom", "Playtable", "Recycle", "Shelves" };
	//std::vector<std::string> folders = { "Backpack", "Bicycle1", "Cable", "Classroom1", "Couch", "Flowers", "Mask", "Shopvac", "Sticks" };
	//std::vector<std::string> folders = { "Motorcycle" };
	int iter = 0;
	for (int ep = start_ep; ep < epoch_number; ep++){
		for (int fold = start_fold; fold < folders.size(); fold++){
			std::string folder = folders.at(fold);
			iter++;
			std::vector<stereoPair> stereo_p;
			std::cout << "load models..." << std::endl;
			stereo_p.push_back(stereoPair(folder_path + "/data/" + folder + "/im0.png", folder_path + "/data/" + folder + "/im1.png", folder_path + "/data/" + folder + "/disp.pfm", folder_path + "/data/" + folder + "/mask0nocc.png"));
			std::vector<trainPair> pairs = create_images(stereo_p, -1., 1.);
			std::vector<vec_t> array_left, array_right, array_other;
			std::vector<label_t> labels;
			read_images(pairs.at(0), array_left, array_right, array_other, labels);

			std::cout << "start test " << fold << " " << folder  << std::endl;
			nn.test_siam_all(array_left, array_right, array_other, labels);

			std::remove(pairs.at(0).left_dst.c_str());
			std::remove(pairs.at(0).right_dst.c_str());
			std::remove(pairs.at(0).labels.c_str());
		}
	}


	// test and show results
	//nn.test(test_images, test_labels).print_detail(std::cout);

}

