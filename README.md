This is MGM code adapted to the OpenCV that was originally written by Gabriele Facciolo, 
Carlo de Franchis, and Enric Meinhardt and extended (e.g. convolutional neural networks' method was integrated and some other methods as well) here for some new methods by Sergei Ovchinnikov.
See the paper improvement_of_semiglobal_stereomatching_algo_ovchinnikov.pdf for details.
Original MGM's code is located at https://github.com/gfacciol/mgm


-MGM folder contains MGM's OpenCV implementation inside with some new occlusion handling methods and new cost function methods. Entry point is mgm.cc (call it without params to see help)


-Convolution Neural Network folder contains modified tiny-cnn's version for utilizing new MGM's cost function called 'CNN' based on siamese nets and some new CNN's loss function (entry point is example/main.cpp for test and training)
