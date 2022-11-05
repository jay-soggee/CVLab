#pragma once
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

#define OUT
int faceRecog(Mat tar, OUT int* face_x, int* face_y);