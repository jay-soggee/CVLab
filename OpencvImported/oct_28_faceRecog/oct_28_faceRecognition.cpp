#include "HOGcompare.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

#define USE_MMNORM
#define DEBUG
#define DEBUG1
#define OUT

#define PI 3.141592

void gradient(Mat img, float** gx, float** gy) {

	/* conv */
	int xfltr[9] = {
		-1, 0, 1,
		-2, 0, 2,
		-1, 0, 1
	};
	int yfltr[9] = {
		-1, -2, -1,
		 0,  0,  0,
		 1,  2,  1
	};
	int h, w;
	h = img.rows - 2;
	w = img.cols - 2;

	int grad_max = 0;
	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++) {
			int fx = 0, fy = 0;
			for (int fi = 0; fi < 3; fi++)
				for (int fj = 0; fj < 3; fj++) {
					fx += img.at<uchar>(i + fi, j + fj) * *(xfltr + fi * 3 + fj);
					fy += img.at<uchar>(i + fi, j + fj) * *(yfltr + fi * 3 + fj);
				}
			*(*gx + i * w + j) = fx;
			*(*gy + i * w + j) = fy;

#ifdef USE_MMNORM
			// min-max normalization
			if (abs(fx) > grad_max)
				grad_max = abs(fx);
			else if (abs(fy) > grad_max)
				grad_max = abs(fy);
		}
	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++) {
			*(*gx + i * w + j) /= grad_max;
			*(*gy + i * w + j) /= grad_max;
#endif
		}

#ifdef DEBUG0
	Mat gximg(h, w, CV_8UC1);
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
			gximg.at<uchar>(y, x) = (int)(*(*gx + y * w + x) * 255);
	imshow("orig", img);
	imshow("asdf", gximg);
	waitKey(0);
#endif
}

/*
* @brief get oriented Histogram from gradient x and y.
* @param gradiant x
* @param gradiant y
* @param image height
* @param image width
* @param the number of division for the target image
* @param output histogram, the lenth must be 9 * div
*/
void getOH(float* gx[], float* gy[], 
	int h, int w, 
	const int div,
	OUT double** hist) 
{

	for (int i = 0; i < 9 * div * div; i++)
		*(*hist + i) = 0;

	for (int ti = 0; ti < div; ti++)
		for (int tj = 0; tj < div; tj++)
			for (int i = 0; i < h / div; i++)
				for (int j = 0; j < w / div; j++) {
					float fx = *(gx[i + ti * h / div] + j + tj * w / div);
					float fy = *(gy[i + ti * h / div] + j + tj * w / div);

					float mag;
					mag = sqrt(fx * fx + fy * fy);

					float dir, deg;
					dir = atan2(fx, fy);
					deg = dir * 180.0 / PI;
					if (deg < 0) deg += 180.0;

					*(*hist + ti * 9 * div + tj * 9 + (int)(deg / 20.0)) += mag;
				}
	double norm = 0.000001;
	for (int i = 0; i < 9 * div * div; i++) {
		norm += *(*hist + i) * *(*hist + i);
	}
	norm = sqrt(norm);
	for (int i = 0; i < 9 * div * div; i++) {
		*(*hist + i) /= norm;
	}
}

#ifdef DEBUG

int main() {

	Mat img_ref = imread("./images/face_ref.bmp", IMREAD_GRAYSCALE);
	Mat img_tar_color = imread("./images/face_tar.bmp");
	Mat img_tar;
	cvtColor(img_tar_color, img_tar, COLOR_BGR2GRAY);
	int h_ref = img_ref.rows;
	int w_ref = img_ref.cols;
	int h_tar = img_tar.rows;
	int w_tar = img_tar.cols;

	// compute each gradiant
	h_ref = h_ref - 2;
	w_ref = w_ref - 2;
	h_tar = h_tar - 2;
	w_tar = w_tar - 2;
	float* ref_grad_x, * ref_grad_y;
	if (!(ref_grad_x = (float*)malloc(sizeof(float) * h_ref * w_ref))) return -1;
	if (!(ref_grad_y = (float*)malloc(sizeof(float) * h_ref * w_ref))) return -1;
	float* tar_grad_x, * tar_grad_y;
	if (!(tar_grad_x = (float*)malloc(sizeof(float) * h_tar * w_tar))) return -1;
	if (!(tar_grad_y = (float*)malloc(sizeof(float) * h_tar * w_tar))) return -1;

	gradient(img_ref, &ref_grad_x, &ref_grad_y);
	gradient(img_tar, &tar_grad_x, &tar_grad_y);

	// compute histogram of reference image
	const int division = 2;
	float** blck_ref_grad_x, ** blck_ref_grad_y;
	if (!(blck_ref_grad_x = (float**)malloc(sizeof(float*) * h_ref))) return -1;
	if (!(blck_ref_grad_y = (float**)malloc(sizeof(float*) * h_ref))) return -1;
	for (int ti = 0; ti < h_ref; ti++) {
		*(blck_ref_grad_x + ti) = ref_grad_x + ti * w_ref;
		*(blck_ref_grad_y + ti) = ref_grad_y + ti * w_ref;
	}
	double* ref_hist;
	if (!(ref_hist = (double*)calloc(9 * division * division, sizeof(double)))) return -1;
	getOH(blck_ref_grad_x, blck_ref_grad_y, h_ref, w_ref, division, &ref_hist);
	free(blck_ref_grad_x);
	free(blck_ref_grad_y);
	free(ref_grad_x);
	free(ref_grad_y);

	double ref_norm = 0;
	for (int i = 0; i < 9 * division * division; i++)
		ref_norm += ref_hist[i] * ref_hist[i];
	ref_norm = sqrtl(ref_norm);

	// compute histogram of target image and simularity between each other.
	for (int i = h_ref / 2; i <= h_tar - h_ref / 2; i++) 
		for (int j = w_ref / 2; j <= w_tar - w_ref / 2; j++) {
			// crop the target image
			float** crop_tar_grad_x, ** crop_tar_grad_y;
			if (!(crop_tar_grad_x = (float**)malloc(sizeof(float*) * h_ref))) return -1;
			if (!(crop_tar_grad_y = (float**)malloc(sizeof(float*) * h_ref))) return -1;
			for (int ti = 0; ti < h_ref; ti++) {
				*(crop_tar_grad_x + ti) = tar_grad_x + (ti + i - h_ref / 2) * w_tar + (j - w_ref / 2);
				*(crop_tar_grad_y + ti) = tar_grad_y + (ti + i - h_ref / 2) * w_tar + (j - w_ref / 2);
			}

#ifdef DEBUG0
			Mat croped(h_ref, w_ref, CV_8UC1);
			for (int y = 0; y < h_ref; y++)
				for (int x = 0; x < h_ref; x++)
					croped.at<uchar>(y, x) = (int)(crop_tar_grad_x[y][x] * 255);
			imshow("asdf", croped);
			waitKey(0);
#endif
			// get HOG from croped image
			double* tar_hist;
			if (!(tar_hist = (double*)calloc(9 * division * division, sizeof(double)))) return -1;
			getOH(crop_tar_grad_x, crop_tar_grad_y, h_ref, w_ref, division, &tar_hist);

			// compare between ref. and target w/ cosine similarity
			double tar_norm = 0, product = 0, sim;
			for (int hi = 0; hi < 9 * division * division; hi++) {
				tar_norm	+= tar_hist[hi] * tar_hist[hi];
				product		+= ref_hist[hi] * tar_hist[hi];
			}
			sim = product / ref_norm / sqrtl(tar_norm);

			if (sim > 0.89) img_tar_color.at<Vec3b>(i + 1, j + 1)[1] = (uchar)(sim * 256);
			free(tar_hist);
			free(crop_tar_grad_y);
			free(crop_tar_grad_x);
		}

	imshow("result", img_tar_color);
	waitKey(0);

	return 0;
}

#endif