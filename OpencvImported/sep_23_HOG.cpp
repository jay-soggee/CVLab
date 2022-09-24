#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

#define PI (float)3.141592

//#define DEBUG01

/* compute convolution w/ some filter to get 2-d gradient */
int gradiant(Mat img, int** gx, int** gy) {

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
	h = img.rows;
	w = img.cols;

	for (int i = 0; i < h - 2; i++)
		for (int j = 0; j < w - 2; j++) {
			int fx = 0, fy = 0; // x, y-coordinate gradient
			for (int fi = 0; fi < 3; fi++)
				for (int fj = 0; fj < 3; fj++) {
					fx += img.at<uchar>(i + fi, j + fj) * *(xfltr + fi * 3 + fj);
					fy += img.at<uchar>(i + fi, j + fj) * *(yfltr + fi * 3 + fj);
				}
#ifdef DEBUG
			printf("(%4d, %4d)  ", fx, fy);
#endif
			*(*gx + i * (w - 2) + j) = fx;
			*(*gy + i * (w - 2) + j) = fy;
		}

	return 0;
}


/* compute the Orientation Histogram */
int getOH(int h, int w, int* gx, int* gy, double** hist) {

	for (int i = 0; i < 9; i++)
		*(*hist + i) = 0;

	double tot = 0;
	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++) {
			int fx = *(gx + i * w + j);
			int fy = *(gy + i * w + j);

			// compute the L2 magnitude
			float mag;
			mag = sqrt(fx * fx + fy * fy);
#ifdef DEBUG
			printf("%Lf ", mag);
#endif
			// compute the phase
			float dir, deg;
			dir = atan2(fx, fy);
			deg = dir * 180.0 / PI;
			if (deg < 0) deg += 180.0;

			if ((int)(deg / 20.2) > 9) printf("!");

			// accumulate in histogram
			*(*hist + (int)(deg / 20.0)) += mag;
			tot += mag * mag;
		}

	// normalize
	tot = sqrt(tot + 0.001);
	for (int i = 0; i < 9; i++)
		*(*hist + i) /= tot;

	return 0;
}



/* Histograms of Oriented Gradients */
int HOG(Mat img, double** hist) {
	int h = img.rows;
	int w = img.cols;

	int* img_grad_x, * img_grad_y; // result of gradiant
	if (!(img_grad_x = (int*)malloc(sizeof(int) * (h - 2) * (w - 2)))) return -1;
	if (!(img_grad_y = (int*)malloc(sizeof(int) * (h - 2) * (w - 2)))) return -1;

	gradiant(img, &img_grad_x, &img_grad_y);

#ifdef DEBUG0
	Mat res_grad_x(h - 2, w - 2, CV_8UC1);
	Mat res_grad_y(h - 2, w - 2, CV_8UC1);
	for (int i = 0; i < h - 2; i++)
		for (int j = 0; j < w - 2; j++) {
			res_grad_x.at<uchar>(i, j) = *(img_grad_x + i * (w - 2) + j) / 32;
			res_grad_y.at<uchar>(i, j) = *(img_grad_y + i * (w - 2) + j) / 32;
		}
	imshow("original", img);
	imshow("grad x", res_grad_x);
	imshow("grad y", res_grad_y);
	waitKey(0);
#endif

	getOH(h - 2, w - 2, img_grad_x, img_grad_y, hist);

#ifdef DEBUG01
	for (int i = 0; i < 9; i++)
		printf("%Lf ", *(*hist + i));
	printf("\n");
#endif

	free(img_grad_x);
	free(img_grad_y);
	return 0;
}



int main() {
	
	Mat img_gray0 = imread("images/lecture3.bmp", IMREAD_GRAYSCALE);
	Mat img_gray1 = imread("images/compare2.bmp", IMREAD_GRAYSCALE);
	int h = img_gray0.rows;
	int w = img_gray0.cols;
	int h1 = img_gray1.rows;
	int w1 = img_gray1.cols;
	if (abs(h - h1) > 8 || abs(w - w1) > 8) {
		printf("invalid image size to compare\n");
		return -1;
	}
	const int h_tile = 16;
	const int w_tile = 16;
	Mat img_tile0(h_tile, w_tile, CV_8UC1);
	Mat img_tile1(h_tile, w_tile, CV_8UC1);

	double* hist0, * hist1;
	if (!(hist0 = (double*)calloc(9, sizeof(double)))) return -1;
	if (!(hist1 = (double*)calloc(9, sizeof(double)))) return -1;

	double result = 0;

	for (int i = 0; i < h - 8; i += 8) {
		printf("\n");
		for (int j = 0; j < w - 8; j += 8) {

			// get 16 by 16 block image
			for (int ti = 0; ti < h_tile; ti++)
				for (int tj = 0; tj < w_tile; tj++) {
					img_tile0.at<uchar>(ti, tj) = img_gray0.at<uchar>(i + ti, j + tj);
					img_tile1.at<uchar>(ti, tj) = img_gray1.at<uchar>(i + ti, j + tj);
				}

			// compute each HOG
			HOG(img_tile0, &hist0);
			HOG(img_tile1, &hist1);

			// compute Euclidean distance
			double dist = 0;
			for (int hi = 0; hi < 9; hi++) 
				dist += (*(hist0 + hi) - *(hist1 + hi)) * (*(hist0 + hi) - *(hist1 + hi));
			dist = sqrt(dist);

			printf("%10Lf", dist);
			result += dist;
		}
		printf("\n\n");
	}
	free(hist0);
	free(hist1);
	
	printf("%Lf", result);

#ifdef DEBUG
	for (int ti = 0; ti < h_tile; ti++)
		for (int tj = 0; tj < w_tile; tj++) {
			img_tile.at<uchar>(ti, tj) = img_gray.at<uchar>(ti, tj);
			printf("%d ", img_gray.at<uchar>(ti, tj));
		}
	printf("\n\n");

	HOG(img_tile);
#endif



	return 0;
}