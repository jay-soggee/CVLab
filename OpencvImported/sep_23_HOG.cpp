#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

#define PI (float)3.141592

#define DEBUG

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
					fx += img.at<uchar>(i - 1 + fi, j - 1 + fj) * *(xfltr + fi * 3 + fj);
					fy += img.at<uchar>(i - 1 + fi, j - 1 + fj) * *(yfltr + fi * 3 + fj);
				}
			*(*gx + i * (w - 2) + j) = fx;
			*(*gy + i * (w - 2) + j) = fy;
		}

	return 0;
}


/* compute the Orientation Histogram */
int getOH(int h, int w, int* gx, int* gy, double** hist) {

	double tot = 0;
	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++) {
			int fx = *(gx + i * w + j);
			int fy = *(gy + i * w + j);

			// compute the L2 magnitude
			float mag;
			mag = sqrt(fx * fx + fy * fy);

			// compute the phase
			float dir, deg;
			dir = atan2(fx, fy);
			deg = dir * 180.0 / PI;
			if (deg < 0) deg += 180.0;

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
int HOG(Mat img) {
	int h = img.rows;
	int w = img.cols;

	int* img_grad_x, * img_grad_y; // result of gradiant
	if (!(img_grad_x = (int*)malloc(sizeof(int) * (h - 2) * (w - 2)))) return -1;
	if (!(img_grad_y = (int*)malloc(sizeof(int) * (h - 2) * (w - 2)))) return -1;

	gradiant(img, &img_grad_x, &img_grad_y);

#ifdef DEBUG
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

	double* img_hist; // result of orientation histogram (0 : 20 : 180 degree)
	if (!(img_hist = (double*)calloc(9, sizeof(double)))) return -1;
	getOH(h, w, img_grad_x, img_grad_y, &img_hist);

#ifdef DEBUG
	for (int i = 0; i < 9; i++)
		printf("%Lf ", img_hist[i]);
	printf("\n");
#endif

	return 0;
}



int main() {

	Mat img_gray = imread("images/lecture3.bmp", IMREAD_GRAYSCALE);
	int h = img_gray.rows;
	int w = img_gray.cols;
	const int h_tile = 16;
	const int w_tile = 16;
	Mat img_tile(h_tile, w_tile, CV_8UC1);

	/*for (int i = 0; i < h; i += 8)
		for (int j = 0; j < w; j += 8) {
			for (int ti = 0; ti < h_tile; ti++)
				for (int tj = 0; tj < w_tile; tj++)
					img_tile.at<uchar>(ti, tj) = img_gray.at<uchar>(i + ti, j + tj);
			HOG(img_tile);
		}*/
	for (int ti = 0; ti < h_tile; ti++)
		for (int tj = 0; tj < w_tile; tj++) {
			img_tile.at<uchar>(ti, tj) = img_gray.at<uchar>(ti, tj);
			printf("%d ", img_gray.at<uchar>(ti, tj));
		}
	HOG(img_tile);




	return 0;
}