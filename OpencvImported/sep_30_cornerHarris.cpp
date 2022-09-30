#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

#define G_SIZE	15 // must be odd num
#define G_VARI	(double)1
#define PI		(double)3.141592

#define DEBUG

double fltr_g[G_SIZE][G_SIZE];

inline void gaussian() {
	const double coeff = 1.0 / sqrt(2.0 * PI * G_VARI);

	for (int y = -(G_SIZE / 2); y <= (G_SIZE / 2); y++) {
		for (int x = -(G_SIZE / 2); x <= (G_SIZE / 2); x++) {
			int norm = x * x + y * y;
			fltr_g[y + (G_SIZE / 2)][x + (G_SIZE / 2)]
				= coeff * exp(-(double)norm / (2 * G_VARI));
#ifdef DEBUG_G
			printf("%10Lf", res[x + (G_SIZE / 2)][y + (G_SIZE / 2)]);
#endif
		}
#ifdef DEBUG_G
		printf("\n");
#endif
	}
}


void LPF(Mat img) {

	int h, w;
	h = img.rows;
	w = img.cols;

	for (int i = 0; i <= h; i++)
		for (int j = 0; j <= w; j++) {
			double lp = 0;
			for (int fi = 0; fi < G_SIZE; fi++)
				for (int fj = 0; fj < G_SIZE; fj++) {
					int x = j + fj - (G_SIZE / 2);
					int y = i + fi - (G_SIZE / 2);
					x = x < 0 ? -x : x >= w ? 2 * w - x : x;
					y = y < 0 ? -y : y >= w ? 2 * w - y : y;
					
					lp += img.at<uchar>(y, x) * fltr_g[fi][fj];
				}
#ifdef DEBUG
			static int f = 0;
			if (f++ < h * w / 5)
				printf("%10Lf  ", lp / G_SIZE / G_SIZE);
#endif
		}
}

// copy gradiant
// compute R
/*
	1. compute Structure tensor M
	2. compute R = det + k * tr * tr (k = 0.04~0.06)
	3. is_corner = R > 100 ? 1 : 0;
*/



int main() {

	gaussian();

	Mat img_gray = imread("images/Lenna.png", IMREAD_GRAYSCALE);
	int h = img_gray.rows;
	int w = img_gray.cols;

#ifdef DEBUG_G
	double* g_fltr = gaussian();
#endif

#ifdef DEBUG
	LPF(img_gray);
#endif

	return 0;
}