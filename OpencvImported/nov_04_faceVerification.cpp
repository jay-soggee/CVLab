#include "oct_28_faceRecog/oct_28_HOGcompare.h"
#include <opencv2/video/tracking.hpp>
#include <vector>

using namespace cv;

#define OUT
#define DEBUG
#define DEBUG0

#define WIDTH 160
#define HIGHT 120

void LBP(Mat img, OUT Mat* res) {

	const int P = 8;
	const int R = 1;
	int h, w;
	h = img.rows - 2;
	w = img.cols - 2;

	int grad_max = 0;
	for (int y = 1; y < h + 1; y++)
		for (int x = 1; x < w + 1; x++) {
			int v = 0;
			
			// no interpolation
			int ix[P];
			int iy[P];
			ix[0] = x;		iy[0] = y - 1;
			ix[1] = x + 1;	iy[1] = y - 1;
			ix[2] = x + 1;	iy[2] = y;
			ix[3] = x + 1;	iy[3] = y + 1;
			ix[4] = x;		iy[4] = y + 1;
			ix[5] = x - 1;	iy[5] = y + 1;
			ix[6] = x - 1;	iy[6] = y;
			ix[7] = x - 1;	iy[7] = y - 1;

			for (int i = 0; i < 8; i++)
				if (img.at<uchar>(y, x) > img.at<uchar>(iy[i], ix[i])) v |= (1 << i);

			res->at<uchar>(y - 1, x - 1) = v;
		}
}

double similarity(Mat a, Mat b, int div) {

	int h = a.cols; // size between a and b must be the same
	int w = a.rows;
	int hist_ab[256 * 2];
	double sim = 0;
	int block_size = h / div;
	int block_count = (div * 2 - 1) * (w * 2 / block_size - 1);
	for (int bi = 0; bi < h - block_size / 2; bi = bi + block_size / 2)
		for (int bj = 0; bj < h - block_size / 2; bj = bj + block_size / 2) {
			double norm_a = 0.000000001, norm_b = 0.000000001, prod = 0.000000001;
			for (int i = 0; i < 256 * 2; i++)
				hist_ab[i] = 0;
			for (int i = 0; i < block_size; i++)
				for (int j = 0; j < block_size; j++) {
					int y = (bi + i);
					int x = (bj + j);
					hist_ab[a.at<uchar>(y, x)]++;
					hist_ab[b.at<uchar>(y, x) + 256]++;
				}
			for (int i = 0; i < 256; i++) {
				norm_a += hist_ab[i] * hist_ab[i];
				norm_b += hist_ab[i + 256] * hist_ab[i + 256];
				prod += hist_ab[i] * hist_ab[i + 256];
			}
			norm_a = sqrt(norm_a);
			norm_b = sqrt(norm_b);
			sim += prod / norm_a / norm_b;
		}
	sim /= block_count;
	return sim;
}



#ifdef DEBUG

int main() {

	const double CROP_SIM = 0.8;
	const int REF_H = 36;
	const int REF_W = 36;
	const int DIVISION = 5;

	VideoCapture capture(0);
	Mat frame, img;
	Mat ref_face(REF_H, REF_H, CV_8UC1);
	Mat ref_faceLBP(REF_H - 2, REF_H - 2, CV_8UC1);
	Mat tar_face(REF_H, REF_H, CV_8UC1);
	Mat tar_faceLBP(REF_H - 2, REF_H - 2, CV_8UC1);
	int faceCenter_x;
	int faceCenter_y;
	Point faceRactPt_a, faceRactPt_b;
	Rect crop;
	double sim;
	int key;

	do {
		capture >> frame;

		resize(frame, img, Size(WIDTH, HIGHT));
		cvtColor(img, frame, COLOR_BGR2GRAY);

		if (!faceRecog(frame, &faceCenter_x, &faceCenter_y)) {
			faceRactPt_a = { faceCenter_x - REF_W /2 ,faceCenter_y - REF_H/2 };
			faceRactPt_b = { faceCenter_x + REF_W /2 ,faceCenter_y + REF_H/2 };
			crop = { faceRactPt_a, faceRactPt_b };
			ref_face = frame(crop);
		}
#ifdef DEBUG0
		rectangle(img, faceRactPt_a, faceRactPt_b, { 255, 0, 0 }, 1, 8, 0);
		imshow("camera", img);
		imshow("captured face", ref_face);
#endif
		key = waitKey(20);
	} while (key == -1);

	do {
		capture >> frame;
		resize(frame, img, Size(WIDTH, HIGHT));
		cvtColor(img, frame, COLOR_BGR2GRAY);

		if (!faceRecog(frame, &faceCenter_x, &faceCenter_y)) {
			faceRactPt_a = { faceCenter_x - REF_W / 2 ,faceCenter_y - REF_H / 2 };
			faceRactPt_b = { faceCenter_x + REF_W / 2 ,faceCenter_y + REF_H / 2 };
			crop = { faceRactPt_a, faceRactPt_b };
			tar_face = frame(crop);

			LBP(ref_face, &ref_faceLBP);
			LBP(tar_face, &tar_faceLBP);

			if ((sim = similarity(ref_faceLBP, tar_faceLBP, DIVISION)) > CROP_SIM) {
				rectangle(img, faceRactPt_a, faceRactPt_b, { 0, 255, 0 }, 1, 8, 0);
			}
			else {
				rectangle(img, faceRactPt_a, faceRactPt_b, { 0, 0, 255 }, 1, 8, 0);
			}
#ifdef DEBUG0
			printf("%Lf\n", sim);
#endif
		}
		imshow("result", img);
		key = waitKey(20);
	} while (key == -1);

	waitKey(0);
	return 0;
}

#endif