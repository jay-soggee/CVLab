#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

#define ROTATE

Mat resizeImg_NN(Mat img, float scale) {
	int h = img.rows;
	int w = img.cols;

	float posx, posy;
	int sx, sy;

	int h_re = (int)(h * scale);
	int w_re = (int)(h * scale);
	Mat res(h_re, w_re, CV_8UC1);

	for (int yo = 0; yo < h_re; yo++) {
		for (int xo = 0; xo < w_re; xo++) {
			//Backward filling
			posx = ((float)1.0 / scale) * xo;
			posy = ((float)1.0 / scale) * yo;

			//NN
			sx = (int)(posx + 0.5);
			sy = (int)(posy + 0.5);

			res.at<uchar>(yo, xo) = img.at<uchar>(sy, sx);
		}
	}

	return res;
}


Mat resizeImg_Bl(Mat img, float scale) {
	int h = img.rows;
	int w = img.cols;

	float posx, posy;
	float x_rate, y_rate;
	int sx, sy;

	int h_re = (int)(h * scale);
	int w_re = (int)(h * scale);
	Mat res(h_re, w_re, CV_8UC1);

	for (int yo = 0; yo < h_re; yo++) {
		for (int xo = 0; xo < w_re; xo++) {
			//Backward filling
			posx = ((float)1.0 / scale) * xo;
			posy = ((float)1.0 / scale) * yo;

			//Bi-Linear
			x_rate = posx - (int)posx;
			y_rate = posy - (int)posy;
			sx = (int)posx;
			sy = (int)posy;
			uchar p1 = img.at<uchar>(sy, sx);
			uchar p2 = img.at<uchar>(sy, sx + 1);
			uchar pA = (uchar)(p1 * (1 - x_rate) + p2 * x_rate);
			uchar p3 = img.at<uchar>(sy + 1, sx);
			uchar p4 = img.at<uchar>(sy + 1, sx + 1);
			uchar pB = (uchar)(p3 * (1 - x_rate) + p4 * x_rate);
			res.at<uchar>(yo, xo) = (uchar)(pA * (1 - y_rate) + pB * y_rate);
		}
	}

	return res;
}


#define PI (float)3.141592

Mat rotateImg_NN(Mat img, float deg) {
	int h = img.rows;
	int w = img.cols;
	float rad = deg * PI / (float)180.0;
	float r_mat[2][2] = {
		{cos(rad), sin(rad)},
		{-sin(rad), cos(rad)}
	};

	float posx, posy;
	float x_rate, y_rate;
	int sx, sy, sx0, sy0, sx1, sy1;

	int h_re = h; //(int)(h * cos(rad) + w * sin(rad) + 0.5);
	int w_re = w; //(int)(w * cos(rad) + h * sin(rad) + 0.5);
	Mat res(h_re, w_re, CV_8UC1);

	for (int yo = 0; yo < h_re; yo++) {
		for (int xo = 0; xo < w_re; xo++) {
			posx = (xo - w_re / 2) * r_mat[0][0] + (yo - h_re / 2) * r_mat[0][1] + w_re / 2;
			posy = (xo - w_re / 2) * r_mat[1][0] + (yo - h_re / 2) * r_mat[1][1] + h_re / 2;

			//NN
			sx = (int)(posx + 0.5);
			sy = (int)(posy + 0.5);

			if (sx >= 0 && sx < w - 1 && sy >= 0 && sy < h - 1)
				res.at<uchar>(yo, xo) = img.at<uchar>(sy, sx);
			else
				res.at<uchar>(yo, xo) = 255;
		}
	}

	return res;
}


Mat rotateImg_Bl(Mat img, float deg) {
	int h = img.rows;
	int w = img.cols;
	float rad = deg * PI / (float)180.0;
	float r_mat[2][2] = { 
		{cos(rad), sin(rad)}, 
		{-sin(rad), cos(rad)} 
	};

	float posx, posy;
	float x_rate, y_rate;
	int sx, sy, sx0, sy0, sx1, sy1;

	int h_re = h; //(int)(h * cos(rad) + w * sin(rad) + 0.5);
	int w_re = w; //(int)(w * cos(rad) + h * sin(rad) + 0.5);
	Mat res(h_re, w_re, CV_8UC1);

	for (int yo = 0; yo < h_re; yo++) {
		for (int xo = 0; xo < w_re; xo++) {
			posx = (xo - w_re / 2) * r_mat[0][0] + (yo - h_re / 2) * r_mat[0][1] + w_re / 2;
			posy = (xo - w_re / 2) * r_mat[1][0] + (yo - h_re / 2) * r_mat[1][1] + h_re / 2;

			//Bi-Linear
			x_rate = posx - (int)posx;
			y_rate = posy - (int)posy;
			sx = (int)posx;
			sy = (int)posy;
			sx0 = sx;
			sy0 = sy;
			sx1 = sx0 + 1;
			sy1 = sy0 + 1;

			uchar p1 = img.at<uchar>(sy0, sx0);
			uchar p2 = img.at<uchar>(sy0, sx1);
			uchar pA = (uchar)(p1 * (1 - x_rate) + p2 * x_rate);
			uchar p3 = img.at<uchar>(sy1, sx0);
			uchar p4 = img.at<uchar>(sy1, sx1);
			uchar pB = (uchar)(p3 * (1 - x_rate) + p4 * x_rate);
			
			if (sx >= 0 && sx < w - 1 && sy >= 0 && sy < h - 1)
				res.at<uchar>(yo, xo) = (uchar)(pA * (1 - y_rate) + pB * y_rate);
			else
				res.at<uchar>(yo, xo) = 255;
		}
	}

	return res;
}


int main()
{
	Mat img = imread("Lenna.png", IMREAD_GRAYSCALE);

#ifndef ROTATE
	float scale;
	printf("scale << ");
	scanf("%f", &scale);

	Mat res_resizeNN;
	res_resizeNN = resizeImg_NN(img, scale);

	Mat res_resizeBl;
	res_resizeBl = resizeImg_Bl(img, scale);
#else
	double rotate;
	printf("\nrotate angle << ");
	scanf("%Lf", &rotate);

	Mat res_rotateNN;
	res_rotateNN = rotateImg_NN(img, rotate);

	Mat res_rotateBl;
	res_rotateBl = rotateImg_Bl(img, rotate);
#endif

	imshow("Original", img);
#ifndef ROTATE
	imshow("resized-NN", res_resizeNN);
	imshow("resized-Bilinear", res_resizeBl);
#else
	imshow("rotated-NN", res_rotateNN);
	imshow("rotated-Bilinear", res_rotateBl);
#endif
	waitKey(0);
	return 0;
}
