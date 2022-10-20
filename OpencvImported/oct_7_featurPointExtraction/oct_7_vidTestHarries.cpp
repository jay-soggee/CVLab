#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "oct_7_LL_corner.h"

using namespace cv;

#define DEBUG



#define PI		3.141592
#define W_SIZE  15			// window size = must be odd num
#define K		0.06		// must be ranged from 0.04 ~ 0.06
#define G_SIZE	W_SIZE		// gaussian window size
#define G_VARI	(double)1
#define N_SIZE  W_SIZE		// non-maximum suppression size
#define B_SIZE  16			// can be even num, maximum W_SIZE + 1.
#define SIM_MIN 0.4			// value for clipping the simmilarity


double FLTR_G[G_SIZE][G_SIZE];

node* TAIL;


inline void gaussian() {
	const double coeff = 1.0 / sqrt(2.0 * PI * G_VARI);
	double tot = 0;

	for (int y = -(G_SIZE / 2); y <= (G_SIZE / 2); y++)
		for (int x = -(G_SIZE / 2); x <= (G_SIZE / 2); x++) {
			int norm = x * x + y * y;
			FLTR_G[y + (G_SIZE / 2)][x + (G_SIZE / 2)]
				= coeff * exp(-(double)norm / (2 * G_VARI));
			tot += FLTR_G[y + (G_SIZE / 2)][x + (G_SIZE / 2)];
		}

	for (int y = 0; y < G_SIZE; y++) {
		for (int x = 0; x < G_SIZE; x++)
			FLTR_G[y][x] /= tot;
	}
}


node* cornerOf(Mat img, float** gx, float** gy) {

	gaussian();
	node* head = initList(TAIL); // linked list for the result

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
		}

	/* compute R */
	node* cornr = head;
	for (int i = 0; i < h - W_SIZE + 1; i++)
		for (int j = 0; j < w - W_SIZE + 1; j++) {
			double IxIx = 0, IxIy = 0, IyIy = 0;
			double R;
			for (int fi = 0; fi < W_SIZE; fi++)
				for (int fj = 0; fj < W_SIZE; fj++) {
					float Ix = *(*gx + (i + fi) * w + j + fj);
					float Iy = *(*gy + (i + fi) * w + j + fj);
					IxIx += Ix * Ix * FLTR_G[fi][fj];
					IxIy += Ix * Iy * FLTR_G[fi][fj];
					IyIy += Iy * Iy * FLTR_G[fi][fj];
				}
			R = IxIx * IyIy - IxIy * IxIy - K * (IxIx + IyIy) * (IxIx + IyIy);
			if (R > 0.001) {
				cornr = insertAfter(cornr, j + (W_SIZE / 2), i + (W_SIZE / 2), R);
			}
		}

	return head;
}


void suppress(node* h) {
	for (node* t = h->next; t != TAIL; t = t->next) {
		double Rmax = 0;
		for (node* a = h; a->next != TAIL; a = a->next) {
			if (abs(a->next->x - t->x) < (N_SIZE / 2) && abs(a->next->y - t->y) < (N_SIZE / 2)) {
				if (a->next->R >= Rmax) {
					Rmax = a->next->R;
				}
				else
					deleteAfter(a);
			}
		}
	}
}


int drawCorner(Mat img_gray, Mat img) {

	int h = img_gray.rows - 2;
	int w = img_gray.cols - 2;
	float* grad_x, * grad_y;
	if (!(grad_x = (float*)malloc(sizeof(int) * h * w))) return -1;
	if (!(grad_y = (float*)malloc(sizeof(int) * h * w))) return -1;
	node* head = cornerOf(img_gray, &grad_x, &grad_y);
	suppress(head);

	for (node* t = head->next; t != TAIL; t = t->next) {
		Scalar c;
		Point pCenter;
		int radius = 9;
		pCenter.x = t->x + 1;
		pCenter.y = t->y + 1;
		c.val[0] = 0;
		c.val[1] = 255;
		c.val[2] = 0;
		circle(img, pCenter, radius, c, 1, 11, 0); // red circle
	}

	while (deleteAfter(head));
	free(grad_x);
	free(grad_y);

	static int dbg_tail = 0;
	dbg_tail++;
	if (!TAIL) printf("!");

	return 0;
}




void main()
{
	VideoCapture capture(0);
	Mat frame;
	int h = frame.rows;
	int w = frame.cols;

	TAIL = (node*)calloc(1, sizeof(node));
	if (!capture.isOpened()) {
		printf("Couldn¡¯t open the web camera¡¦\n");
		return;
	}
	while (true) {
		Mat frame_gray(h, w, CV_8UC1);
		capture >> frame;

		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		drawCorner(frame_gray, frame);








		imshow("Video", frame);
		if (waitKey(30) >= 0) break;
	}
}



