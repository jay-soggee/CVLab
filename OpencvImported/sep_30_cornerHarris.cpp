#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "sep_30_LL_corner.h"

using namespace cv;

#define PI		3.141592
#define W_SIZE  15			// window size = must be odd num
#define K		0.06		// must be ranged from 0.04 ~ 0.06
#define G_SIZE	W_SIZE		// gaussian window size
#define G_VARI	(double)1
#define N_SIZE  W_SIZE		// non-maximum suppression size
#define B_SIZE  16			// can be even num, maximum W_SIZE + 1.
#define SIM_MIN 0.4			// value for clipping the simmilarity

#define DEBUG_EX


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
#ifdef DEBUG_C
					if ((j + fj) > w || (i + fi) > h)
						printf("!");
#endif
					IxIx += Ix * Ix * FLTR_G[fi][fj];
					IxIy += Ix * Iy * FLTR_G[fi][fj];
					IyIy += Iy * Iy * FLTR_G[fi][fj];
				}
			R = IxIx * IyIy - IxIy * IxIy - K * (IxIx + IyIy) * (IxIx + IyIy);
#ifdef DEBUG_C
			static int flag = 0;
			const int off = 248;
			if (flag++ > off && flag < off + 50)
				printf("%30Lf", R);
			if (flag == off + 50)
				printf("\n");
#endif
			if (R > 0.001) {
#ifdef DEBUG_C
				Scalar c;
				Point pCenter;
				int radius = 5;
				pCenter.x = j + 8;
				pCenter.y = i + 8;
				c.val[0] = 0;
				c.val[1] = 0;
				c.val[2] = 255;
				circle(img, pCenter, radius, c, 1, 8, 0); // red circle
#endif
				cornr = insertAfter(cornr, j + (W_SIZE / 2), i + (W_SIZE / 2), R);
			}
		}

#ifdef DEBUG_C
	imshow("corners", img);
	waitKey(0);
#endif

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


void getOH(int h, int w, float* gx[], float* gy[], double** hist) {

	for (int i = 0; i < 36; i++)
		*(*hist + i) = 0;

	for (int ti = 0; ti < 2; ti++)
		for (int tj = 0; tj < 2; tj++)
			for (int i = 0; i < h / 2; i++)
				for (int j = 0; j < w / 2; j++) {
					float fx = *(gx[i + ti * h / 2] + j + tj * w / 2);
					float fy = *(gy[i + ti * h / 2] + j + tj * w / 2);

					float mag;
					mag = sqrt(fx * fx + fy * fy);

					float dir, deg;
					dir = atan2(fx, fy);
					deg = dir * 180.0 / PI;
					if (deg < 0) deg += 180.0;

					*(*hist + ti * 18 + tj * 9 + (int)(deg / 20.0)) += mag;
				}
	double norm = 0.000001;
	for (int i = 0; i < 36; i++) {
		norm += *(*hist + i) * *(*hist + i);
	}
	norm = sqrt(norm);
	for (int i = 0; i < 36; i++) {
		*(*hist + i) /= norm;
	}
}

void HOGdescript(int h, int w, float* img_grad_x, float* img_grad_y, node* head) {

	for (node* t = head->next; t != TAIL; t = t->next) {
		int x = t->x < (B_SIZE / 2) ? (B_SIZE / 2) : t->x;
		int y = t->y < (B_SIZE / 2) ? (B_SIZE / 2) : t->y;
		float* block_gx[B_SIZE];
		float* block_gy[B_SIZE];
		for (int ti = - (B_SIZE / 2); ti < (B_SIZE / 2); ti++) {
#ifdef DEBUG_HOG
			for (int tj = -(B_SIZE / 2); tj < (B_SIZE / 2); tj++) {
				if ((x + tj) < 0 || (x + tj) >= h || (y + ti) < 0 || (y + ti) >= w) printf("!");
			}
#endif		
			block_gx[ti + (B_SIZE / 2)] = img_grad_x + (y + ti) * w + x - (B_SIZE / 2);
			block_gy[ti + (B_SIZE / 2)] = img_grad_y + (y + ti) * w + x - (B_SIZE / 2);
		}
		getOH(B_SIZE, B_SIZE, block_gx, block_gy, &(t->hist));
	}
}


void similarity(node* hr, node* ht) {

	for (node* r = hr->next; r != TAIL; r = r->next) {
		node* sim_node = NULL;
		double min = DBL_MAX;
		for (node* t = ht->next; t != TAIL; t = t->next) {
			double dist = 0;
			for (int hi = 0; hi < 36; hi++) {
				dist += (*(r->hist + hi) - *(t->hist + hi)) * (*(r->hist + hi) - *(t->hist + hi));
			}
			dist = sqrt(dist);
			if (min > dist) {
				min = dist;
				sim_node = t;
#ifdef DEBUG_SIM
				printf("%15Lf", dist);
#endif
			}
			if (dist < SIM_MIN)
				r->sim_to = sim_node;
		}
#ifdef DEBUG_SIM
		printf("\n\n");
#endif
	}
}



int main() {

	Mat img_ref = imread("images/ref.bmp", IMREAD_GRAYSCALE);
	Mat img_tar = imread("images/tar.bmp", IMREAD_GRAYSCALE);
	int h = img_ref.rows; // image size must be the same
	int w = img_ref.cols;


#ifdef DEBUG_EX
	Mat img_small(h, w, CV_8UC1);
	resize(img_tar, img_small, Size(h * 4 / 3, w * 4 / 3));
	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
			img_tar.at<uchar>(i, j) = 0;
	for (int i = 0; i < (h * 4 / 3); i++)
		for (int j = 0; j < (w * 4 / 3); j++) {
			int x = j - 60;
			int y = i - 50;
			if (x >= 0 && x < w && y >= 0 && y < h)
				img_tar.at<uchar>(y, x) = img_small.at<uchar>(i, j);
		}
#endif


	// generate gaussian window function as kernel
	gaussian();

#ifdef DEBUG_G
	Mat g_img(G_SIZE, G_SIZE, CV_8UC1);
	double tot = 0;
	for (int y = 0; y < G_SIZE; y++) {
		for (int x = 0; x < G_SIZE; x++) {
			g_img.at<uchar>(y, x) = FLTR_G[y][x] * 255;
			tot += FLTR_G[y][x] * 255;
		}
	}
	imwrite("gaussian.bmp", g_img);
	printf("%15Lf\n", tot); 
#endif

	
	// get gradiant of image, compute R and estimate the edge
	TAIL = (node*)calloc(1, sizeof(node));
	h = h - 2;
	w = w - 2;
	float* ref_grad_x, * ref_grad_y;
	if (!(ref_grad_x = (float*)malloc(sizeof(int) * h * w))) return -1;
	if (!(ref_grad_y = (float*)malloc(sizeof(int) * h * w))) return -1;
	node* head_ref = cornerOf(img_ref, &ref_grad_x, &ref_grad_y);
	
	float* tar_grad_x, * tar_grad_y;
	if (!(tar_grad_x = (float*)malloc(sizeof(int) * h * w))) return -1;
	if (!(tar_grad_y = (float*)malloc(sizeof(int) * h * w))) return -1;
	node* head_tar = cornerOf(img_tar, &tar_grad_x, &tar_grad_y);

#ifdef DEBUG_C_RES
	for (node* t = head_ref->next; t != TAIL; t = t->next) {
		Scalar c;
		Point pCenter;
		int radius = 9;
		pCenter.x = t->x + 1;
		pCenter.y = t->y + 1;
		c.val[0] = 0;
		c.val[1] = 0;
		c.val[2] = 255;
		circle(img_tar, pCenter, radius, c, 1, 8, 0); // red circle
	}
	imshow("corners", img_tar);
	waitKey(0);
#endif

	
	// Non-maxima suppression
	suppress(head_ref);
	suppress(head_tar);


#ifdef DEBUG_SUP
	for (node* t = head_ref->next; t != TAIL; t = t->next) {
		Scalar c;
		Point pCenter;
		int radius = 9;
		pCenter.x = t->x + 1;
		pCenter.y = t->y + 1;
		c.val[0] = 0;
		c.val[1] = 0;
		c.val[2] = 255;
		circle(img_ref, pCenter, radius, c, 1, 8, 0); // red circle
	}
	imshow("corners", img_ref);
	waitKey(0);
#endif


	// Compute HOG descriptors around corner points
	HOGdescript(h, w, ref_grad_x, ref_grad_y, head_ref);
	HOGdescript(h, w, tar_grad_x, tar_grad_y, head_tar);


	// Compare each Histograms and find similar corner points
	similarity(head_ref, head_tar);


	// See result image
	Mat img_clr_ref(img_ref.rows, img_ref.cols, CV_8UC3);
	Mat img_clr_tar(img_tar.rows, img_tar.cols, CV_8UC3);
	cvtColor(img_ref, img_clr_ref, COLOR_GRAY2BGR);
	cvtColor(img_tar, img_clr_tar, COLOR_GRAY2BGR);
	Mat img_res(img_ref.rows, img_ref.cols * 2, CV_8UC3);
	hconcat(img_clr_ref, img_clr_tar, img_res);

	for (node* t = head_ref->next; t != TAIL; t = t->next) {
		Scalar color;
		Point p_cornr;
		p_cornr.x = t->x + 1;
		p_cornr.y = t->y + 1;
		color.val[0] = 0;
		color.val[1] = 255;
		color.val[2] = 0;
		circle(img_res, p_cornr, 9, color, 1, 8, 0);

		if (t->sim_to) {
			Point p_targt;
			p_targt.x = t->sim_to->x + img_ref.rows + 1;
			p_targt.y = t->sim_to->y + 1;
			color.val[0] = 0;
			color.val[1] = 255;
			color.val[2] = 0;
			circle(img_res, p_targt, 9, color, 1, 8, 0);
			color.val[0] = 255;
			color.val[1] = 0;
			color.val[2] = 0;
			line(img_res, p_cornr, p_targt, color, 1, 8, 0);
		}
	}

	imshow("result", img_res);
	waitKey(0);

	return 0;
}