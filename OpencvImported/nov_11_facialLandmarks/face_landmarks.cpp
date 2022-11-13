#include "./face_landmarks.h"
#include <vector>
#include <iostream>
#include <fstream>


#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "ldmarkmodel.h"


using namespace std;
using namespace cv;

#define DEBUG


void LBP(Mat img, Point p, Mat* res) {

    const int P = 8;
    const int R = 1;
    int h = 14, w = 14;
    Mat res;

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

            res->at<uchar>(y - 1, x - 1) = uniform[v];
        }

}


#ifdef DEBUG

int main()
{
    /*********************
    std::vector<ImageLabel> mImageLabels;
    if(!load_ImageLabels("mImageLabels-test.bin", mImageLabels)){
        mImageLabels.clear();
        ReadLabelsFromFile(mImageLabels, "labels_ibug_300W_test.xml");
        save_ImageLabels(mImageLabels, "mImageLabels-test.bin");
    }
    std::cout << "测试数据一共有: " <<  mImageLabels.size() << std::endl;
    *******************/

    ldmarkmodel modelt;
    std::string modelFilePath = "nov_11_facialLandmarks/roboman-landmark-model.bin";
    while (!load_ldmarkmodel(modelFilePath, modelt)) {
        std::cout << "文件打开错误，请重新输入文件路径." << std::endl;
        std::cin >> modelFilePath;
    }

    cv::VideoCapture mCamera(0);
    if (!mCamera.isOpened()) {
        std::cout << "Camera opening failed..." << std::endl;
        system("pause");
        return 0;
    }
    cv::Mat img_ref;
    cv::Mat shape_ref;
    while (waitKey(5) == -1) {
        mCamera >> img_ref;
        modelt.track(img_ref, shape_ref);
        //cv::Vec3d eav;
        //modelt.EstimateHeadPose(current_shape, eav);
        //modelt.drawPose(Image, current_shape, 50);

        int numLandmarks = shape_ref.cols / 2;
        for (int j = 0; j < numLandmarks; j++) {
            int x = shape_ref.at<float>(j);
            int y = shape_ref.at<float>(j + numLandmarks);
            //std::stringstream ss;
            //ss << j;
            //cv::putText(Image, ss.str(), cv::Point(x, y), 0.5, 0.5, cv::Scalar(0, 0, 255));
            
            cv::circle(img_ref, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), -1);
        }

        cv::imshow("Camera", img_ref);
    }
    cv::Mat img_tar;
    cv::Mat shape_tar;
    while (waitKey(5) == -1) {
        mCamera >> img_tar;
        modelt.track(img_tar, shape_tar);
        //cv::Vec3d eav;
        //modelt.EstimateHeadPose(current_shape, eav);
        //modelt.drawPose(Image, current_shape, 50);

        int numLandmarks = shape_tar.cols / 2;
        for (int j = 0; j < numLandmarks; j++) {
            int x = shape_tar.at<float>(j);
            int y = shape_tar.at<float>(j + numLandmarks);
            Scalar color;
            //std::stringstream ss;
            //ss << j;
            //cv::putText(Image, ss.str(), cv::Point(x, y), 0.5, 0.5, cv::Scalar(0, 0, 255));



            cv::circle(img_tar, cv::Point(x, y), 1, color, -1);
        }

        cv::imshow("Camera", img_tar);
    }


    mCamera.release();
    cv::destroyAllWindows();
    system("pause");
    return 0;
}

#endif // DEBUG




















