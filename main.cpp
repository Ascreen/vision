#include "opencv2/highgui/highgui.hpp"
#include "opencv/cv.h"
#include <iostream>

/*

using namespace cv;
using namespace std;

int main(){

    Mat img = imread("nothing.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat thresh_img = cvCreateMat(img.rows, img.cols, CV_8UC1);

    int x,y;

    int threshold_value = 120;

    for(y=0; y<img.rows; y++){
        for(x=0; x<img.cols; x++){
            int k = (int)(img.at<uchar>(y,x));
            if(k <= threshold_value)
                thresh_img.at<uchar>(y,x) = 0;
            else
                thresh_img.at<uchar>(y,x) = 255;
        }
    }

    namedWindow("Origin Image", CV_WINDOW_AUTOSIZE);
    imshow("Origin Image", img);

    namedWindow("thresholded Image", CV_WINDOW_AUTOSIZE);
    imshow("thresholded Image", thresh_img);

    waitKey(0);
    destroyAllWindows;

    return 0;
}

*/

#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

static char* INPUT_WINDOW = "input window";
static char* RESULT_WINDOW = "result window";

int main(){
    VideoCapture capture("testing.h264");
    Mat frame;
    Mat skinMat;

    namedWindow(INPUT_WINDOW, CV_WINDOW_AUTOSIZE);
    namedWindow(RESULT_WINDOW, CV_WINDOW_AUTOSIZE);

    while(1){
        capture.read(frame);

        cvtColor(frame, skinMat, CV_BGR2YCrCb);
        inRange(skinMat, Scalar(0,133,77), Scalar(255, 173, 127), skinMat);

        imshow(INPUT_WINDOW,frame);
        imshow(RESULT_WINDOW,skinMat);

        if(cvWaitKey(10)>0)
            break;
    }
    cvDestroyAllWindows();
    capture.release();

    return 0;
}
