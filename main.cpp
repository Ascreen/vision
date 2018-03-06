#include "opencv2/highgui/highgui.hpp"
#include "opencv/cv.h"
#include <iostream>

#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc.hpp"

#include <stdio.h>
#include "opencv/cxcore.h"
#include <windows.h>

#include "opencv2/core/core.hpp"

#include <stdlib.h>
#include "opencv2/imgproc/imgproc_c.h"

#include "opencv2/opencv.hpp"

/*
////////mulmi test code

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

/*
////////YCbCr color model SKIN DETECTION

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

*/


/*

////////binary SKIN DETECTION

class WatershedSegmenter{

  private:

  cv::Mat markers;

  public:

  void setMarkers(const cv::Mat& markerImage) {

    markerImage.convertTo(markers, CV_32S); //32��Ʈ��Ŀ�����ڷ�����ȯ

  }

  cv::Mat process(const cv::Mat& image) {

    cv::watershed(image, markers);

    //���Ұ����markers������

    return markers;

  }

  cv::Mat getSegmentation() {

    cv::Mat tmp;

    markers.convertTo(tmp, CV_8U); return tmp;

  }

  cv::Mat getWatersheds() {

    cv::Mat tmp;

    markers.convertTo(tmp, CV_8U, 255, 255); return tmp;

  }

};


int main(){

    cv::VideoCapture capture("testing.h264");
    cv::Mat frame;
    cv::Mat skinMat;

    while(1){
        capture.read(frame);

        cv::Mat image = cv::imread("hand.jpg");
        cv::imshow("Original Image", frame); //����
        cv::Mat gray_image;

        cv::cvtColor(frame,gray_image,CV_BGR2GRAY);

        cv::imshow("Gray Image", gray_image); //gray����

        cv::Mat binary_image;

        cv::threshold(gray_image,binary_image,90,255, cv::THRESH_BINARY_INV);

        cv::imshow("Binary Image", binary_image); //�����������κ�ȯ(���Ͼ�Բ�inverse)



        cv::Mat fg;

        cv::erode(binary_image, fg, cv::Mat(), cv::Point(-1,-1), 12); //ħ��

        cv::imshow("Foreground", fg);



        cv::Mat bg;

        cv::dilate(binary_image, bg, cv::Mat(), cv::Point(-1,-1), 40); //��â

        cv::threshold(bg, bg, 1, 128, cv::THRESH_BINARY_INV);

        //(1��������)�����128, (1����ū)��ü0. Threshold����INVERSE ����.

        cv::imshow("Background", bg);



        cv::Mat markers(binary_image.size(), CV_8U, cv::Scalar(0));

        markers = fg + bg; //ħ��+��â= ��Ŀ������������. ���ͽ���˰��� �Է����� ����.

        cv::imshow("Marker", markers);



        WatershedSegmenter segmenter; //���ͽ�����Ұ�ü����

        segmenter.setMarkers(markers); //set��Ŀ�ϸ�signed �̹����ιٲ�

        segmenter.process(frame); //0,128,255�α�����

        cv::imshow("Segmentation", segmenter.getSegmentation());



        cv::imshow("Watershed", segmenter.getWatersheds()); // 0,255�α�����



        if(cvWaitKey(10)>0)
            break;
    }

  return 0;

}

*/



////////HSV color model SKIN DETECTION

using namespace cv;
using std::cout;

int main () {
    VideoCapture cap("testing.h264");
    Mat frame2;
    Mat3b frame;

	while(cap.read(frame)&cap.read(frame2)){

		/* THRESHOLD ON HSV*/
		cvtColor(frame, frame, CV_BGR2HSV);
		GaussianBlur(frame, frame, Size(7,7), 1, 1);
		for(int r=0; r<frame.rows; ++r){
			for(int c=0; c<frame.cols; ++c)
				// 0<H<0.25  -   0.15<S<0.9    -    0.2<V<0.95
				if( (frame(r,c)[0]>5) && (frame(r,c)[0] < 17) && (frame(r,c)[1]>38) && (frame(r,c)[1]<250) && (frame(r,c)[2]>51) && (frame(r,c)[2]<242) ); // do nothing
				else for(int i=0; i<3; ++i)	frame(r,c)[i] = 0;
		}

		/* BGR CONVERSION AND THRESHOLD */
		Mat1b frame_gray;
		cvtColor(frame, frame, CV_HSV2BGR);
		cvtColor(frame, frame_gray, CV_BGR2GRAY);
		threshold(frame_gray, frame_gray, 60, 255, CV_THRESH_BINARY);
		morphologyEx(frame_gray, frame_gray, CV_MOP_ERODE, Mat1b(3,3,1), Point(-1, -1), 3);
		morphologyEx(frame_gray, frame_gray, CV_MOP_OPEN, Mat1b(7,7,1), Point(-1, -1), 1);
		morphologyEx(frame_gray, frame_gray, CV_MOP_CLOSE, Mat1b(9,9,1), Point(-1, -1), 1);

		medianBlur(frame_gray, frame_gray, 15);
        imshow("Threshold", frame_gray);

		cvtColor(frame, frame, CV_BGR2HSV);
		resize(frame, frame, Size(), 0.5, 0.5);
		imshow("Video",frame);
		imshow("input",frame2);

		waitKey(5);
	}
}


