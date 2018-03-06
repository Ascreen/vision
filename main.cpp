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

    markerImage.convertTo(markers, CV_32S); //32비트마커스로자료형변환

  }

  cv::Mat process(const cv::Mat& image) {

    cv::watershed(image, markers);

    //분할결과를markers에저장

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
        cv::imshow("Original Image", frame); //원본
        cv::Mat gray_image;

        cv::cvtColor(frame,gray_image,CV_BGR2GRAY);

        cv::imshow("Gray Image", gray_image); //gray영상

        cv::Mat binary_image;

        cv::threshold(gray_image,binary_image,90,255, cv::THRESH_BINARY_INV);

        cv::imshow("Binary Image", binary_image); //이진영상으로변환(손하얗게끔inverse)



        cv::Mat fg;

        cv::erode(binary_image, fg, cv::Mat(), cv::Point(-1,-1), 12); //침식

        cv::imshow("Foreground", fg);



        cv::Mat bg;

        cv::dilate(binary_image, bg, cv::Mat(), cv::Point(-1,-1), 40); //팽창

        cv::threshold(bg, bg, 1, 128, cv::THRESH_BINARY_INV);

        //(1보다작은)배경을128, (1보다큰)객체0. Threshold설정INVERSE 적용.

        cv::imshow("Background", bg);



        cv::Mat markers(binary_image.size(), CV_8U, cv::Scalar(0));

        markers = fg + bg; //침식+팽창= 마커영상으로조합. 워터쉐드알고리즘에 입력으로 사용됨.

        cv::imshow("Marker", markers);



        WatershedSegmenter segmenter; //워터쉐드분할객체생성

        segmenter.setMarkers(markers); //set마커하면signed 이미지로바뀜

        segmenter.process(frame); //0,128,255로구성됨

        cv::imshow("Segmentation", segmenter.getSegmentation());



        cv::imshow("Watershed", segmenter.getWatersheds()); // 0,255로구성됨



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


