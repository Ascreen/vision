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


/*
////////HSV color model SKIN DETECTION

using namespace cv;
using std::cout;

int main () {
    VideoCapture cap("dl003.h264");
    Mat frame2;
    Mat3b frame;

	while(cap.read(frame)&cap.read(frame2)){

		// THRESHOLD ON HSV
		cvtColor(frame, frame, CV_BGR2HSV);
		GaussianBlur(frame, frame, Size(7,7), 1, 1);
		for(int r=0; r<frame.rows; ++r){
			for(int c=0; c<frame.cols; ++c)
				// 0<H<0.25  -   0.15<S<0.9    -    0.2<V<0.95
				if( (frame(r,c)[0]>5) && (frame(r,c)[0] < 17) && (frame(r,c)[1]>38) && (frame(r,c)[1]<250) && (frame(r,c)[2]>51) && (frame(r,c)[2]<242) ); // do nothing
				else for(int i=0; i<3; ++i)	frame(r,c)[i] = 0;
		}

		// BGR CONVERSION AND THRESHOLD
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

*/


/*
////////RGB-H-CbCr color model SKIN DETECTION & FINGER CONTOUR

using namespace cv;

using std::cout;
using std::endl;

bool R1(int R, int G, int B) {
    bool e1 = (R>95) && (G>40) && (B>20) && ((max(R,max(G,B)) - min(R, min(G,B)))>15) && (abs(R-G)>15) && (R>G) && (R>B);
    bool e2 = (R>220) && (G>210) && (B>170) && (abs(R-G)<=15) && (R>B) && (G>B);
    return (e1||e2);
}

bool R2(float Y, float Cr, float Cb) {
    bool e3 = Cr <= 1.5862*Cb+20;
    bool e4 = Cr >= 0.3448*Cb+76.2069;
    bool e5 = Cr >= -4.5652*Cb+234.5652;
    bool e6 = Cr <= -1.15*Cb+301.75;
    bool e7 = Cr <= -2.2857*Cb+432.85;
    return e3 && e4 && e5 && e6 && e7;
}

bool R3(float H, float S, float V) {
    return (H<25) || (H > 230);
}

Mat GetSkin(Mat const &src) {
    // allocate the result matrix
    Mat dst = src.clone();

    Vec3b cwhite = Vec3b::all(255);
    Vec3b cblack = Vec3b::all(0);

    Mat src_ycrcb, src_hsv;
    // OpenCV scales the YCrCb components, so that they
    // cover the whole value range of [0,255], so there's
    // no need to scale the values:
    cvtColor(src, src_ycrcb, CV_BGR2YCrCb);
    // OpenCV scales the Hue Channel to [0,180] for
    // 8bit images, so make sure we are operating on
    // the full spectrum from [0,360] by using floating
    // point precision:
    src.convertTo(src_hsv, CV_32FC3);
    cvtColor(src_hsv, src_hsv, CV_BGR2HSV);
    // Now scale the values between [0,255]:
    normalize(src_hsv, src_hsv, 0.0, 255.0, NORM_MINMAX, CV_32FC3);

    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {

            Vec3b pix_bgr = src.ptr<Vec3b>(i)[j];
            int B = pix_bgr.val[0];
            int G = pix_bgr.val[1];
            int R = pix_bgr.val[2];
            // apply rgb rule
            bool a = R1(R,G,B);

            Vec3b pix_ycrcb = src_ycrcb.ptr<Vec3b>(i)[j];
            int Y = pix_ycrcb.val[0];
            int Cr = pix_ycrcb.val[1];
            int Cb = pix_ycrcb.val[2];
            // apply ycrcb rule
            bool b = R2(Y,Cr,Cb);

            Vec3f pix_hsv = src_hsv.ptr<Vec3f>(i)[j];
            float H = pix_hsv.val[0];
            float S = pix_hsv.val[1];
            float V = pix_hsv.val[2];
            // apply hsv rule
            bool c = R3(H,S,V);

            if(!(a&&b&&c))
                dst.ptr<Vec3b>(i)[j] = cblack;
        }
    }
    return dst;
}



IplImage* g_image=NULL;
IplImage* g_gray=NULL;
IplImage* g_binary=NULL;
int g_thresh=30; //contour bar 초기값
CvMemStorage* g_storage=NULL;


//
void on_trackbar(int pos) {
        if(g_storage==NULL) {
               g_gray=cvCreateImage(cvGetSize(g_image), 8, 1);
               g_binary=cvCreateImage(cvGetSize(g_image), 8, 1);
               g_storage=cvCreateMemStorage(0);
        } else {
               cvClearMemStorage(g_storage);
        }
        CvSeq* contours=0;

        //g_image영상을 BRG색공간을 그레이 스케일로 변환(BGR to Gray = BGR2GRAY)
        cvCvtColor(g_image, g_gray, CV_BGR2GRAY);
        //임계값 이하:0, 임계값초과값:1 설정
        cvThreshold(g_gray, g_gray, g_thresh, 255, CV_THRESH_BINARY);
        cvCopy(g_gray, g_binary);
        //윤곽선 찾기
        cvFindContours(
               g_gray,                //입력영상
               g_storage,             //검출된 외곽선을 기록하기 위한 메모리 스토리지
               &contours,             //외곽선의 좌표들이 저장된 Sequence
               sizeof(CvContour),
               CV_RETR_TREE           //어떤종류의 외곽선 찾을지, 어떻게 보여줄지에 대한정보
        );

        cvZero(g_gray);

        if(contours) {
               //외곽선을 찾은 정보(contour)를 이용하여 외곽선을 그림
               cvDrawContours(
                       g_gray,                //외곽선이 그려질 영상
                       contours,              //외곽선 트리의 루트노드
                       cvScalarAll(255),      //외부 외곽선의 색상
                       cvScalarAll(128),      //내부 외곽선의 색상
                       100                    //외곽선을 그릴때 이동할 깊이
               );
        }

        cvShowImage("Binary", g_binary);
        cvShowImage("Contours", g_gray);
}
//


int main(int argc, const char *argv[]) {
    VideoCapture capture("dl003.h264");
    //VideoCapture capture("C:/Users/macbook/Desktop/record/dr002.h264");
    Mat image;
    Mat skin;
    //
    IplImage temp;
    //

    while(1){
        capture.read(image);
        skin = GetSkin(image);

        //
        temp = skin;
        g_image = &temp;
        cvCreateTrackbar("Threshold", "Contours", &g_thresh, 255, on_trackbar);
        on_trackbar(0);
        //

        namedWindow("original");
        namedWindow("skin");
        imshow("original",image);
        imshow("skin",skin);

        if(cvWaitKey(10)>0)
            break;
    }
    cvDestroyAllWindows();
    capture.release();

    return 0;
}

*/

////////RGB-H-CbCr + YCrCb color model SKIN DETECTION & FINGER CONTOUR.v2

using namespace cv;

using std::cout;
using std::endl;

bool R1(int R, int G, int B) {
	//e1 값 : 노란색 min값을 높임
    bool e1 = (R>130) && (G>130) && (B>20) && ((max(R,max(G,B)) - min(R, min(G,B)))>15) && (abs(R-G)>15) && (R>G) && (R>B);
    bool e2 = (R>220) && (G>210) && (B>170) && (abs(R-G)<=15) && (R>B) && (G>B);
    return (e1||e2);
}

bool R2(float Y, float Cr, float Cb) {
    bool e3 = Cr <= 1.5862*Cb+20;
    bool e4 = Cr >= 0.3448*Cb+76.2069;
    bool e5 = Cr >= -4.5652*Cb+234.5652;
    bool e6 = Cr <= -1.15*Cb+301.75;
    bool e7 = Cr <= -2.2857*Cb+432.85;
    return e3 && e4 && e5 && e6 && e7;
}

bool R3(float H, float S, float V) {
    return (H<25) || (H > 230);
}

Mat GetSkin(Mat const &src) {
    // allocate the result matrix
    Mat dst = src.clone();

    Vec3b cblack = Vec3b::all(0);

    Mat src_ycrcb, src_hsv;
    // OpenCV scales the YCrCb components, so that they
    // cover the whole value range of [0,255], so there's
    // no need to scale the values:
    cvtColor(src, src_ycrcb, CV_BGR2YCrCb);
    // OpenCV scales the Hue Channel to [0,180] for
    // 8bit images, so make sure we are operating on
    // the full spectrum from [0,360] by using floating
    // point precision:
    src.convertTo(src_hsv, CV_32FC3);
    cvtColor(src_hsv, src_hsv, CV_BGR2HSV);
    // Now scale the values between [0,255]:
    normalize(src_hsv, src_hsv, 0.0, 255.0, NORM_MINMAX, CV_32FC3);

    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {

            Vec3b pix_bgr = src.ptr<Vec3b>(i)[j];
            int B = pix_bgr.val[0];
            int G = pix_bgr.val[1];
            int R = pix_bgr.val[2];
            // apply rgb rule
            bool a = R1(R,G,B);

            Vec3b pix_ycrcb = src_ycrcb.ptr<Vec3b>(i)[j];
            int Y = pix_ycrcb.val[0];
            int Cr = pix_ycrcb.val[1];
            int Cb = pix_ycrcb.val[2];
            // apply ycrcb rule
            bool b = R2(Y,Cr,Cb);

            Vec3f pix_hsv = src_hsv.ptr<Vec3f>(i)[j];
            float H = pix_hsv.val[0];
            float S = pix_hsv.val[1];
            float V = pix_hsv.val[2];
            // apply hsv rule
            bool c = R3(H,S,V);

            if(!(a&&b&&c))
                dst.ptr<Vec3b>(i)[j] = cblack;
        }
    }
    return dst;
}


int main()
{
	Mat tmpImg, handImg, mask;
	std::vector<std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;

    VideoCapture video("C:/Users/macbook/Desktop/record/dr005.h264"); //dl005 dr002
	//VideoCapture video("dl003.h264");
    Mat image;

    namedWindow("hand1_image", CV_WINDOW_AUTOSIZE);
	namedWindow("hand2_image", CV_WINDOW_AUTOSIZE);
	namedWindow("original_image", CV_WINDOW_AUTOSIZE);

	while (true)
	{
		video.read(image);
        tmpImg = GetSkin(image);

		cvtColor(tmpImg, handImg, CV_BGR2YCrCb);
		inRange(handImg, Scalar(0, 133, 77), Scalar(255, 173, 127), handImg);

		mask = handImg.clone();

		findContours(mask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_TC89_L1, Point(0,0));

		int largestContour = 0;
		for (int i = 0; i < contours.size(); i++) {
			if (contourArea(contours[i]) > contourArea(contours[largestContour])) {
				largestContour = i;
			}
		}

		drawContours(image, contours, largestContour, Scalar(0, 255, 255), 1, 8, std::vector <Vec4i>(), 0, Point());	//YELLOW contour 그리기





            // loop through the contours/hierarchy
            for (int i=0; i<contours.size(); i++) {
                if (hierarchy[i][3]==-1 && arcLength(contours[i], true)!=arcLength(contours[i], false)) {
                    std::vector<std::vector<Point> >hull(1);
                    convexHull(Mat(contours[i]), hull[0], false);
                    drawContours(image, hull, 0, Scalar(0, 255, 0), 1, 8, std::vector<Vec4i>(), 0, Point());		//GREEN �� �ձ�
                    if(1){
                        std::vector<RotatedRect> minRect( contours.size() );
                        minRect[i] = minAreaRect( Mat(contours[i]) );
                        Point2f rect_points[4];
                        minRect[i].points( rect_points );
                        for( int j = 0; j < 4; j++ )
                            line( image, rect_points[j], rect_points[(j+1)%4], Scalar(255, 0, 0), 1, 8 );   //BLUE �簢�� �׸���
                    }
                }
            }




		/*
		if (!contours.empty()) {
			std::vector<std::vector<Point> >hull(1);

			convexHull(Mat(contours[largestContour]), hull[0], false);
			drawContours(image, hull, 0, Scalar(0, 255, 0), 1, 8, std::vector<Vec4i>(), 0, Point());		//GREEN 점 잇기
		}
		*/


        //Size(960,540) , Size(800,450) , Size(720,405)
		resize(image, image, Size(960,540), 0, 0, CV_INTER_LINEAR);
		resize(tmpImg, tmpImg, Size(720,405), 0, 0, CV_INTER_LINEAR);
        resize(handImg, handImg, Size(720,405), 0, 0, CV_INTER_LINEAR);

		imshow("hand1_image", tmpImg);
		imshow("hand2_image", handImg);
		imshow("original_image", image);

		if (waitKey(10)>0)
			break;
	}

	video.release();
	tmpImg.release();
    handImg.release();
    image.release();

	destroyAllWindows();

	return 0;
}

