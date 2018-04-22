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
#include <cstring>


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

    markerImage.convertTo(markers, CV_32S); //32ë¹„íŠ¸ë§ˆì»¤?¤ë¡œ?ë£Œ?•ë???

  }

  cv::Mat process(const cv::Mat& image) {

    cv::watershed(image, markers);

    //ë¶„í• ê²°ê³¼ë¥¼markers?ì???

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
        cv::imshow("Original Image", frame); //?ë³¸
        cv::Mat gray_image;

        cv::cvtColor(frame,gray_image,CV_BGR2GRAY);

        cv::imshow("Gray Image", gray_image); //gray?ìƒ

        cv::Mat binary_image;

        cv::threshold(gray_image,binary_image,90,255, cv::THRESH_BINARY_INV);

        cv::imshow("Binary Image", binary_image); //?´ì§„?ìƒ?¼ë¡œë³€???í•˜?—ê²Œ?”inverse)



        cv::Mat fg;

        cv::erode(binary_image, fg, cv::Mat(), cv::Point(-1,-1), 12); //ì¹¨ì‹

        cv::imshow("Foreground", fg);



        cv::Mat bg;

        cv::dilate(binary_image, bg, cv::Mat(), cv::Point(-1,-1), 40); //?½ì°½

        cv::threshold(bg, bg, 1, 128, cv::THRESH_BINARY_INV);

        //(1ë³´ë‹¤?‘ì?)ë°°ê²½??28, (1ë³´ë‹¤??ê°ì²´0. Threshold?¤ì •INVERSE ?ìš©.

        cv::imshow("Background", bg);



        cv::Mat markers(binary_image.size(), CV_8U, cv::Scalar(0));

        markers = fg + bg; //ì¹¨ì‹+?½ì°½= ë§ˆì»¤?ìƒ?¼ë¡œì¡°í•©. ?Œí„°?ë“œ?Œê³ ë¦¬ì¦˜???…ë ¥?¼ë¡œ ?¬ìš©??

        cv::imshow("Marker", markers);



        WatershedSegmenter segmenter; //?Œí„°?ë“œë¶„í• ê°ì²´?ì„±

        segmenter.setMarkers(markers); //setë§ˆì»¤?˜ë©´signed ?´ë?ì§€ë¡œë°”??

        segmenter.process(frame); //0,128,255ë¡œêµ¬?±ë¨

        cv::imshow("Segmentation", segmenter.getSegmentation());



        cv::imshow("Watershed", segmenter.getWatersheds()); // 0,255ë¡œêµ¬?±ë¨



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
int g_thresh=30; //contour bar ì´ˆê¸°ê°?
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

        //g_image?ìƒ??BRG?‰ê³µê°„ì„ ê·¸ë ˆ???¤ì??¼ë¡œ ë³€??BGR to Gray = BGR2GRAY)
        cvCvtColor(g_image, g_gray, CV_BGR2GRAY);
        //?„ê³„ê°??´í•˜:0, ?„ê³„ê°’ì´ˆê³¼ê°’:1 ?¤ì •
        cvThreshold(g_gray, g_gray, g_thresh, 255, CV_THRESH_BINARY);
        cvCopy(g_gray, g_binary);
        //?¤ê³½??ì°¾ê¸°
        cvFindContours(
               g_gray,                //?…ë ¥?ìƒ
               g_storage,             //ê²€ì¶œëœ ?¸ê³½? ì„ ê¸°ë¡?˜ê¸° ?„í•œ ë©”ëª¨ë¦??¤í† ë¦¬ì?
               &contours,             //?¸ê³½? ì˜ ì¢Œí‘œ?¤ì´ ?€?¥ëœ Sequence
               sizeof(CvContour),
               CV_RETR_TREE           //?´ë–¤ì¢…ë¥˜???¸ê³½??ì°¾ì„ì§€, ?´ë–»ê²?ë³´ì—¬ì¤„ì????€?œì •ë³?
        );

        cvZero(g_gray);

        if(contours) {
               //?¸ê³½? ì„ ì°¾ì? ?•ë³´(contour)ë¥??´ìš©?˜ì—¬ ?¸ê³½? ì„ ê·¸ë¦¼
               cvDrawContours(
                       g_gray,                //?¸ê³½? ì´ ê·¸ë ¤ì§??ìƒ
                       contours,              //?¸ê³½???¸ë¦¬??ë£¨íŠ¸?¸ë“œ
                       cvScalarAll(255),      //?¸ë? ?¸ê³½? ì˜ ?‰ìƒ
                       cvScalarAll(128),      //?´ë? ?¸ê³½? ì˜ ?‰ìƒ
                       100                    //?¸ê³½? ì„ ê·¸ë¦´???´ë™??ê¹Šì´
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
	//e1 ê°?: ?¸ë???minê°’ì„ ?’ìž„
	/*
    bool e1 = (R>130) && (G>130) && (B>20) && ((max(R,max(G,B)) - min(R, min(G,B)))>15) && (abs(R-G)>15) && (R>G) && (R>B);
    bool e2 = (R>220) && (G>210) && (B>170) && (abs(R-G)<=15) && (R>B) && (G>B);
    */

    bool e1 = (R>80) && (G>80) && (B>10) && ((max(R,max(G,B)) - min(R, min(G,B)))>15) && (abs(R-G)>15) && (R>G) && (R>B);
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


Point2f clickPoint = Point2f(0.0,0.0); //BLUE circle point

int frameUnit=3, frameUnitMid=1;  // 3 frames ´ÜÀ§  +-3.5 ¿ÀÂ÷¹üÀ§
float errorRange= 3.5;

int comparePoints(vector<Point2f> points){
    vector<Point2f> differ;
    if(points.size()==frameUnit){
        for(int i=0; i<(frameUnit-1); i++){
            differ.push_back(Point2f(fabs(points[i+1].x-points[i].x),fabs(points[i+1].y-points[i].y)));
        }

        float differX=0.0, differY=0.0;
        for(int i=0; i<(frameUnit-1); i++){
            differX += differ[i].x;
            differY += differ[i].y;
        }
        differX = differX/(frameUnit-1);
        differY = differY/(frameUnit-1);

        if(0.0<=differX && differX<errorRange){
            if(0.0<=differY && differY<=errorRange){
                clickPoint = points[frameUnitMid];
                std::cout<<"***********"<<std::endl;
                std::cout<<points[frameUnitMid]<<std::endl;
                std::cout<<"***********"<<std::endl;
            }
        }
        differ.clear();
        return 1;
    } else
        return 0;
}



int main()
{
	Mat tmpImg, handImg, mask;
	std::vector<std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;
	vector<Point2f> points;


	/*
	VideoCapture video(0);
	if (!video.isOpened()) return 0;
	*/
    VideoCapture video("C:/Users/macbook/Desktop/record3/new_ul01.h264"); //dl005 dr002bigerror dr005error »ç°¢Çü »çÀÌÁî ¹üÀ§¸¦ Á¤ÇÏ´Â°Ô ÁÁÀ»µí
	//VideoCapture video("dl03.h264");
    Mat image;



    namedWindow("hand1_image", CV_WINDOW_AUTOSIZE);
	namedWindow("hand2_image", CV_WINDOW_AUTOSIZE);
	namedWindow("original_image", CV_WINDOW_AUTOSIZE);


	while (true)
	{
	    /*
	    video >> image;
		if (image.empty()) break;
		*/
		video.read(image);

        tmpImg = GetSkin(image);

		cvtColor(tmpImg, handImg, CV_BGR2YCrCb);
		inRange(handImg, Scalar(0, 133, 77), Scalar(255, 173, 127), handImg);

		mask = handImg.clone();

		findContours(mask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_TC89_L1, Point(0,0));

		int largestContour = 0;
		for (int i = 0; i < contours.size(); i++) {
            //double a = contourArea(contours[i]);
			if (contourArea(contours[i]) > contourArea(contours[largestContour])) {
				largestContour = i;
			}
		}

		drawContours(image, contours, largestContour, Scalar(0, 255, 255), -1, 8, std::vector <Vec4i>(), 0, Point());	//YELLOW contour ê·¸ë¦¬ê¸?
        //5th index must be -1 for drawing only one contour from present Contours variable)


        // loop through the contours/hierarchy
        for (int i=0; i<contours.size(); i++) {
            if(i==largestContour){  //¼±ÅÃµÈ À±°û¼±¸¸ blue »ç°¢Çü ±×¸®±â
                if (hierarchy[i][3]==-1 && arcLength(contours[i], true)!=arcLength(contours[i], false)) {
                    std::vector<std::vector<Point> >hull(1);
                    convexHull(Mat(contours[i]), hull[0], false);
                    drawContours(image, hull, 0, Scalar(0, 255, 0), 1, 8, std::vector<Vec4i>(), 0, Point());		//GREEN Á¡ ÀÕ±â
                    if(1){
                        std::vector<RotatedRect> minRect( contours.size() );
                        minRect[i] = minAreaRect( Mat(contours[i]) );
                        Point2f rect_points[4];
                        minRect[i].points( rect_points );
                        for( int j = 0; j < 4; j++ )
                            line( image, rect_points[j], rect_points[(j+1)%4], Scalar(255, 0, 0), 1, 8 );   //BLUE »ç°¢Çü ±×¸®±â
                        std::cout<<rect_points[2]<<std::endl;          //point ÁÂÇ¥ °ª printÇÏ±â

                        points.push_back(Point2f(rect_points[2].x,rect_points[2].y));
                        if(comparePoints(points)){
                            if(clickPoint.x!=0.0 && clickPoint.y!=0.0){
                                circle(image, clickPoint, 10, Scalar(255, 0, 0), 10); //BLUE circle
                                clickPoint = Point2f(0.0,0.0);
                            }
                            points.clear();
                        }
                    }
                }
            }
        }

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

