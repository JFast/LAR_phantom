#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <stdint.h>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <pigpio.h>
#include <wiringPi.h>
#include <sys/time.h>
#include <time.h>

#define CVUI_IMPLEMENTATION
#include "cvui.h"

#define WINDOW_NAME "CONFIG"
#define pi 3.14159265

using namespace cv;
using namespace std;

// global variables
bool RG = false;
Point pt;

// used functions
void on_MouseHandle(int event, int x, int y, int flags, void *ustc);
Mat RegionGrow(Mat src, Point2i pt/*, int th*/);

// program start

int main(int argc, char *argv[])
{
	// flags
	bool use_gaussian = true;
	bool use_rotate = true;
	bool use_live = false;
	bool motionDetected = false;
	bool backgroundUpdate = true;
	
	// parameter setup for blob detection
	cv::SimpleBlobDetector::Params params;
	params.minDistBetweenBlobs = 100.0f;
	params.filterByInertia = false;
	params.filterByConvexity = false;
	params.filterByColor = true;
	params.blobColor = 255; // extract light blobs
	params.filterByCircularity = false;
	params.filterByArea = false;
	params.minArea = 10.0f;
	params.maxArea = 100.0f;
	
	// vector for blob storage
	vector<cv::KeyPoint> keypoints;
	
	// floats for storage of blob coordinates, diameters and radii
	float X, Y, D, R;

	// parameters of Hough circle detection
	int nParam1 = 60, nParam2 = 30, nMinRadius = 5, nMaxRadius = 30, nFoundWhites = 0; 
	vector<Vec3f> vCircles;
	const int nMinWhitePixels = 10;
	
	// parameters of found lines defining vocal fold edges
	float theta[3];
	float a[3], b[3];
	bool vertLabel[3];
	
	// parameters for processing time calculation
	struct timeval t1, t2, t3;
	long long elapsedTime = 0, t4 = 0;

	// LAR parameters
	int closeAngleLeft = 30;
	int closeAngleRight = 30;
	int closureDuration = 200;
	int LAR1 = 100;
	int LAR2 = 100;
	int motorLAR1, motorLAR2;

	// servomotor rotation range
	int motorPos1 = 1080, motorPos2 = 1250;

	// servomotor end positions
	int motorPosLAR1, motorPosLAR2;

	// definition of servomotor control output pins
	const int MotorPinLeft = 19;
	const int MotorPinRight = 18;
	
											
	// points for vocal fold midline
	cv::Point2f pt1, pt2;
	
	// declarations: current frame ("frame", enlarged frame ("frameZoom"), Frame with LAR phantom and droplet ("backgroundFrame"), last for comparison with current frame ("lastFrame"), resulting image ("result"), rotated resulting image ("resultRotate")
	Mat frame, frameZoom, backgroundFrame, lastFrame, result, resultRotate;
	
	// declarations for region growing
	Mat edge, GrowingBild1, GrowingBild2;
	vector<Vec4i> linesP;
	
	// declarations for further image processing
	Mat vForeground, pHoughSearch;
	
	// GUI initialization
	Mat config = Mat(500, 650, CV_8UC3);
	
	// further flags
	int state = 0;
	bool start = 0;
	bool dropletDetected = false;
	
	// initialize camera module
	VideoCapture cap(0);
	
	// check: camera initialized?
	if (!cap.isOpened())
	{
		cerr << "ERROR: Cannot initialize camera." << endl;
		return 0;
	}
	
	// camera setup
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	cap.set(CV_CAP_PROP_FPS, 120);
	
	// GUI initialization
	cv::namedWindow(WINDOW_NAME);
	cvui::init(WINDOW_NAME);	
	
	// blob detector setup
	cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
	
	while (1)
	{	
		// button "Exit" pressed?
		config = cv::Scalar(49, 52, 49);
		if (cvui::button(config, 500, 450, "Exit")) {
			return 0;
		}
		
		switch(state)
		{
			
		// find first vocal fold and calculate angle
		case 0:
			cap >> frame;
			
			// find user-selected point
			namedWindow("Live", WINDOW_NORMAL);
			setMouseCallback("Live", on_MouseHandle);
			imshow("Live", frame);
			
			if (RG == 1)
			{
				cv::cvtColor(frame, frame, COLOR_BGR2GRAY);
				
				// pre-processing
				GaussianBlur(frame, frame, Size(5, 5), 0, 0);
				equalizeHist(frame,frame);
				Canny(frame, frame, 50, 150, 3);
				
				// region growing
				GrowingBild1 = RegionGrow(frame, pt);
				imshow("RG", GrowingBild1);
				
				// edge detection
				Canny(GrowingBild1, edge, 50, 200, 3);
				
				// find Hough line
				HoughLinesP(edge, linesP, 1, CV_PI/180, 30, 30, 10);
				
				Vec4f l=linesP[0];
				
				// calculate line parameters
				if (l[2]==l[0]) 
				{
					vertLabel[0] = 1;
					b[0] = l[2];
					theta[0] = pi/2;
				}
				else
				{	
					vertLabel[0] = 0;
					a[0] = (l[3]-l[1])/(l[2]-l[0]);
					b[0] = l[3]-a[0]*l[2];
					theta[0] = atan(a[0]);
				}
			}
			
			if (!GrowingBild1.empty()) 
			{
				cvui::printf(config, 50, 280, 0.5, 0xffffff, "If segmentation correct, click \"Continue\"!");
				if (cvui::button(config, 280, 450, "Continue")) 
				{
					state = 1;
					
					if (theta[0] < 0) theta[0] = pi+theta[0];
					
					destroyWindow("RG");
				}
			}
			
			cvui::printf(config, 50, 200, 0.5, 0xffffff, "Please click on one vocal fold!");
			break;
		
		// find other vocal fold and calculate angle
		case 1:
		
			cap >> frame;
			
			imshow("Live", frame);
			
			if (RG == 1)
			{
				cv::cvtColor(frame, frame, COLOR_BGR2GRAY);
				
				// pre-processing
				GaussianBlur(frame, frame, Size(5, 5), 0, 0);
				equalizeHist(frame,frame);
				Canny(frame, frame, 50, 150, 3);
				
				// region growing
				GrowingBild2 = RegionGrow(frame, pt);
				imshow("RG", GrowingBild2);
				
				// edge detection
				Canny(GrowingBild2, edge, 50, 200, 3);
				
				// find Hough line
				HoughLinesP(edge, linesP, 1, CV_PI/180, 30, 30, 10);
				
				Vec4f l=linesP[0];
				
				// calculate line parameters
				if (l[2]==l[0]) 
				{
					vertLabel[1] = 1;
					b[1] = l[2];
					theta[1] = pi/2;
				}
				else
				{
					vertLabel[1] = 0;
					a[1] = (l[3]-l[1])/(l[2]-l[0]);
					b[1] = l[3]-a[1]*l[2];
					theta[1] = atan(a[1]);
				}
			}
			
			if (!GrowingBild2.empty()) 
			{
				cvui::printf(config, 50, 280, 0.5, 0xffffff, "If finished, click \"Continue\"!");
				if (cvui::button(config, 280, 450, "Continue"))
				{
					state = 2;
					
					if (theta[1] < 0) theta[1] = pi+theta[1];
					
					theta[2] = (theta[0]+theta[1])/2;
					
					if (round(theta[2]/pi*180)==90)
					{
						vertLabel[2] = 1;
						b[2] = (b[0]-b[1])/(a[1]-a[0]);
						theta[3] = pi/2;
					}
					
					else
					
					{
						vertLabel[2] = 0;
						a[2] = tan(theta[2]);
						b[2] = (a[2]-a[1])*(b[1]-b[0])/(a[1]-a[0])+b[1];
					}

					destroyWindow("RG");
					destroyWindow("Live");
				}
			}
			
			cvui::printf(config, 50, 200, 0.5, 0xffffff, "Please click on other vocal fold!");
			break;
			
		case 2:
			gettimeofday(&t1, NULL);

			switch(start)
			{
			// droplet detection not yet started, LAR parameters can be set
			case 0:
				if (LAR2 >= LAR1)
				{
					if (cvui::button(config, 280, 450, "Start")) 
					{
						start = 1;
						if (gpioInitialise() < 0) return -1;
						destroyWindow("Live view");
						destroyWindow("Result");
						destroyWindow("Difference image");
						destroyWindow("Frame");
						destroyWindow("BackgroundFrame");
					}
					if (elapsedTime != 0)
						cvui::printf(config, 50, 340, 0.5, 0xff0000, "Processing time: %11d us", elapsedTime);
				}
				
				else
					cvui::printf(config, 50, 340, 0.5, 0xff0000, "ERROR: check LAR2 > LAR1!");
				
				// repeat segmentation
				if (cvui::button(config, 50, 450, "Segmentation"))
				{
					state = 0;
					GrowingBild1.release();
					GrowingBild2.release();
					destroyWindow("Live view");
					continue;
				}
			
				// latency trackbars
				cvui::printf(config, 50, 170, 0.5, 0xff0000, "LAR1 latency (in ms)");
				cvui::printf(config, 350, 170, 0.5, 0xff0000, "LAR2 latency (in ms)");
				cvui::trackbar(config, 50, 200, 250, &LAR1, 50, 200);
				cvui::trackbar(config, 350, 200, 250, &LAR2, 50, 200);
			
				// closure angle trackbars
				cvui::printf(config, 50, 90, 0.5, 0xff0000, "Closing angle left (in degrees)");
				cvui::printf(config, 350, 90, 0.5, 0xff0000, "Closing angle rechts (in degrees)");
				cvui::trackbar(config, 50, 120, 250, &closeAngleLeft, 0, 30);
				cvui::trackbar(config, 350, 120, 250, &closeAngleRight, 0, 30);
			
				// closure time span trackbar
				cvui::printf(config, 50, 260, 0.5, 0xff0000, "Glottis closure (in ms)");
				cvui::trackbar(config, 50, 290, 550, &closureDuration, 100, 2000);
			
				cvui::checkbox(config, 50, 30, "Use Gaussian blur", &use_gaussian);
				cvui::checkbox(config, 350, 30, "Rotate image", &use_rotate);
				cvui::checkbox(config, 50, 50, "Show live view", &use_live);
			
				break;
			
			// LAR parameters set, droplet detection process starting
			case 1:
				if (cvui::button(config, 280, 450, "Stop"))
				{
					// reset all parameters
					start = 0;
					lastFrame.release();
					backgroundFrame.release();
					vForeground.release();
					pHoughSearch.release();
					result. release();
					gpioTerminate();
					dropletDetected = false;
					destroyWindow("Live");
				}
				
				cvui::printf(config, 50, 140, 0.5, 0x909090, "LAR1 latency: %d ms", LAR1);
				cvui::printf(config, 350, 140, 0.5, 0x909090, "LAR2 latency: %d ms", LAR2);
				cvui::printf(config, 50, 80, 0.5, 0x909090, "Closing angle left: %d degrees", closeAngleLeft);
				cvui::printf(config, 350, 80, 0.5, 0x909090, "Closing angle right: %d degrees", closeAngleRight);
				cvui::printf(config, 50, 280, 0.5, 0x909090, "Glottis closure: %d ms", closureDuration);
				
				break;
			}
			
			// record frame
			cap >> frame;

			// check: frame recorded?
			if (frame.empty())
			{
				cerr << "ERROR: No frame received!" << endl;
				break;
			}
			
			if (start == 0) 
			{
				// enlarge live view
				resize(frame, frameZoom, Size(640,480), 0, 0, INTER_LINEAR);
				
				if (use_rotate) flip(frameZoom, frameZoom, -1);
				
				imshow("Live view",frameZoom);
			}
			
			if (start == 1)
			{
				// initialization for lastFrame and backgroundFrame
				if (lastFrame.empty()) cap >> lastFrame;
				if (backgroundFrame.empty()) cap >> backgroundFrame;
				
				// grayscale conversion
				if (backgroundFrame.channels() != 1)
				{
				cv::cvtColor(backgroundFrame, backgroundFrame, COLOR_BGR2GRAY);
				}
							
				result = frame;
			
				if (use_live) 
				{
					if (use_rotate)
					{
						flip(result, resultRotate, -1);
						imshow ("Live", resultRotate);
					}
					else imshow("Live", result);
				}
				
				// convert current and last frame in grayscale images
				cv::cvtColor(frame, frame, COLOR_BGR2GRAY);
				cv::cvtColor(lastFrame, lastFrame, COLOR_BGR2GRAY);
			
				// Gaussian blur
				if (use_gaussian)
				{
					GaussianBlur(frame, frame, Size(3, 3), 0, 0);
					GaussianBlur(lastFrame, lastFrame, Size(3, 3), 0, 0);
					if (backgroundUpdate)
					{
						GaussianBlur(backgroundFrame, backgroundFrame, Size(3, 3), 0, 0);
						backgroundUpdate = false;
					}
				}			
								
				// difference image analysis for motion detection
				absdiff(lastFrame, frame, vForeground);
								
				// difference image binarization
				threshold(vForeground, vForeground, 15, 255, THRESH_BINARY);
				
				resize(vForeground, vForeground, Size(320, 240), 0, 0);
			
				// count white pixels in difference image
				nFoundWhites = countNonZero(vForeground);
				// cout << nFoundWhites << endl;
			
				// if more than nMinWhitePixels found -> motion detected
				if ((nFoundWhites > nMinWhitePixels) && (!motionDetected))
				{
					motionDetected = true;
					dropletDetected = false;
				}
				
				// if less than nMinWhitePixels found and last state was "motion found" -> droplet impact detected
				if ((nFoundWhites < nMinWhitePixels) && (motionDetected))
				{
					dropletDetected = true;
					motionDetected = false;				
				}
				
				
				// if no droplet impact and no motion detected: update background with current frame
				if ((!dropletDetected) && (!motionDetected))
				{
					cap >> backgroundFrame;
					backgroundUpdate = true;
				}
				
								
				// if droplet impact detected: execute blob detection				
				if(dropletDetected)
				{
											
					// convert frames into 8 bit grayscale images
					
					frame.convertTo(frame, CV_8UC1);
					backgroundFrame.convertTo(backgroundFrame, CV_8UC1);
					
					// enhance histogram
					equalizeHist(frame,frame);
					equalizeHist(backgroundFrame,backgroundFrame);
					
					// isolate droplet by background subtraction
					absdiff(frame, backgroundFrame, pHoughSearch);	
					
					if(keypoints.size() >= 1) 
					{
						gettimeofday(&t3, NULL);
						t4 = ((t3.tv_sec * 1000000) + t3.tv_usec) - ((t1.tv_sec * 1000000) + t1.tv_usec);
						
						// save x/y coordinates, diameter and radius of the first blob found 
						X = keypoints[0].pt.x; 
						Y = keypoints[0].pt.y;
						D = keypoints[0].size;		
						R = D/2;
					
						// cout << "X,Y,D,R: " << X << " " << Y << " " << D << " " << R << endl;
					
						if (use_live)
						{
							// draw centroid of found blob
							circle(result, Point(X, Y), 1, Scalar(0, 0, 255), 3, LINE_AA);
							// draw circumference of found blob
							circle(result, Point(X, Y), R, Scalar(0, 0, 255), 3, LINE_AA);
							if (use_rotate) 
							{
								flip(result, resultRotate, -1);
								imshow ("Live", resultRotate);
							}
							else imshow ("Live", result);
						}
					
					
					// calculate servomotor actuation parameters
					if (vertLabel[2]==1)
					{
						if (X>b[2])
						{
							motorLAR1 = MotorPinRight;
							motorLAR2 = MotorPinLeft;
							motorPosLAR1 = motorPos1 + closeAngleRight * 220 / 30;
							motorPosLAR2 = motorPos2 - closeAngleLeft * 220 / 30;
						}
						else
						{
							motorLAR1 = MotorPinLeft;
							motorLAR2 = MotorPinRight;
							motorPosLAR1 = motorPos2 - closeAngleLeft * 220 / 30;
							motorPosLAR2 = motorPos1 + closeAngleRight * 220 / 30;
						}
					}
					

					if (X>((Y-b[2])/a[2]))
					{
						motorLAR1 = MotorPinRight;
						motorLAR2 = MotorPinLeft;
						motorPosLAR1 = motorPos1 + closeAngleRight * 220 / 30;
						motorPosLAR2 = motorPos2 - closeAngleLeft * 220 / 30;
					}
					else
					{
						motorLAR1 = MotorPinLeft;
						motorLAR2 = MotorPinRight;
						motorPosLAR1 = motorPos2 - closeAngleLeft * 220 / 30;
						motorPosLAR2 = motorPos1 + closeAngleRight * 220 / 30;
					}
												
					gettimeofday(&t2, NULL);
					delay(LAR1);
					gpioServo (motorLAR1, motorPosLAR1);
					delay(LAR2 - LAR1);
					gpioServo (motorLAR2, motorPosLAR2);
					delay(closureDuration);
					gpioServo (MotorPinLeft, motorPos2);
					gpioServo (MotorPinRight, motorPos1);
					delay(500);
								
					
					// show result of droplet detection
					
					// draw centroid of found blob
					circle(result, Point(X, Y), 1, Scalar(0, 0, 255), 3, LINE_AA);
					// draw circumference of found blob
					circle(result, Point(X, Y), R, Scalar(0, 0, 255), 3, LINE_AA);
					
					if (vertLabel[2]==1)
					{
						pt1.x = cvRound(b[2]);
						pt1.y = 0;
						pt2.x = cvRound(b[2]);
						pt2.y = 240;
					}	
					else
					{
						pt1.y = 0;
						pt1.x = cvRound(-b[2]/a[2]);
						pt2.y = 240;
						pt2.x = cvRound((pt2.y-b[2])/a[2]);
					}
					
					line(result, pt1, pt2, Scalar(0,255,0), 1, CV_AA);
					
					if (use_rotate) flip(result, result, -1);
					
					imshow("Result", result);
					imwrite("Result.png", result);
				
					// LAR simulation finished, reset system
					start = 0;
					dropletDetected = false;
					lastFrame.release();
					backgroundFrame.release();
					vForeground.release();
					pHoughSearch.release();
					result.release();
					elapsedTime = ((t2.tv_sec * 1000000) + t2.tv_usec) - ((t1.tv_sec * 1000000) + t1.tv_usec) + t4;
					destroyWindow("Live");
					
					}
				}
				

				// current frame becomes last frame in next iteration
				cap >> lastFrame;
			}
			break;
		}
		
		cvui::update();
		cv::imshow(WINDOW_NAME, config);
		waitKey(3);

	}
	
	cap.release();
	lastFrame.release();
	backgroundFrame.release();
	vForeground.release();
	pHoughSearch.release();
	result.release();
	gpioTerminate();
	destroyAllWindows();
	return 0;
}


Mat RegionGrow(Mat src, Point2i pt)
{
	Point2i ptGrowing;
	int nGrowLabel[src.rows][src.cols] = {0};
	int nSrcValue = 0;
	double nCurValue = 0;
	Mat matDst = Mat::zeros(src.size(), CV_8UC1);
  
	int DIR[4][2]= {{0,-1},{1,0},{0,1},{-1,0}};
	vector<Point2i> vcGrowPt;
	vector<double>pixelValue;
	vcGrowPt.push_back(pt);
	matDst.at<uchar>(pt.y, pt.x) = 255;
	nSrcValue = src.at<uchar>(pt.y, pt.x);
  
	while(!vcGrowPt.empty())
	{
		bool edgeLabel = false;
		pt = vcGrowPt.back();
		vcGrowPt.pop_back();
		for (int i = 0; i < 4; i++)
		{
			ptGrowing.x = pt.x + DIR[i][0];
			ptGrowing.y = pt.y + DIR[i][1];
			if ((ptGrowing.x < 0) || (ptGrowing.y < 0) || (ptGrowing.x > (src.cols - 1)) || (ptGrowing.y > (src.rows - 1)))
				continue;
			if (nGrowLabel[ptGrowing.y][ptGrowing.x] == 0)
			{
				nCurValue = src.at<uchar>(ptGrowing.y, ptGrowing.x);
				if (nCurValue != nSrcValue)
					edgeLabel = true;
			}
		}
 
		if(edgeLabel) continue;
		for (int i = 0; i < 4; i++)
		{
			ptGrowing.x = pt.x + DIR[i][0];
			ptGrowing.y = pt.y + DIR[i][1];
     
			if ((ptGrowing.x < 0) || (ptGrowing.y < 0) || (ptGrowing.x > (src.cols - 1)) || (ptGrowing.y > (src.rows - 1)))
				continue;
     
			if (nGrowLabel[ptGrowing.y][ptGrowing.x] == 0)
			{
				nCurValue = src.at<uchar>(ptGrowing.y, ptGrowing.x);
				if (nCurValue == nSrcValue)
				{
					matDst.at<uchar>(ptGrowing.y, ptGrowing.x) = 255;
					vcGrowPt.push_back(ptGrowing);
					nGrowLabel[ptGrowing.y][ptGrowing.x] = 1;
				}
			}
		}
	}
  return matDst.clone();
}

void on_MouseHandle(int event, int x, int y, int flags, void *ustc)
{

	switch(event)
	{
	case EVENT_LBUTTONDOWN:
		{
			pt = Point(x, y);
			RG = true;
			break;
		}
	case EVENT_LBUTTONUP:
		{
			RG = false;
			break;
		}
	}
}
