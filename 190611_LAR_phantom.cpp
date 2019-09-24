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

// Globale Variablen
bool RG = false;
Point pt;


//verwendete Funktionen
void Census(Mat inputImg, Mat outputImg);
void HammingDistance(Mat inputImg1, Mat inputImg2, Mat dist);
void on_MouseHandle(int event, int x, int y, int flags, void *ustc);
Mat RegionGrow(Mat src, Point2i pt/*, int th*/);

// Programmstart

int main(int argc, char *argv[])
{
	// Flags
	bool use_gaussian = true;
	bool use_census = false;
	bool use_morph = false;
	bool use_rotate = true;
	bool use_live = false;
	bool motionDetected = false;
	bool backgroundUpdate = true;
	
	// Parameter-Setup Blob Detector
	cv::SimpleBlobDetector::Params params;
	params.minDistBetweenBlobs = 100.0f;
	params.filterByInertia = false;
	params.filterByConvexity = false;
	params.filterByColor = true;
	params.blobColor = 255; // extract dark blobs (255 for light blobs)
	params.filterByCircularity = false;
	params.filterByArea = false;
	params.minArea = 10.0f;
	params.maxArea = 100.0f;
		
	// Vektor zur Speicherung detektierter Blobs
	vector<cv::KeyPoint> keypoints;
	
	// Floats zur Speicherung der Blob-Koordinaten, -Durchmesser und -Radien
	float X, Y, D, R;

	// Parameter der Hough-Kreiserkennung
	int nParam1 = 60, nParam2 = 30, nMinRadius = 5, nMaxRadius = 30, nFoundWhites = 0; 
	vector<Vec3f> vCircles;
	const int nMinWhitePixels = 10;
	
	// Parameter der HoughLines
	float theta[3];
	float a[3], b[3];
	bool vertLable[3];
	
	// Parameter für Zeitmessung
	struct timeval t1, t2, t3;
	long long elapsedTime = 0, t4 = 0;

	// Parameter des LAR
	int closeAngleLeft = 30;
	int closeAngleRight = 30;
	int closureDuration = 200;
	int LAR1 = 100;
	int LAR2 = 100;
	int motorLAR1, motorLAR2;

	// Rotationsbereich der Servos
	int motorPos1 = 1080, motorPos2 = 1250;

	// Endposition der Servos
	int motorPosLAR1, motorPosLAR2;

	// Signalausgänge definieren
	const int MotorPinLeft = 19;
	const int MotorPinRight = 18;
	
											
	// Mittelinie der Stimmlippen zeichnen
	cv::Point2f pt1, pt2;
	
	Mat frame, frameZoom, backgroundFrame, lastFrame, result, resultRotate;
	
	// Mat-Struktur und Vector für Region Growing
	Mat edge, GrowingBild1, GrowingBild2;
	vector<Vec4i> linesP;
	
	// Deklaration: Mat-Strukturen für weitere Bildverarbeitung
	Mat vForeground, pHoughSearch;
	Mat dist(240, 320,CV_8UC1);
	Mat frameCensus(240, 320,CV_8UC1);
	Mat lastFrameCensus(240, 320,CV_8UC1);
	
	// Deklaration: Initialisierung der GUI
	Mat config = Mat(500, 650, CV_8UC3);
	
	// Flags
	int state = 0;
	bool start = 0;
	bool dropletDetected = false;
		
	// Kamera öffnen
	VideoCapture cap(0);
	// Prüfung: Kamera geöffnet?
	if (!cap.isOpened())
	{
		cerr << "ERROR: Kamera kann nicht geöffnet werden." << endl;
		return 0;
	}
	
	// Kamera einstellen
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	cap.set(CV_CAP_PROP_FPS, 120);
	
	// Initialisierung des GUI-Fensters
	cv::namedWindow(WINDOW_NAME);
	cvui::init(WINDOW_NAME);	
	
	cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
	
	while (1)
	{	
		// Button "Beenden" gedrückt?
		config = cv::Scalar(49, 52, 49);
		if (cvui::button(config, 500, 450, "Beenden")) {
			return 0;
		}
		
		switch(state)
		{
			
		// Stimmlippe erkennen und Kantenlage berechnen
		case 0:
			cap >> frame;
			
			// Angeklickten Punkt erkennen
			namedWindow("Live", WINDOW_NORMAL);
			setMouseCallback("Live", on_MouseHandle);
			imshow("Live", frame);
			
			if (RG == 1)
			{
				cv::cvtColor(frame, frame, COLOR_BGR2GRAY);
				
				// Vorverarbeitung
				GaussianBlur(frame, frame, Size(5, 5), 0, 0);
				equalizeHist(frame,frame);
				Canny(frame, frame, 50, 150, 3);
				
				// Region Growing
				GrowingBild1 = RegionGrow(frame, pt);
				imshow("RG", GrowingBild1);
				
				// Kantenerkennung
				Canny(GrowingBild1, edge, 50, 200, 3);
				
				// Hough Line Detection
				HoughLinesP(edge, linesP, 1, CV_PI/180, 30, 30, 10);
				
				Vec4f l=linesP[0];
				
				// Parameter der Linie berechnen
				if (l[2]==l[0]) 
				{
					vertLable[0] = 1;
					b[0] = l[2];
					theta[0] = pi/2;
				}
				else
				{	
					vertLable[0] = 0;
					a[0] = (l[3]-l[1])/(l[2]-l[0]);
					b[0] = l[3]-a[0]*l[2];
					theta[0] = atan(a[0]);
				}
			}
			
			if (!GrowingBild1.empty()) 
			{
				cvui::printf(config, 50, 280, 0.5, 0xffffff, "Falls Stimmlippe gut erkannt, klicken Sie \"Weiter\"!");
				if (cvui::button(config, 280, 450, "Weiter")) 
				{
					state = 1;
					
					if (theta[0] < 0) theta[0] = pi+theta[0];
					
					destroyWindow("RG");
				}
			}
			
			cvui::printf(config, 50, 200, 0.5, 0xffffff, "Bitte klicken Sie eine Stimmlippe im Bild an!");
			break;
		
		// Die andere Stimmlippe erkennen und Kantenlage berechnen
		case 1:
		
			cap >> frame;
			
			imshow("Live", frame);
			
			if (RG == 1)
			{
				cv::cvtColor(frame, frame, COLOR_BGR2GRAY);
				
				// Vorverarbeitung
				GaussianBlur(frame, frame, Size(5, 5), 0, 0);
				equalizeHist(frame,frame);
				Canny(frame, frame, 50, 150, 3);
				
				// Regiongrowing
				GrowingBild2 = RegionGrow(frame, pt);
				imshow("RG", GrowingBild2);
				
				// Kantenerkennung
				Canny(GrowingBild2, edge, 50, 200, 3);
				
				//Hough Line Detection
				HoughLinesP(edge, linesP, 1, CV_PI/180, 30, 30, 10);
				
				Vec4f l=linesP[0];
				
				// Parameter der Linie berechnen
				if (l[2]==l[0]) 
				{
					vertLable[1] = 1;
					b[1] = l[2];
					theta[1] = pi/2;
				}
				else
				{
					vertLable[1] = 0;
					a[1] = (l[3]-l[1])/(l[2]-l[0]);
					b[1] = l[3]-a[1]*l[2];
					theta[1] = atan(a[1]);
				}
			}
			
			if (!GrowingBild2.empty()) 
			{
				cvui::printf(config, 50, 280, 0.5, 0xffffff, "Falls fertig, klicken Sie \"Weiter\"!");
				if (cvui::button(config, 280, 450, "Weiter"))
				{
					state = 2;
					
					if (theta[1] < 0) theta[1] = pi+theta[1];
					
					theta[2] = (theta[0]+theta[1])/2;
					
					if (round(theta[2]/pi*180)==90)
					{
						vertLable[2] = 1;
						b[2] = (b[0]-b[1])/(a[1]-a[0]);
						theta[3] = pi/2;
					}
					
					else
					
					{
						vertLable[2] = 0;
						a[2] = tan(theta[2]);
						b[2] = (a[2]-a[1])*(b[1]-b[0])/(a[1]-a[0])+b[1];
					}

					destroyWindow("RG");
					destroyWindow("Live");
				}
			}
			
			cvui::printf(config, 50, 200, 0.5, 0xffffff, "Bitte klicken Sie die andere Stimmlippe im Bild an!");
			break;
			
		case 2:
			gettimeofday(&t1, NULL);

			switch(start)
			{
			// Prozess noch nicht gestartet, Parameter sind einstellbar
			case 0:
				if (LAR2 >= LAR1)
				{
					if (cvui::button(config, 280, 450, "Start")) 
					{
						start = 1;
						if (gpioInitialise() < 0) return -1;
						destroyWindow("Livebild");
						destroyWindow("Ergebnis");
						destroyWindow("Differenzbild");
						destroyWindow("Frame");
						destroyWindow("BackgroundFrame");
					}
					if (elapsedTime != 0)
						cvui::printf(config, 50, 340, 0.5, 0xff0000, "Dauer der Erkennung: %11d us", elapsedTime);
				}
				
				else
					cvui::printf(config, 50, 340, 0.5, 0xff0000, "Achtung: LAR2 > LAR1 muss gelten!");
				
				// Erneute Segmentierung
				if (cvui::button(config, 50, 450, "Segmentierung"))
				{
					state = 0;
					GrowingBild1.release();
					GrowingBild2.release();
					destroyWindow("Livebild");
					continue;
				}
			
				// Trackbar für Latenz
				cvui::printf(config, 50, 170, 0.5, 0xff0000, "LAR1-Latenz (in ms)");
				cvui::printf(config, 350, 170, 0.5, 0xff0000, "LAR2-Latenz (in ms)");
				cvui::trackbar(config, 50, 200, 250, &LAR1, 50, 200);
				cvui::trackbar(config, 350, 200, 250, &LAR2, 50, 200);
			
				// Trackbar für Schlusswinkel
				cvui::printf(config, 50, 90, 0.5, 0xff0000, "Schlusswinkel links (in Grad)");
				cvui::printf(config, 350, 90, 0.5, 0xff0000, "Schlusswinkel rechts (in Grad)");
				cvui::trackbar(config, 50, 120, 250, &closeAngleLeft, 0, 30);
				cvui::trackbar(config, 350, 120, 250, &closeAngleRight, 0, 30);
			
				// Trackbar für Schlusszeit
				cvui::printf(config, 50, 260, 0.5, 0xff0000, "Glottis-Schlusszeit (in ms)");
				cvui::trackbar(config, 50, 290, 550, &closureDuration, 100, 2000);
			
				cvui::checkbox(config, 50, 10, "Census-Transform verwenden", &use_census);
				cvui::checkbox(config, 50, 30, "Gaussschen Weichzeichner verwenden", &use_gaussian);
				cvui::checkbox(config, 350, 10, "Morphologische Operatoren verwenden", &use_morph);
				cvui::checkbox(config, 350, 30, "Bild um 180 Grad drehen", &use_rotate);
				cvui::checkbox(config, 50, 50, "Livebild anzeigen", &use_live);
			
				break;
			
			// Prozess startet, Parameter sind festgelegt
			case 1:
				if (cvui::button(config, 280, 450, "Stop"))
				{
					// Alle Parameter zurücksetzen
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
				
				cvui::printf(config, 50, 140, 0.5, 0x909090, "LAR1-Latenz: %d ms", LAR1);
				cvui::printf(config, 350, 140, 0.5, 0x909090, "LAR2-Latenz: %d ms", LAR2);
				cvui::printf(config, 50, 80, 0.5, 0x909090, "Schlusswinkel links: %d Grad", closeAngleLeft);
				cvui::printf(config, 350, 80, 0.5, 0x909090, "Schlusswinkel rechts: %d Grad", closeAngleRight);
				cvui::printf(config, 50, 280, 0.5, 0x909090, "Glottis-Schlusszeit: %d ms", closureDuration);
				
				break;
			}
			
			// Frame aufnehmen		
			cap >> frame;

			// Prüfung: Bilder in Frame?
			if (frame.empty())
			{
				cerr << "FEHLER: Kein Kamera-Frame empfangen!" << endl;
				break;
			}
			
			if (start == 0) 
			{
				// Livebild vergrößen
				resize(frame, frameZoom, Size(640,480), 0, 0, INTER_LINEAR);
				
				if (use_rotate) flip(frameZoom, frameZoom, -1);
				
				imshow("Livebild",frameZoom);
			}
			
			if (start == 1)
			{
				// Initialisierung für lastFrame und backgroundFrame
				if (lastFrame.empty()) cap >> lastFrame;
				if (backgroundFrame.empty()) cap >> backgroundFrame;
				
				// Graustufenkonvertierung
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
				
				// Aktuelles und letztes Frame in Graustufenbilder konvertieren
				cv::cvtColor(frame, frame, COLOR_BGR2GRAY);
				cv::cvtColor(lastFrame, lastFrame, COLOR_BGR2GRAY);
			
				// Gaussscher Weichzeichner		
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
								
				// Zwei angrenzende Frames werden subtrahiert, um Bewegung im Bild zu erkennen
				absdiff(lastFrame, frame, vForeground);
								
				// Binarisierung des Differenzbildes
				threshold(vForeground, vForeground, 15, 255, THRESH_BINARY);
				//imwrite("Binaerbild.jpg", vForeground);
						
				// Morphologisches Closing zur Eliminierung vereinzelter Fehlpixel
				if (use_morph)
				{
					morphologyEx(vForeground, vForeground, MORPH_OPEN,
						getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1)), Point(-1, -1), 
						1, BORDER_CONSTANT, 0);
				}
				
				resize(vForeground, vForeground, Size(320, 240), 0, 0);
			
				if (use_census)
				{
					// Census-Transformation und Hamming-Distance berechnen
					
					Census(frame, frameCensus);
					if (lastFrameCensus.empty()) Census(lastFrame, lastFrameCensus);
					
					HammingDistance(frameCensus, lastFrameCensus, dist);
				
					// Ergebnis der Vordergrund-Erkennung durch Census-Transformation ausgeben
					vForeground = dist.mul(vForeground);
				}
			
				// Zählen der weißen Pixel im Differenzbild
				nFoundWhites = countNonZero(vForeground);
			
				// Falls mehr als nMinWhitePixels weiße Pixel im Bild --> Bewegung im Bild gefunden!
				if ((nFoundWhites > nMinWhitePixels) && (!motionDetected))
				{
					motionDetected = true;
					dropletDetected = false;
				}
				
				// Falls weniger als nMinWhitePixels weiße Pixel im Bild und vorheriger Zustand war "Bewegung gefunden" --> Tropfenaufprall!
				if ((nFoundWhites < nMinWhitePixels) && (motionDetected))
				{
					dropletDetected = true;
					motionDetected = false;				
				}
				
				
				// Falls kein Tropfenaufprall und keine Bewegung gefunden: aktuelles Frame als Hintergrund-Frame speichern
				if ((!dropletDetected) && (!motionDetected))
				{
					cap >> backgroundFrame;
					backgroundUpdate = true;
				}
				
								
				// Falls Tropfenaufprall gefunden: Blob Detection/Kreiserkennung durchführen				
				if(dropletDetected)
				{
											
					// Frames für die Hough-Kreiserkennung in 8-Bit-Graustufenbild konvertieren
					
					frame.convertTo(frame, CV_8UC1);
					backgroundFrame.convertTo(backgroundFrame, CV_8UC1);
					
					equalizeHist(frame,frame);
					equalizeHist(backgroundFrame,backgroundFrame);
					
					// Subtraktion des Hintergrunds, um Tropfen zu isolieren
					absdiff(frame, backgroundFrame, pHoughSearch);
									
					// Blob Detection zur Tropfen-Ortung im Aufprallframe
					detector->detect(pHoughSearch, keypoints);			
					
					if(keypoints.size() >= 1) 
					{
						gettimeofday(&t3, NULL);
						t4 = ((t3.tv_sec * 1000000) + t3.tv_usec) - ((t1.tv_sec * 1000000) + t1.tv_usec);
						
						// x/y-Koordinaten, Durchmesser und Radius des ersten gefundenen Blobs abspeichern 
						X = keypoints[0].pt.x; 
						Y = keypoints[0].pt.y;
						D = keypoints[0].size;		
						R = D/2;
					
						// cout << "X,Y,D,R: " << X << " " << Y << " " << D << " " << R << endl;
					
						if (use_live)
						{
							// Kreismittelpunkt des gefundenen Kreises zeichnen
							circle(result, Point(X, Y), 1, Scalar(0, 0, 255), 3, LINE_AA);
							// Umfang des gefundenen Kreises zeichnen
							circle(result, Point(X, Y), R, Scalar(0, 0, 255), 3, LINE_AA);
							if (use_rotate) 
							{
								flip(result, resultRotate, -1);
								imshow ("Live", resultRotate);
							}
							else imshow ("Live", result);
						}
					
					
					// Parameter für Servosteuerung berechnen
					if (vertLable[2]==1)
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
								
					
					// Ergebnisbild zeigen	
					
					// Kreismittelpunkt der gefundenen Kreise zeichnen
					circle(result, Point(X, Y), 1, Scalar(0, 0, 255), 3, LINE_AA);
					// Umfang der gefundenen Kreise zeichnen
					circle(result, Point(X, Y), R, Scalar(0, 0, 255), 3, LINE_AA);
					
					if (vertLable[2]==1)
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
					
					imshow("Ergebnis", result);
					imwrite("Ergebnisbild.png", result);
				
					// LAR-Bewegung wurde simuliert, nun Reset
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
				

				// Aktuelles Frame wird lastFrame
				cap >> lastFrame;
				if(use_census) lastFrameCensus = frameCensus;
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

// Berechnung der Census-Transformation
void Census(Mat inputImg, Mat outputImg)
{
	Size imgSize = inputImg.size();
	unsigned int census = 0;
	unsigned int bit = 0;
	int i,j,x,y;
	int shiftCount = 0;
	for (x = 1; x < imgSize.height - 1; x++)
	{
		for (y = 1; y < imgSize.width - 1; y++)
		{
			census = 0;
			shiftCount = 0;
			for (i = x - 1; i <= x + 1; i++)
			{
				for (j = y - 1; j <= y + 1; j++)
				{
					if (shiftCount != 4)
					{
						census = census << 1;
						if (inputImg.at<uchar>(i,j) < inputImg.at<uchar>(x,y))
						bit = 1;
						else
						bit = 0;
						census = census + bit;
					}
					shiftCount ++;
				}
			}
			outputImg.at<uchar>(x,y) = census;
		}
	}
}

// Berechnung der Hamming-Distanz
void HammingDistance(Mat inputImg1, Mat inputImg2, Mat dist)
{
	
	int temp, num;
	for (int i = 0; i < inputImg1.rows; i++)
	{
		for (int j = 0; j < inputImg1.cols; j++)
		{
		num = 0;
		temp = inputImg1.at<uchar>(i,j)^inputImg2.at<uchar>(i,j);
		while(temp)
			{
				if (temp % 2) num++;
				temp /= 2;
			}
		dist.at<uchar>(i,j) = (num>1);
		}
	}
}

Mat RegionGrow(Mat src, Point2i pt)
{
	Point2i ptGrowing;
	int nGrowLable[src.rows][src.cols] = {0};
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
		bool edgeLable = false;
		pt = vcGrowPt.back();
		vcGrowPt.pop_back();
		for (int i = 0; i < 4; i++)
		{
			ptGrowing.x = pt.x + DIR[i][0];
			ptGrowing.y = pt.y + DIR[i][1];
			if ((ptGrowing.x < 0) || (ptGrowing.y < 0) || (ptGrowing.x > (src.cols - 1)) || (ptGrowing.y > (src.rows - 1)))
				continue;
			if (nGrowLable[ptGrowing.y][ptGrowing.x] == 0)
			{
				nCurValue = src.at<uchar>(ptGrowing.y, ptGrowing.x);
				if (nCurValue != nSrcValue)
					edgeLable = true;
			}
		}
 
		if(edgeLable) continue;
		for (int i = 0; i < 4; i++)
		{
			ptGrowing.x = pt.x + DIR[i][0];
			ptGrowing.y = pt.y + DIR[i][1];
     
			if ((ptGrowing.x < 0) || (ptGrowing.y < 0) || (ptGrowing.x > (src.cols - 1)) || (ptGrowing.y > (src.rows - 1)))
				continue;
     
			if (nGrowLable[ptGrowing.y][ptGrowing.x] == 0)
			{
				nCurValue = src.at<uchar>(ptGrowing.y, ptGrowing.x);
				if (nCurValue == nSrcValue)
				{
					matDst.at<uchar>(ptGrowing.y, ptGrowing.x) = 255;
					vcGrowPt.push_back(ptGrowing);
					nGrowLable[ptGrowing.y][ptGrowing.x] = 1;
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
