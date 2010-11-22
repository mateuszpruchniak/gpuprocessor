#include <oclUtils.h>
#include <iostream>
#include "cv.h"
#include "cxmisc.h"
#include "highgui.h"
#include <vector>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <ctype.h>
#include <time.h>
#include "GPUTransferManager.h"
#include "GPUImageProcessor.h"
#include "SobelFilter.h"
#include "LUTFilter.h"
#include "MedianFilter.h"
#include "MeanFilter.h"
#include "MeanVariableCentralPointFilter.h"
#include "ErodeFilter.h"
#include "DilateFilter.h"
#include "MinFilter.h"
#include "MaxFilter.h"
#include "CloseFilter.h"
#include "OpenFilter.h"
#include "PrewittFilter.h"
#include "RobertsFilter.h"
#include "LaplaceFilter.h"
#include "CornerDetectionFilter.h"
#include "BinarizationFilter.h"


using namespace std;




int main(int argc, const char** argv)
{

//
//    // median CPU ---------------------------------------------------
//	for(int j = 1 ; j < 12 ; j++ )
//	{
//
//
//		IplImage* img = cvLoadImage("./car.jpg");
//		IplImage* img2 = cvLoadImage("./car.jpg");
//		IplImage* hsv = cvCreateImage( cvGetSize(img2), 8, 3 ) ;
//		
//		
//		IplImage* image4 = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 3);
//
//		IplImage *newImage = cvCreateImage( cvSize(img->width * 0.25 * j ,img->height* 0.25 * j), img->depth, img->nChannels );
//		IplImage *newImage2 = cvCreateImage( cvSize(img->width * 0.25 * j ,img->height* 0.25 * j), img->depth, img->nChannels );
//		cvResize(img, newImage);
//		//IplImage *img = cvLoadImage("gory.jpg");
//		//IplImage* out = cvCreateImage( cvGetSize(img), IPL_DEPTH_8U, 3 );
//		IplImage *sobx = cvCreateImage(cvGetSize(newImage),32,1);
//		IplImage *soby = cvCreateImage(cvGetSize(newImage),32,1);
//		IplImage *sobxy = cvCreateImage(cvGetSize(newImage),32,1);
//		IplImage *sobxy1 = cvCreateImage(cvGetSize(newImage),32,1);
//		IplImage *dst = cvCreateImage( cvSize( newImage->width, newImage->height ), IPL_DEPTH_8U, 1 );
//
//		char lut[256];
//		for(int i = 0 ; i < 256 ; ++i )
//		{
//			lut[i] = (char)(255-i);
//		}
//
//		CvMat* lut_mat = cvCreateMatHeader( 1, 256, CV_8UC1 );
//		cvSetData( lut_mat, lut, 0 );
//		// Create image to find chessboard corners
//		IplImage* gray = cvCreateImage(cvSize( newImage->width, newImage->height ), IPL_DEPTH_8U,1);
// 
//
//		int w= 11, h=11; 
//
//CvMat *mask= cvCreateMat(h, w, CV_32FC1);
//for (int y= 0; y<h; y++)
//for (int x= 0; x<w; x++)
//cvSet2D(mask, y, x, cvScalar(1));
//
//
//
//		clock_t start, finish;
//		double duration = 0;
//		start = clock();
//		for(int i = 0 ; i < 100 ; i++)
//		{
//			//convert image to grayscale
//			//cvCvtColor(newImage, gray, CV_RGB2GRAY);
//			
//			//cvSmooth( newImage, newImage2, CV_BLUR, 110, 110);
//			
//			cvFilter2D(newImage, newImage, mask);
//
//			//cvLUT( gray, gray, lut_mat );
//
//			//cvSobel(gray, sobx, 0, 1, 3);
//			//cvSobel(gray, soby, 1, 0, 3);
//			//cvCartToPolar(sobx, soby , sobxy, sobxy1, 1);
//			//cvConvert(sobxy,dst);
//			
//			//cvErode(newImage,newImage,NULL,1);
//			//cvDilate(newImage,newImage,NULL,1);
//		}
//		finish = clock();
//		duration = (double)(finish - start) / CLOCKS_PER_SEC;
//		cout << "Czas wykonania 100 iteracji: " << endl;
//		cout << duration << endl;
//		cout << "rozmiar: " << endl;
//		cout << newImage->width <<"x"<< newImage->height << endl;
//		cout << "-------------------------\n\n" << endl;
//
//		//cvNamedWindow("sobelCPU", CV_WINDOW_AUTOSIZE); 
//		//cvShowImage("sobelCPU", newImage );
//		//cvWaitKey(2);
//
//	}
//

  


    //  GPU -----------------------------------------------------

 //   
 //   
	double avg = 0;
	int i = 0;
	for(i = 0; i < 1; i++ )
	{
		int j = 4;

		IplImage* img = cvLoadImage("./car.jpg");
		IplImage* img2 = cvLoadImage("./car.jpg");
		IplImage* hsv = cvCreateImage( cvGetSize(img2), 8, 3 ) ;
		

		IplImage *newImage = cvCreateImage( cvSize(img->width * 0.25 * j ,img->height* 0.25 * j), img->depth, img->nChannels );

		cvResize(img, newImage);
		GPUImageProcessor* GPU = new GPUImageProcessor(newImage->width,newImage->height,newImage->nChannels);

		int lut[256];
		for(int i = 0 ; i < 256 ; ++i )
		{
			lut[i] = 255-i;
		}

	    int* maskH = new int[9];
		for(int i = 0 ; i < 9 ; ++i )
		{
			maskH[i] = 1;
		}

		maskH[4] = 0;

		int* maskV = new int[9];
		for(int i = 0 ; i < 9 ; ++i )
		{
			maskV[i] = 1;
		}
		
		GPU->AddProcessing( new MeanVariableCentralPointFilter(GPU->GPUContext,GPU->Transfer,0) );
		
		//GPU->AddProcessing( new MeanFilter(GPU->GPUContext,GPU->Transfer) );
		//GPU->AddProcessing( new MedianFilter(GPU->GPUContext,GPU->Transfer) );
		//GPU->AddProcessing( new DilateFilter(GPU->GPUContext,transf) );
		//GPU->AddProcessing( new ErodeFilter(GPU->GPUContext,GPU->Transfer) );
		
		//GPU->AddProcessing( new LUTFilter(GPU->GPUContext,GPU->Transfer,lut) );
		//GPU->AddProcessing( new SobelFilter(GPU->GPUContext,GPU->Transfer) );
		//GPU->AddProcessing( new MinFilter(GPU->GPUContext,GPU->Transfer) );
		////GPU->AddProcessing( new MaxFilter(GPU->GPUContext,GPU->Transfer) );
		//GPU->AddProcessing( new OpenFilter(GPU->GPUContext,GPU->Transfer) );
		//GPU->AddProcessing( new CloseFilter(GPU->GPUContext,transf) );
		//GPU->AddProcessing( new PrewittFilter(GPU->GPUContext,GPU->Transfer) );
		//GPU->AddProcessing( new RobertsFilter(GPU->GPUContext,GPU->Transfer) );
		//GPU->AddProcessing( new LaplaceFilter(GPU->GPUContext,GPU->Transfer) );
		//GPU->AddProcessing( new CornerDetectionFilter(GPU->GPUContext,transf) );
		//GPU->AddProcessing( new RGB2YUV(GPU->GPUContext,GPU->Transfer) );
		//GPU->AddProcessing( new BinarizationFilter(GPU->GPUContext,GPU->Transfer,120) );

		cout << ((char*)newImage->imageData)[0] << endl;
		clock_t start, finish;
		double duration = 0;
		int i = 0;
		start = clock();
		for( ; i < 10 ; i++)
		{
			GPU->Transfer->SendImage(newImage);
			GPU->Process();
			newImage = GPU->Transfer->ReceiveImage();
		}
		finish = clock();
		duration = (double)(finish - start) / CLOCKS_PER_SEC;
		
		avg += duration;
		
		cout << "Czas wykonania "<< i <<" iteracji: " << endl;
		cout << duration << endl;
		cout << "rozmiar: " << endl;
		cout << newImage->width <<"x"<< newImage->height << endl;
		cout << "-------------------------\n\n" << endl;
		
		cout << (int)newImage->imageData[0] << endl;
		cvNamedWindow("sobel", CV_WINDOW_AUTOSIZE); 
		cvShowImage("sobel", newImage );
		cvWaitKey(0);
		//delete GPU;
		//cvCvtColor(img, image4, CV_HSV2BGR);
	}

	//cout << "avg: " << endl;
	//cout << avg/3 << endl;
	//cout << "-------------------------\n\n" << endl;

   /* cvNamedWindow("sobel", CV_WINDOW_AUTOSIZE); 
    cvShowImage("sobel", img2 );
    cvWaitKey(2);*/



	getchar();
    return 0;

}




