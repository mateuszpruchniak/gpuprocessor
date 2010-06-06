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
#include "RGB2HSV.h"
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

	for(int j = 4; j < 5; j++ )
	{


		IplImage* img = cvLoadImage("./car.jpg");
		IplImage* img2 = cvLoadImage("./car.jpg");
		IplImage* hsv = cvCreateImage( cvGetSize(img2), 8, 3 ) ;
		

		//
		//IplImage* image4 = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 3);

		//
		//
		IplImage *newImage = cvCreateImage( cvSize(img->width * 0.25 * j ,img->height* 0.25 * j), img->depth, img->nChannels );
		IplImage *newImage2 = cvCreateImage( cvSize(img->width * 0.25 * j ,img->height* 0.25 * j), img->depth, img->nChannels );

		cvResize(img, newImage);


		GPUImageProcessor* GPU = new GPUImageProcessor(newImage->width,newImage->height,newImage->nChannels);

		//zamiana na 4 kanaly
		IplImage* image4 = cvCreateImage(cvSize(newImage->width, newImage->height), IPL_DEPTH_8U, 4);

		cvCvtColor(newImage, image4, CV_BGR2BGRA);


		//zamiana na 4 kanaly
		IplImage* image3 = cvCreateImage( cvGetSize(newImage), 8, 3 ) ;

		cvCvtColor(newImage, image3, CV_RGB2HSV);

		
    
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



		//GPU->addFilter( new MeanVariableCentralPointFilter(GPU->GPUContext,transf,maskH) );
		//GPU->addFilter( new MeanFilter(GPU->GPUContext,GPU->Transfer) );
		//GPU->addFilter( new MedianFilter(GPU->GPUContext,GPU->Transfer) );
		//GPU->addFilter( new DilateFilter(GPU->GPUContext,transf) );
		//GPU->addFilter( new ErodeFilter(GPU->GPUContext,GPU->Transfer) );
		
		//GPU->addFilter( new SobelFilter(GPU->GPUContext,GPU->Transfer) );
		GPU->addFilter( new LUTFilter(GPU->GPUContext,GPU->Transfer,lut) );
		//GPU->addFilter( new MinFilter(GPU->GPUContext,GPU->Transfer) );
		////GPU->addFilter( new MaxFilter(GPU->GPUContext,GPU->Transfer) );
		//GPU->addFilter( new OpenFilter(GPU->GPUContext,GPU->Transfer) );
		//GPU->addFilter( new CloseFilter(GPU->GPUContext,transf) );
		//GPU->addFilter( new PrewittFilter(GPU->GPUContext,transf) );
		//GPU->addFilter( new RobertsFilter(GPU->GPUContext,transf) );
		//GPU->addFilter( new LaplaceFilter(GPU->GPUContext,transf) );
		//GPU->addFilter( new CornerDetectionFilter(GPU->GPUContext,transf) );
		//GPU->addFilter( new RGB2HSV(GPU->GPUContext,GPU->Transfer) );

		clock_t start, finish;
		double duration = 0;
		int i = 0;

		start = clock();
		for( ; i < 1 ; i++)
		{
			GPU->Transfer->LoadImageToGPU(newImage);
			GPU->process();
			newImage = GPU->Transfer->GetImageFromGPU();
		}
		finish = clock();
		duration = (double)(finish - start) / CLOCKS_PER_SEC;
		cout << "Czas wykonania "<< i <<" iteracji: " << endl;
		cout << duration << endl;
		cout << "rozmiar: " << endl;
		cout << newImage->width <<"x"<< newImage->height << endl;
		cout << "-------------------------\n\n" << endl;


		cvNamedWindow("sobel", CV_WINDOW_AUTOSIZE); 
		cvShowImage("sobel", newImage );
		cvWaitKey(2);
		
		delete GPU;
		

		//cvCvtColor(img, image4, CV_HSV2BGR);

		

	}

   /* cvNamedWindow("sobel", CV_WINDOW_AUTOSIZE); 
    cvShowImage("sobel", img2 );
    cvWaitKey(2);*/



	getchar();
    return 0;

}



