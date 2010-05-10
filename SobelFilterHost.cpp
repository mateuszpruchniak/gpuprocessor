/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

// standard utilities and systems includes
#include <oclUtils.h>

//*****************************************************************
//! Exported Host/C++ RGB Sobel gradient magnitude function
//! Gradient intensity is from RSS combination of H and V gradient components
//! R, G and B gradient intensities are treated separately then combined with linear weighting
//!
//! Implementation below is equivalent to linear 2D convolutions for H and V compoonents with:
//!	    Convo Coefs for Horizontal component {1,0,-1,   2,0,-2,  1,0,-1}
//!	    Convo Coefs for Vertical component   {-1,-2,-1,  0,0,0,  1,2,1};
//! @param uiInputImage     pointer to input data
//! @param uiOutputImage    pointer to output dataa
//! @param uiWidth          width of image
//! @param uiHeight         height of image
//! @param fThresh          output intensity threshold 
//*****************************************************************
extern "C" void SobelFilterHost(unsigned int* uiInputImage, unsigned int* uiOutputImage, unsigned int uiWidth, unsigned int uiHeight, float fThresh)
{
	// do the Sobel magnitude with thresholding 
	for(unsigned int y = 0; y < uiHeight; y++)			// local section of rows
	{
		for(unsigned int x = 0; x < uiWidth; x++)		// all the columns
		{
            // local registers for working with RGB subpixels and managing border
            unsigned char* ucRGBA; 
            const unsigned int uiZero = 0U;

            // Init summation registers to zero
            float fTemp = 0.0f; 
            float fHSum[3] = {0.0f, 0.0f, 0.0f}; 
            float fVSum[3] = {0.0f, 0.0f, 0.0f}; 

            // Read in pixel value to local register:  if boundary pixel, use zero
            if ((x > 0) && (y > 0))
            {
                ucRGBA = (unsigned char*)&uiInputImage [((y - 1) * uiWidth) + (x - 1)];
            }
            else 
            {
                ucRGBA = (unsigned char*)&uiZero;
            }

            // NW
		    fHSum[0] += ucRGBA[0];    // horizontal gradient of Red
		    fHSum[1] += ucRGBA[1];    // horizontal gradient of Green
		    fHSum[2] += ucRGBA[2];    // horizontal gradient of Blue
            fVSum[0] -= ucRGBA[0];    // vertical gradient of Red
		    fVSum[1] -= ucRGBA[1];    // vertical gradient of Green
		    fVSum[2] -= ucRGBA[2];    // vertical gradient of Blue

            // Read in next pixel value to a local register:  if boundary pixel, use zero
            if (y > 0) 
            {
                ucRGBA = (unsigned char*)&uiInputImage [((y - 1) * uiWidth) + x];
            }
            else 
            {
                ucRGBA = (unsigned char*)&uiZero;
            }

            // N
		    fVSum[0] -= (ucRGBA[0] << 1);  // vertical gradient of Red
		    fVSum[1] -= (ucRGBA[1] << 1);  // vertical gradient of Green
		    fVSum[2] -= (ucRGBA[2] << 1);  // vertical gradient of Blue

            // Read in next pixel value to a local register:  if boundary pixel, use zero
            if ((x < (uiWidth - 1)) && (y > 0))
            {
                ucRGBA = (unsigned char*)&uiInputImage [((y - 1) * uiWidth) + (x + 1)];
            }
            else 
            {
                ucRGBA = (unsigned char*)&uiZero;
            }

            // NE
		    fHSum[0] -= ucRGBA[0];    // horizontal gradient of Red
		    fHSum[1] -= ucRGBA[1];    // horizontal gradient of Green
		    fHSum[2] -= ucRGBA[2];    // horizontal gradient of Blue
		    fVSum[0] -= ucRGBA[0];    // vertical gradient of Red
		    fVSum[1] -= ucRGBA[1];    // vertical gradient of Green
		    fVSum[2] -= ucRGBA[2];    // vertical gradient of Blue

            // Read in pixel value to a local register:  if boundary pixel, use zero
            if (x > 0) 
            {
                ucRGBA = (unsigned char*)&uiInputImage [(y * uiWidth) + (x - 1)];
            }
            else 
            {
                ucRGBA = (unsigned char*)&uiZero;
            }

            // W
		    fHSum[0] += (ucRGBA[0] << 1);  // vertical gradient of Red
		    fHSum[1] += (ucRGBA[1] << 1);  // vertical gradient of Green
		    fHSum[2] += (ucRGBA[2] << 1);  // vertical gradient of Blue

            // C
            // nothing to do for center pixel

            // Read in pixel value to a local register:  if boundary pixel, use zero
            if (x < (uiWidth - 1))
            {
                ucRGBA = (unsigned char*)&uiInputImage [(y * uiWidth) + (x + 1)];
            }
            else 
            {
                ucRGBA = (unsigned char*)&uiZero;
            }

            // E
		    fHSum[0] -= (ucRGBA[0] << 1);  // vertical gradient of Red
		    fHSum[1] -= (ucRGBA[1] << 1);  // vertical gradient of Green
		    fHSum[2] -= (ucRGBA[2] << 1);  // vertical gradient of Blue

            // Read in pixel value to a local register:  if boundary pixel, use zero
            if ((x > 0) && (y < (uiHeight - 1)))
            {
                ucRGBA = (unsigned char*)&uiInputImage [((y + 1) * uiWidth) + (x - 1)];
            }
            else 
            {
                ucRGBA = (unsigned char*)&uiZero;
            }

            // SW
		    fHSum[0] += ucRGBA[0];    // horizontal gradient of Red
		    fHSum[1] += ucRGBA[1];    // horizontal gradient of Green
		    fHSum[2] += ucRGBA[2];    // horizontal gradient of Blue
		    fVSum[0] += ucRGBA[0];    // vertical gradient of Red
		    fVSum[1] += ucRGBA[1];    // vertical gradient of Green
		    fVSum[2] += ucRGBA[2];    // vertical gradient of Blue

            // Read in pixel value to a local register:  if boundary pixel, use zero
            if (y < (uiHeight - 1))
            {
                ucRGBA = (unsigned char*)&uiInputImage [((y + 1) * uiWidth) + x];
            }
            else 
            {
                ucRGBA = (unsigned char*)&uiZero;
            }

            // S
		    fVSum[0] += (ucRGBA[0] << 1);  // vertical gradient of Red
		    fVSum[1] += (ucRGBA[1] << 1);  // vertical gradient of Green
		    fVSum[2] += (ucRGBA[2] << 1);  // vertical gradient of Blue

            // Read in pixel value to a local register:  if boundary pixel, use zero
            if ((x < (uiWidth - 1)) && (y < (uiHeight - 1)))
            {
                ucRGBA = (unsigned char*)&uiInputImage [((y + 1) * uiWidth) + (x + 1)];
            }
            else 
            {
                ucRGBA = (unsigned char*)&uiZero;
            }

            // SE
		    fHSum[0] -= ucRGBA[0];    // horizontal gradient of Red
		    fHSum[1] -= ucRGBA[1];    // horizontal gradient of Green
		    fHSum[2] -= ucRGBA[2];    // horizontal gradient of Blue
		    fVSum[0] += ucRGBA[0];    // vertical gradient of Red
		    fVSum[1] += ucRGBA[1];    // vertical gradient of Green
		    fVSum[2] += ucRGBA[2];    // vertical gradient of Blue
            
		    // Weighted combination of Root-Sum-Square per-color-band H & V gradients for each of RGB
            fTemp =  0.30f * sqrtf((fHSum[0] * fHSum[0]) + (fVSum[0] * fVSum[0]));
			fTemp += 0.55f * sqrtf((fHSum[1] * fHSum[1]) + (fVSum[1] * fVSum[1]));
			fTemp += 0.15f * sqrtf((fHSum[2] * fHSum[2]) + (fVSum[2] * fVSum[2]));

            // threshold and clamp
            if (fTemp < fThresh)
            {
                fTemp = 0.0f;
            }
            else if (fTemp > 255.0f)
            {
                fTemp = 255.0f;
            }

            // pack into a monochrome uint (including average alpha)
            unsigned int uiPackedPix = 0x000000FF & (unsigned int)fTemp;
            uiPackedPix |= 0x0000FF00 & (((unsigned int)fTemp) << 8);
            uiPackedPix |= 0x00FF0000 & (((unsigned int)fTemp) << 16);

			// convert and copy to output
			uiOutputImage[y * uiWidth + x] = uiPackedPix;	
		}
	}
}
