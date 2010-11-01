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
//! Exported Host/C++ RGB 3x3 Median function
//! Gradient intensity is from RSS combination of H and V gradient components
//! R, G and B medians are treated separately 
//!
//! @param uiInputImage     pointer to input data
//! @param uiOutputImage    pointer to output dataa
//! @param uiWidth          width of image
//! @param uiHeight         height of image
//*****************************************************************
extern "C" void MedianFilterHost(unsigned int* uiInputImage, unsigned int* uiOutputImage, 
                                 unsigned int uiWidth, unsigned int uiHeight)
{
	// do the Median 
	for(unsigned int y = 0; y < uiHeight; y++)			// local section of rows
	{
		for(unsigned int x = 0; x < uiWidth; x++)		// all the columns
		{
            // local registers for working with RGB subpixels and managing border
            unsigned char* ucRGBA; 
            const unsigned int uiZero = 0U;

		    // reset accumulators  
            float fMedianEstimate [3] = {128.0f, 128.0f, 128.0f};
            float fMinBound [3]= {0.0f, 0.0f, 0.0f};
            float fMaxBound[3] = {255.0f, 255.0f, 255.0f};

		    // now find the median using a binary search - Divide and Conquer 256 gv levels for 8 bit plane
            // for 8 bit data, use 0..8.  For 16 bit data, 0..16. More iterations for more bits.
		    for(int iSearch = 0; iSearch < 8; iSearch++)  
		    {
                unsigned int uiHighCount[3] = {0,0,0};

			    for (int iRow = 0; iRow < 3; iRow++)
			    {
                    int iLocalOffset = ((y + iRow - 1) * uiWidth) + x;

                    // Read in pixel value to local register:  if boundary pixel, use zero
                    if ((x > 0) && (y > 0) && (y < (uiHeight - 1)))
                    {
                        ucRGBA = (unsigned char*)&uiInputImage [iLocalOffset - 1];
                    }
                    else 
                    {
                        ucRGBA = (unsigned char*)&uiZero;
                    }

				    // Left Pix (RGB)
				    uiHighCount[0] += (fMedianEstimate[0] < ucRGBA[0]);					
				    uiHighCount[1] += (fMedianEstimate[1] < ucRGBA[1]);					
				    uiHighCount[2] += (fMedianEstimate[2] < ucRGBA[2]);	

                    // Read in next pixel value to a local register:  if boundary pixel, use zero
                    if ((y > 0) && (y < (uiHeight - 1))) 
                    {
                        ucRGBA = (unsigned char*)&uiInputImage [iLocalOffset];
                    }
                    else 
                    {
                        ucRGBA = (unsigned char*)&uiZero;
                    }

				    // Middle Pix (RGB)
				    uiHighCount[0] += (fMedianEstimate[0] < ucRGBA[0]);					
				    uiHighCount[1] += (fMedianEstimate[1] < ucRGBA[1]);					
				    uiHighCount[2] += (fMedianEstimate[2] < ucRGBA[2]);	

                    // Read in next pixel value to a local register:  if boundary pixel, use zero
                    if ((x < (uiWidth - 1)) && (y > 0) && (y < (uiHeight - 1)))
                    {
                        ucRGBA = (unsigned char*)&uiInputImage [iLocalOffset + 1];
                    }
                    else 
                    {
                        ucRGBA = (unsigned char*)&uiZero;
                    }

				    // Right Pix (RGB)
				    uiHighCount[0] += (fMedianEstimate[0] < ucRGBA[0]);					
				    uiHighCount[1] += (fMedianEstimate[1] < ucRGBA[1]);					
				    uiHighCount[2] += (fMedianEstimate[2] < ucRGBA[2]);	
			    }

			    //********************************
			    // reset the appropriate bound, depending upon counter
			    if(uiHighCount[0] > 4)
			    {
				    fMinBound[0] = fMedianEstimate[0];				
			    }
			    else
			    {
				    fMaxBound[0] = fMedianEstimate[0];				
			    }

			    if(uiHighCount[1] > 4)
			    {
				    fMinBound[1] = fMedianEstimate[1];				
			    }
			    else
			    {
				    fMaxBound[1] = fMedianEstimate[1];				
			    }

			    if(uiHighCount[2] > 4)
			    {
				    fMinBound[2] = fMedianEstimate[2];				
			    }
			    else
			    {
				    fMaxBound[2] = fMedianEstimate[2];				
			    }

			    // refine the estimate
			    fMedianEstimate[0] = (fMaxBound[0] + fMinBound[0]) * 0.5f;
			    fMedianEstimate[1] = (fMaxBound[1] + fMinBound[1]) * 0.5f;
			    fMedianEstimate[2] = (fMaxBound[2] + fMinBound[2]) * 0.5f;
		    }

            // pack into a monochrome uint 
            unsigned int uiPackedPix = 0x000000FF & (unsigned int)fMedianEstimate[0];
            uiPackedPix |= 0x0000FF00 & (((unsigned int)fMedianEstimate[1]) << 8);
            uiPackedPix |= 0x00FF0000 & (((unsigned int)fMedianEstimate[2]) << 16);

			// convert and copy to output
			uiOutputImage[y * uiWidth + x] = uiPackedPix;	
		}
	}
}
