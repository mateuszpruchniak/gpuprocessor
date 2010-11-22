



__kernel void ckMedian(__global uchar* ucSource,
                      __local uchar* ucLocalData, int iLocalPixPitch, 
                      unsigned int uiImageWidth, unsigned int uiDevImageHeight, int nChannels)
{
		
	    LoadToLocalMemNew(ucSource,ucLocalData, iLocalPixPitch, uiImageWidth, uiDevImageHeight,nChannels);
	    
	    barrier(CLK_LOCAL_MEM_FENCE);

	    
	    int iImagePosX = get_global_id(0);
	    int iDevYPrime = get_global_id(1) - 1;  // Shift offset up 1 radius (1 row) for reads
	    int iDevGMEMOffset = mul24(iDevYPrime, (int)get_global_size(0)) + iImagePosX;
	    
	    float fMedianEstimate[3] = {128.0f, 128.0f, 128.0f};
	    float fMinBound[3] = {0.0f, 0.0f, 0.0f};
	    float fMaxBound[3] = {255.0f, 255.0f, 255.0f};

		// now find the median using a binary search - Divide and Conquer 256 gv levels for 8 bit plane
		for(int iSearch = 0; iSearch < 8; iSearch++)  // for 8 bit data, use 0..8.  For 16 bit data, 0..16. More iterations for more bits.
		{
		uint uiHighCount [3] = {0, 0, 0};
		
		// set local offset and kernel offset
		int iLocalPixOffset = mul24((int)get_local_id(1), iLocalPixPitch) + get_local_id(0);

			// Row1 Left Pix (RGB)
			uiHighCount[0] += (fMedianEstimate[0] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x);					
			uiHighCount[1] += (fMedianEstimate[1] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y);					
			uiHighCount[2] += (fMedianEstimate[2] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset++,nChannels).z);					

			// Row1 Middle Pix (RGB)
			uiHighCount[0] += (fMedianEstimate[0] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x);					
			uiHighCount[1] += (fMedianEstimate[1] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y);					
			uiHighCount[2] += (fMedianEstimate[2] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset++,nChannels).z);					

			// Row1 Right Pix (RGB)
			uiHighCount[0] += (fMedianEstimate[0] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x);					
			uiHighCount[1] += (fMedianEstimate[1] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y);					
			uiHighCount[2] += (fMedianEstimate[2] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).z);					

			// set the offset into SMEM for next row
			iLocalPixOffset += (iLocalPixPitch - 2);	

			// Row2 Left Pix (RGB)
			uiHighCount[0] += (fMedianEstimate[0] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x);					
			uiHighCount[1] += (fMedianEstimate[1] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y);					
			uiHighCount[2] += (fMedianEstimate[2] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset++,nChannels).z);					

			// Row2 Middle Pix (RGB)
			uiHighCount[0] += (fMedianEstimate[0] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x);					
			uiHighCount[1] += (fMedianEstimate[1] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y);					
			uiHighCount[2] += (fMedianEstimate[2] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset++,nChannels).z);					

			// Row2 Right Pix (RGB)
			uiHighCount[0] += (fMedianEstimate[0] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x);					
			uiHighCount[1] += (fMedianEstimate[1] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y);					
			uiHighCount[2] += (fMedianEstimate[2] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).z);					

			// set the offset into SMEM for next row
			iLocalPixOffset += (iLocalPixPitch - 2);	

			// Row3 Left Pix (RGB)
			uiHighCount[0] += (fMedianEstimate[0] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x);					
			uiHighCount[1] += (fMedianEstimate[1] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y);					
			uiHighCount[2] += (fMedianEstimate[2] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset++,nChannels).z);					

			// Row3 Middle Pix (RGB)
			uiHighCount[0] += (fMedianEstimate[0] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x);					
			uiHighCount[1] += (fMedianEstimate[1] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y);					
			uiHighCount[2] += (fMedianEstimate[2] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset++,nChannels).z);					

			// Row3 Right Pix (RGB)
			uiHighCount[0] += (fMedianEstimate[0] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x);					
			uiHighCount[1] += (fMedianEstimate[1] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y);					
			uiHighCount[2] += (fMedianEstimate[2] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).z);					

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

	    //pack into a monochrome uint 
	    //unsigned int uiPackedPix = 0x000000FF & (unsigned int)fMedianEstimate[0];
	    //uiPackedPix |= 0x0000FF00 & (((unsigned int)fMedianEstimate[1]) << 8);
	    //uiPackedPix |= 0x00FF0000 & (((unsigned int)fMedianEstimate[2]) << 16);

		uchar4 result;
		result.x = fMedianEstimate[0];
		result.y = fMedianEstimate[1];
		result.z = fMedianEstimate[2];

	    // Write out to GMEM with restored offset
	    if((iDevYPrime < uiDevImageHeight) && (iImagePosX < uiImageWidth))
	    {
		     setData(ucSource,result.x ,result.y, result.z, iDevGMEMOffset,nChannels );
	    }
}
