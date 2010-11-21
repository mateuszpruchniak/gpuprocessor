
__kernel void ckGradient(__global uchar* ucSource, __global int* maskGlobalH, __global int* maskGlobalV,
                      __local uchar* ucLocalData, __local int* maskLocalH, __local int* maskLocalV, int iLocalPixPitch, 
                      unsigned int uiImageWidth, unsigned int uiDevImageHeight, int channels)
{
		nChannels = channels;

	    LoadToLocalMemNew(ucSource,ucLocalData, iLocalPixPitch, uiImageWidth, uiDevImageHeight);
	    

	    barrier(CLK_LOCAL_MEM_FENCE);

	    unsigned int isZero = 0;
	    int iImagePosX = get_global_id(0);
	    int iDevYPrime = get_global_id(1) - 1;  // Shift offset up 1 radius (1 row) for reads
	    int iDevGMEMOffset = mul24(iDevYPrime, (int)get_global_size(0)) + iImagePosX;


		int tmp = get_local_id(0);
		int size = get_local_size(0);

		if( size < 9 ) return;

		if( get_local_id(1) == 0)
		{
			maskLocalH[tmp] = maskGlobalH[tmp];
			maskLocalV[tmp] = maskGlobalV[tmp];
		}

		barrier(CLK_LOCAL_MEM_FENCE);



	    // Init summation registers to zero
	    float fTemp = 0.0f; 
	    float fHSum [3] = {0.0f, 0.0f, 0.0f};
	    float fVSum [3] = {0.0f, 0.0f, 0.0f};

	    
	    // set local offset
	    
	    int iLocalPixOffset = mul24((int)get_local_id(1), iLocalPixPitch) + get_local_id(0);

	    // NW
		fVSum[0] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).x*maskLocalV[0];    
		fVSum[1] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).y*maskLocalV[0];    
		fVSum[2] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).z*maskLocalV[0];  
		fHSum[0] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).x*maskLocalH[0];    
		fHSum[1] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).y*maskLocalH[0];    
		fHSum[2] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset++).z*maskLocalH[0];

	    // N
		fVSum[0] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).x*maskLocalV[1];    
		fVSum[1] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).y*maskLocalV[1];    
		fVSum[2] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).z*maskLocalV[1];  
		fHSum[0] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).x*maskLocalH[1];    
		fHSum[1] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).y*maskLocalH[1];    
		fHSum[2] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset++).z*maskLocalH[1];

	    // NE
		fVSum[0] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).x*maskLocalV[2];    
		fVSum[1] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).y*maskLocalV[2];    
		fVSum[2] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).z*maskLocalV[2];  
		fHSum[0] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).x*maskLocalH[2];    
		fHSum[1] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).y*maskLocalH[2];    
		fHSum[2] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).z*maskLocalH[2];

	    // increment LMEM block to next row, and unwind increments
	    iLocalPixOffset += (iLocalPixPitch - 2);    
		        
	    // W
		fVSum[0] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).x*maskLocalV[3];    
		fVSum[1] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).y*maskLocalV[3];    
		fVSum[2] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).z*maskLocalV[3];  
		fHSum[0] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).x*maskLocalH[3];    
		fHSum[1] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).y*maskLocalH[3];    
		fHSum[2] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset++).z*maskLocalH[3];

	    // C
	    fVSum[0] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).x*maskLocalV[4];    
		fVSum[1] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).y*maskLocalV[4];    
		fVSum[2] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).z*maskLocalV[4];  
		fHSum[0] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).x*maskLocalH[4];    
		fHSum[1] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).y*maskLocalH[4];    
		fHSum[2] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset++).z*maskLocalH[4];

	    // E
		fVSum[0] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).x*maskLocalV[5];    
		fVSum[1] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).y*maskLocalV[5];    
		fVSum[2] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).z*maskLocalV[5];  
		fHSum[0] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).x*maskLocalH[5];    
		fHSum[1] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).y*maskLocalH[5];    
		fHSum[2] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).z*maskLocalH[5];

	    // increment LMEM block to next row, and unwind increments
	    iLocalPixOffset += (iLocalPixPitch - 2);    

	    // SW
		fVSum[0] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).x*maskLocalV[6];    
		fVSum[1] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).y*maskLocalV[6];    
		fVSum[2] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).z*maskLocalV[6];  
		fHSum[0] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).x*maskLocalH[6];    
		fHSum[1] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).y*maskLocalH[6];    
		fHSum[2] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset++).z*maskLocalH[6];

	    // S
		fVSum[0] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).x*maskLocalV[7];    
		fVSum[1] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).y*maskLocalV[7];    
		fVSum[2] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).z*maskLocalV[7];  
		fHSum[0] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).x*maskLocalH[7];    
		fHSum[1] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).y*maskLocalH[7];    
		fHSum[2] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset++).z*maskLocalH[7];

	    // SE
		fVSum[0] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).x*maskLocalV[8];    
		fVSum[1] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).y*maskLocalV[8];    
		fVSum[2] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).z*maskLocalV[8];  
		fHSum[0] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).x*maskLocalH[8];    
		fHSum[1] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).y*maskLocalH[8];    
		fHSum[2] +=  (float)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).z*maskLocalH[8];

		// Weighted combination of Root-Sum-Square per-color-band H & V gradients for each of RGB
		fTemp =  0.30f * sqrt((fHSum[0] * fHSum[0]) + (fVSum[0] * fVSum[0]));
		fTemp += 0.30f * sqrt((fHSum[1] * fHSum[1]) + (fVSum[1] * fVSum[1]));
		fTemp += 0.30f * sqrt((fHSum[2] * fHSum[2]) + (fVSum[2] * fVSum[2]));

	    
	    
	    uchar4 pix;
		if (fTemp < 255.0f)
	    {
			pix.x = (char)fTemp;
			pix.y = (char)fTemp;
			pix.z = (char)fTemp;
		}
		else
		{
			pix.x = 255;
			pix.y = 255;
			pix.z = 255;
		}
	    
		// Write out to GMEM with restored offset
	    if((iDevYPrime < uiDevImageHeight) && (iImagePosX < uiImageWidth))
	    {
		    setData(ucSource,pix.x ,pix.y, pix.z, iDevGMEMOffset );
	    }
}
