



__kernel void ckLUT(__global uchar* ucSource, __global int* LUT,
                      __local uchar* ucLocalData, int iLocalPixPitch, 
                      unsigned int uiImageWidth, unsigned int uiDevImageHeight, int nChannels)
{
		
		int iImagePosX = get_global_id(0);
	    int iDevYPrime = get_global_id(1) - 1;  // Shift offset up 1 radius (1 row) for reads
	    int iDevGMEMOffset = mul24(iDevYPrime, (int)get_global_size(0)) + iImagePosX;

		uchar4 input = GetDataFromGlobalMemory(ucSource,iDevGMEMOffset,nChannels);
		input.x = LUT[input.x];
		input.y = LUT[input.y];
		input.z = LUT[input.z];
		

		// Write out to GMEM with restored offset
	    if((iDevYPrime < uiDevImageHeight) && (iImagePosX < uiImageWidth))
	    {
		     setData(ucSource,input.x ,input.y, input.z, iDevGMEMOffset,nChannels );
	    }
}
