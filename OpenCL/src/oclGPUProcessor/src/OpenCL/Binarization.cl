



__kernel void ckBin(__global uchar* ucSource,unsigned int Threshold,
                      unsigned int uiImageWidth, unsigned int uiDevImageHeight,unsigned int nChannels)
{

		int iImagePosX = get_global_id(0);
	    int iDevYPrime = get_global_id(1) - 1;  // Shift offset up 1 radius (1 row) for reads
	    int iDevGMEMOffset = mul24(iDevYPrime, (int)get_global_size(0)) + iImagePosX;

		uchar4 input = GetDataFromGlobalMemory(ucSource,iDevGMEMOffset,nChannels);

		float fTemp =  0.30f * (int)input.x + 0.30f * (int)input.y + 0.30f * (int)input.z;

		// threshold and clamp
		if (fTemp < Threshold)
		{
			input.x = 0;
		}
		else
		{
			input.x = 255;
		}

		// Write out to GMEM with restored offset
	    if((iDevYPrime < uiDevImageHeight) && (iImagePosX < uiImageWidth))
	    {
		     setData(ucSource,input.x ,input.x, input.x, iDevGMEMOffset,nChannels);
	    }
}
