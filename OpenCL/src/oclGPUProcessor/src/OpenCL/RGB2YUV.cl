
__kernel void ckRGB2HSV(__global uchar4* ucSource,
                      unsigned int uiImageWidth, unsigned int uiDevImageHeight)
{
		int nChannels = 3;
		int iImagePosX = get_global_id(0);
	    int iDevYPrime = get_global_id(1) - 1;  // Shift offset up 1 radius (1 row) for reads
	    int iDevGMEMOffset = mul24(iDevYPrime, (int)get_global_size(0)) + iImagePosX;

		uchar4 pix = ucSource[iDevGMEMOffset];
		float R = (float)pix.x;
		float G = (float)pix.y;
		float B = (float)pix.z;

		float Y = (0.257 * R) + (0.504 * G) + (0.098 * B) + 16;

		float V = (0.439 * R) - (0.368 * G) - (0.071 * B) + 128;

		float U = -(0.148 * R) - (0.291 * G) + (0.439 * B) + 128;

		pix.x = (char)Y;
		pix.y = (char)U;
		pix.z = (char)V;

		// Write out to GMEM with restored offset
	    if((iDevYPrime < uiDevImageHeight) && (iImagePosX < uiImageWidth))
	    {
		     setData(ucSource,pix.x ,pix.y, pix.z, iDevGMEMOffset,nChannels );
	    }
		
}