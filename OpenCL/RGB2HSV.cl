
__kernel void ckRGB2HSV(__global uchar4* ucSource,
                      unsigned int uiImageWidth, unsigned int uiDevImageHeight)
{

		int iImagePosX = get_global_id(0);
	    int iDevYPrime = get_global_id(1) - 1;  // Shift offset up 1 radius (1 row) for reads
	    int iDevGMEMOffset = mul24(iDevYPrime, (int)get_global_size(0)) + iImagePosX;

		uchar4 pix = ucSource[iDevGMEMOffset];
		float r = (float)pix.z/255;
		float g = (float)pix.y/255;
		float b = (float)pix.x/255;

		float h;
		float s;
		float v;

		float max = b;
		if (max < g) max = g;
		if (max < r) max = r;
		float min = b;
		if (min > g) min = g;
		if (min > r) min = r;

		float delta;
		v = max; // v
		delta = max - min;

		if( max != 0 )
			s = delta / max; // s
		else {
			// r = g = b = 0 // s = 0, v is undefined
			s = 0;
			h = -1;
			return;
		}
		if( r == max )
			h = ( g - b ) / delta; // between yellow &magenta
		else if( g == max )
			h = 2 + ( b - r ) / delta; // between cyan & yellow
		else
			h = 4 + ( r - g ) / delta; // between magenta & cyan
		h = 60; // degrees
		if( h < 0 )
			h += 360;

		pix.z = (char)1;//h;
		pix.y = (char)200;//s;
		pix.x = (char)1;//v;

		 Write out to GMEM with restored offset
	    if((iDevYPrime < uiDevImageHeight) && (iImagePosX < uiImageWidth))
	    {
		     setData(ucSource,pix.x ,pix.y, pix.z, iDevGMEMOffset );
	    }

		// Write out to GMEM with restored offset
	    if((iDevYPrime < uiDevImageHeight) && (iImagePosX < uiImageWidth))
	    {
		    //setData(ucSource,(char)res[0] ,(char)res[1], (char)res[2], iDevGMEMOffset );
			setData(ucSource,(char)1 ,(char)100, (char)1, iDevGMEMOffset );
	    }
}