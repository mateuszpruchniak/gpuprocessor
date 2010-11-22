



__kernel void ckErode(__global uchar* ucSource,
                      __local uchar* ucLocalData, int iLocalPixPitch, 
                      unsigned int uiImageWidth, unsigned int uiDevImageHeight, int nChannels)
{
		
	    LoadToLocalMemNew(ucSource,ucLocalData, iLocalPixPitch, uiImageWidth, uiDevImageHeight,nChannels);

	    barrier(CLK_LOCAL_MEM_FENCE);

	    unsigned int isZero = 0;
        int iImagePosX = get_global_id(0);
	    int iDevYPrime = get_global_id(1) - 1;  // Shift offset up 1 radius (1 row) for reads
	    int iDevGMEMOffset = mul24(iDevYPrime, (int)get_global_size(0)) + iImagePosX;
	    // Init summation registers to zero
	    
	    isZero = 0;

	    // set local offset
	    int iLocalPixOffset = mul24((int)get_local_id(1), iLocalPixPitch) + get_local_id(0);

	    // NW
	    if(GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x == 0 ) 
	    {
		isZero = 1;
	    } 
	    iLocalPixOffset++;
	
	    // N
	    if(isZero == 0 && GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x == 0 ) 
	    {
		isZero = 1;
	    } 
	    iLocalPixOffset++;

	    // NE
	    if(isZero == 0 && GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x == 0 )
	    {
		isZero = 1;
	    } 

	    // increment LMEM block to next row, and unwind increments
	    iLocalPixOffset += (iLocalPixPitch - 2);    
		        
	    // W
	    if(isZero == 0 && GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x == 0 )
	    {
		isZero = 1;
	    } 
		iLocalPixOffset++;

	    // C
	    
	    iLocalPixOffset++;

	    // E
	    if(isZero == 0 && GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x == 0 )
	    {
		isZero = 1;
	    } 

	    // increment LMEM block to next row, and unwind increments
	    iLocalPixOffset += (iLocalPixPitch - 2);    

	    // SW
	    if(isZero == 0 && GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x == 0 )
	    {
		isZero = 1;
	    } 
		iLocalPixOffset++;

	    // S
	    if(isZero == 0 && GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x == 0 )
	    {
		isZero = 1;
	    } 
		iLocalPixOffset++;

	    // SE
	    if(isZero == 0 && GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x == 0 )
	    {
		isZero = 1;
	    } 

		uchar4 pix;
		if ( isZero == 1 )
	    {
			pix.x = 0;
			pix.y = 0;
			pix.z = 0;
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
		    setData(ucSource,pix.x ,pix.y, pix.z, iDevGMEMOffset, nChannels);
	    }
}

