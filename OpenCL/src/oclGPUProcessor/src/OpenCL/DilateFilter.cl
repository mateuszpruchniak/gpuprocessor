




__kernel void ckDilate(__global uchar* ucSource,
                      __local uchar* ucLocalData, int iLocalPixPitch, 
                      unsigned int uiImageWidth, unsigned int uiDevImageHeight, int nChannels)
{
		
	    LoadToLocalMemNew(ucSource,ucLocalData, iLocalPixPitch, uiImageWidth, uiDevImageHeight,nChannels);
	

	    int iImagePosX = get_global_id(0);
	    int iDevYPrime = get_global_id(1) - 1;  // Shift offset up 1 radius (1 row) for reads
	    int iDevGMEMOffset = mul24(iDevYPrime, (int)get_global_size(0)) + iImagePosX;
	    // Synchronize the read into LMEM
	    barrier(CLK_LOCAL_MEM_FENCE);

	    int iLocalPixOffset = mul24((int)get_local_id(1), iLocalPixPitch) + get_local_id(0);
	    
	    unsigned int isOne = 0;

	   

	    // Init summation registers to zero
	    
	    isOne = 0;

	    // set local offset
	    iLocalPixOffset = mul24((int)get_local_id(1), iLocalPixPitch) + get_local_id(0);

		//GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x

	    // NW
	    if(GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y == 255 ) 
	    {
		isOne = 1;
	    } 
	    iLocalPixOffset++;
	
	    // N
	    if(isOne == 0 && GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y == 255 ) 
	    {
		isOne = 1;
	    } 
	    iLocalPixOffset++;

	    // NE
	    if(isOne == 0 && GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y == 255 ) 
	    {
		isOne = 1;
	    }  

	    // increment LMEM block to next row, and unwind increments
	    iLocalPixOffset += (iLocalPixPitch - 2);    
		        
	    // W
	    if(isOne == 0 && GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y == 255 ) 
	    {
		isOne = 1;
	    } 
		iLocalPixOffset++;

	    // C
	    
	    iLocalPixOffset++;

	    // E
	    if(isOne == 0 && GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y == 255 ) 
	    {
		isOne = 1;
	    } 

	    // increment LMEM block to next row, and unwind increments
	    iLocalPixOffset += (iLocalPixPitch - 2);    

	    // SW
	    if(isOne == 0 && GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y == 255 ) 
	    {
		isOne = 1;
	    }  
		iLocalPixOffset++;

	    // S
	    if(isOne == 0 && GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y == 255 ) 
	    {
		isOne = 1;
	    }  
		iLocalPixOffset++;

	    // SE
	    if(isOne == 0 && GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y == 255 ) 
	    {
		isOne = 1;
	    } 


	    uchar4 pix;
		if ( isOne != 1 )
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
			setData(ucSource,pix.x ,pix.y, pix.z, iDevGMEMOffset,nChannels );
	    }
}
