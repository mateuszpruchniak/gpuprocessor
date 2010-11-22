__kernel void ckMin(__global uchar* ucSource,
                      __local uchar* ucLocalData, int iLocalPixPitch, 
                      unsigned int uiImageWidth, unsigned int uiDevImageHeight, int nChannels)
{
	
	LoadToLocalMemNew(ucSource,ucLocalData, iLocalPixPitch, uiImageWidth, uiDevImageHeight,nChannels);
	    
	barrier(CLK_LOCAL_MEM_FENCE);

	int iImagePosX = get_global_id(0);
	int iDevYPrime = get_global_id(1) - 1;  // Shift offset up 1 radius (1 row) for reads
	int iDevGMEMOffset = mul24(iDevYPrime, (int)get_global_size(0)) + iImagePosX;
    int fMinimalEstimate[3] = { 0, 0, 0};
    
    // set local offset and kernel offset
    int iLocalPixOffset = mul24((int)get_local_id(1), iLocalPixPitch) + get_local_id(0);

	// Row1 Left Pix (RGB)
	fMinimalEstimate[0] = GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x;					// red
	fMinimalEstimate[1] = GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y;				    // green
	fMinimalEstimate[2] = GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).z;				    //blue
    ++iLocalPixOffset;

	// Row1 Middle Pix (RGB)
    fMinimalEstimate[0] = fMinimalEstimate[0] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x ? fMinimalEstimate[0] : GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x ;					
	fMinimalEstimate[1] = fMinimalEstimate[1] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y ? fMinimalEstimate[1] : GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y ;						
	fMinimalEstimate[2] = fMinimalEstimate[2] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).z ? fMinimalEstimate[2] : GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).z ;	
    ++iLocalPixOffset;

	// Row1 Right Pix (RGB)
	fMinimalEstimate[0] = fMinimalEstimate[0] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x ? fMinimalEstimate[0] : GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x ;					
	fMinimalEstimate[1] = fMinimalEstimate[1] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y ? fMinimalEstimate[1] : GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y ;						
	fMinimalEstimate[2] = fMinimalEstimate[2] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).z ? fMinimalEstimate[2] : GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).z ;						

	// set the offset into SMEM for next row
	iLocalPixOffset += (iLocalPixPitch - 2);	

	// Row2 Left Pix (RGB)
	fMinimalEstimate[0] = fMinimalEstimate[0] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x ? fMinimalEstimate[0] : GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x ;					
	fMinimalEstimate[1] = fMinimalEstimate[1] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y ? fMinimalEstimate[1] : GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y ;						
	fMinimalEstimate[2] = fMinimalEstimate[2] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).z ? fMinimalEstimate[2] : GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).z ;	
    ++iLocalPixOffset;				

	// Row2 Middle Pix (RGB)
	fMinimalEstimate[0] = fMinimalEstimate[0] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x ? fMinimalEstimate[0] : GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x ;					
	fMinimalEstimate[1] = fMinimalEstimate[1] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y ? fMinimalEstimate[1] : GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y ;						
	fMinimalEstimate[2] = fMinimalEstimate[2] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).z ? fMinimalEstimate[2] : GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).z ;		
    ++iLocalPixOffset;				

	// Row2 Right Pix (RGB)
	fMinimalEstimate[0] = fMinimalEstimate[0] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x ? fMinimalEstimate[0] : GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x ;					
	fMinimalEstimate[1] = fMinimalEstimate[1] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y ? fMinimalEstimate[1] : GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y ;						
	fMinimalEstimate[2] = fMinimalEstimate[2] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).z ? fMinimalEstimate[2] : GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).z ;					

	// set the offset into SMEM for next row
	iLocalPixOffset += (iLocalPixPitch - 2);	

	// Row3 Left Pix (RGB)
	fMinimalEstimate[0] = fMinimalEstimate[0] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x ? fMinimalEstimate[0] : GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x ;					
	fMinimalEstimate[1] = fMinimalEstimate[1] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y ? fMinimalEstimate[1] : GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y ;						
	fMinimalEstimate[2] = fMinimalEstimate[2] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).z ? fMinimalEstimate[2] : GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).z ;		
    ++iLocalPixOffset;					

	// Row3 Middle Pix (RGB)
	fMinimalEstimate[0] = fMinimalEstimate[0] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x ? fMinimalEstimate[0] : GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x ;					
	fMinimalEstimate[1] = fMinimalEstimate[1] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y ? fMinimalEstimate[1] : GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y ;						
	fMinimalEstimate[2] = fMinimalEstimate[2] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).z ? fMinimalEstimate[2] : GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).z ;	
    ++iLocalPixOffset;			

	// Row3 Right Pix (RGB)
	fMinimalEstimate[0] = fMinimalEstimate[0] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x ? fMinimalEstimate[0] : GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).x ;					
	fMinimalEstimate[1] = fMinimalEstimate[1] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y ? fMinimalEstimate[1] : GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).y ;						
	fMinimalEstimate[2] = fMinimalEstimate[2] < GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).z ? fMinimalEstimate[2] : GetDataFromLocalMemory(ucLocalData,iLocalPixOffset,nChannels).z ;					


    uchar4 result;
	result.x = (char)fMinimalEstimate[0];
	result.y = (char)fMinimalEstimate[1];
	result.z = (char)fMinimalEstimate[2];

	// Write out to GMEM with restored offset
	if((iDevYPrime < uiDevImageHeight) && (iImagePosX < uiImageWidth))
	{
		    setData(ucSource,result.x ,result.y, result.z, iDevGMEMOffset,nChannels );
	}
}
