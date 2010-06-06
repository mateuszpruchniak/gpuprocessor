



__kernel void ckConv2(__global uchar* ucSource, __global unsigned int* maskGlobal,
                      __local uchar* ucLocalData, __local unsigned int* maskLocal, int radius, 
                      unsigned int uiImageWidth, unsigned int uiDevImageHeight, int channels)
{
		nChannels = channels;
	    

		int iImagePosX = get_global_id(0);
	    int iDevYPrime = get_global_id(1) - 1;  // Shift offset up 1 radius (1 row) for reads
	    int iDevGMEMOffset = mul24(iDevYPrime, (int)get_global_size(0)) + iImagePosX;

		int fVSum [3] = {0, 0, 0};
		int res[3] = {0, 0, 0};

		radius = 1;

		if( iImagePosX - radius >= 0 && iDevYPrime - radius >= 0 && iImagePosX <= uiImageWidth - radius && iDevYPrime <= uiDevImageHeight - radius )
		{
			int maskDim = 3;
			int k,l;
			int sum=maskDim*maskDim;
			int a,b;

			
			for(k=0;k<maskDim;k++)
			{
				for(l=0;l<maskDim;l++)
				{
				    a=iDevYPrime+k-radius;
					b=iImagePosX+l-radius;
					
					int tmp = a * (int)get_global_size(0) + b;
					fVSum[0] += (int)(GetDataFromGlobalMemory(ucSource,tmp).x);    
					fVSum[1] += (int)(GetDataFromGlobalMemory(ucSource,tmp).y);    
					fVSum[2] += (int)(GetDataFromGlobalMemory(ucSource,tmp).z);
				}
			}
			
			res[0] = fVSum[0] / sum;
			if( res[0] > 255 ) res[0] = 255;
			res[1] = fVSum[1] / sum;
			if( res[1] > 255 ) res[1] = 255;
			res[2] = fVSum[2] / sum;
			if( res[2] > 255 ) res[2] = 255;

	
			barrier(CLK_LOCAL_MEM_FENCE);

			
			//setData(ucSource,(char)0 ,(char)0, (char)fVSum[0], iDevGMEMOffset );
			setData(ucSource,(char)res[0] ,(char)res[1], (char)res[2], iDevGMEMOffset );
			

		}
		
		barrier(CLK_LOCAL_MEM_FENCE);

		if( iImagePosX - radius < 0 )
		{
			uchar4 input = GetDataFromGlobalMemory(ucSource,mul24(iDevYPrime, (int)get_global_size(0)) + radius);

			
			//setData(ucSource,(char)0 ,(char)0, (char)fVSum[0], iDevGMEMOffset );
			setData(ucSource,input.x,input.y, input.z, iDevGMEMOffset );
			
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if( iImagePosX > uiImageWidth - radius )
		{
			uchar4 input = GetDataFromGlobalMemory(ucSource,mul24(iDevYPrime, (int)get_global_size(0)) + uiImageWidth - radius);

			
			//setData(ucSource,(char)0 ,(char)0, (char)fVSum[0], iDevGMEMOffset );
			setData(ucSource,input.x,input.y, input.z, iDevGMEMOffset );
			
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if( iDevYPrime - radius < 0 )
		{
			uchar4 input = GetDataFromGlobalMemory(ucSource,mul24(radius, (int)get_global_size(0)) + iImagePosX);

			setData(ucSource,input.x,input.y, input.z, iDevGMEMOffset );
			
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if( iDevYPrime+2 > uiDevImageHeight-radius )
		{
			uchar4 input = GetDataFromGlobalMemory(ucSource,mul24((int)(uiDevImageHeight-radius), (int)get_global_size(0)) + iImagePosX);
			
			iDevGMEMOffset = mul24(iDevYPrime+2, (int)get_global_size(0)) + iImagePosX;
			//setData(ucSource,(char)0 ,(char)0, (char)fVSum[0], iDevGMEMOffset );
			setData(ucSource,input.x,input.y, input.z, iDevGMEMOffset );
			
		}

}


__kernel void ckConv(__global uchar* ucSource, __global unsigned int* maskGlobal,
                      __local uchar* ucLocalData, __local unsigned int* maskLocal, int iLocalPixPitch, 
                      unsigned int uiImageWidth, unsigned int uiDevImageHeight, int channels)
{
		nChannels = channels;
	    LoadToLocalMemNew(ucSource,ucLocalData, iLocalPixPitch, uiImageWidth, uiDevImageHeight);
	    
	    barrier(CLK_LOCAL_MEM_FENCE);

		int iImagePosX = get_global_id(0);
	    int iDevYPrime = get_global_id(1) - 1;  // Shift offset up 1 radius (1 row) for reads
	    int iDevGMEMOffset = mul24(iDevYPrime, (int)get_global_size(0)) + iImagePosX;

		int tmp = get_local_id(0);
		int size = get_local_size(0);

		if( size < 9 ) return;

		if( get_local_id(1) == 0)
		{
			maskLocal[tmp] = maskGlobal[tmp];
		}

		barrier(CLK_LOCAL_MEM_FENCE);


		// Init summation registers to zero
	    float fTemp = 0.0f; 
	    int fVSum [3] = {0, 0, 0};
		int res[3] = {0, 0, 0};
	    
	    // set local offset
	    
	    int iLocalPixOffset = mul24((int)get_local_id(1), iLocalPixPitch) + get_local_id(0);
		
	    // NW
	    fVSum[0] = fVSum[0] + (int)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).x*maskLocal[0];    
		fVSum[1] = fVSum[1] + (int)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).y*maskLocal[0];    
		fVSum[2] = fVSum[2] + (int)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset++).z*maskLocal[0];  

	    // N
	    fVSum[0] = fVSum[0] + (int)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).x*maskLocal[1];    
		fVSum[1] = fVSum[1] + (int)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).y*maskLocal[1];    
		fVSum[2] = fVSum[2] + (int)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset++).z*maskLocal[1];  

	    // NE
	    fVSum[0] = fVSum[0] + (int)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).x*maskLocal[2];    
		fVSum[1] = fVSum[1] + (int)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).y*maskLocal[2];    
		fVSum[2] = fVSum[2] + (int)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).z*maskLocal[2];  

	    // increment LMEM block to next row, and unwind increments
	    iLocalPixOffset += (iLocalPixPitch - 2);    
		        
	    // W
	    fVSum[0] = fVSum[0] + (int)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).x*maskLocal[3];    
		fVSum[1] = fVSum[1] + (int)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).y*maskLocal[3];    
		fVSum[2] = fVSum[2] + (int)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset++).z*maskLocal[3]; 

	    // C
	    fVSum[0] = fVSum[0] + (int)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).x*maskLocal[4];    
		fVSum[1] = fVSum[1] + (int)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).y*maskLocal[4];    
		fVSum[2] = fVSum[2] + (int)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset++).z*maskLocal[4];  

	    // E
	    fVSum[0] = fVSum[0] + (int)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).x*maskLocal[5];    
		fVSum[1] = fVSum[1] + (int)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).y*maskLocal[5];  
		fVSum[2] = fVSum[2] + (int)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).z*maskLocal[5]; 

	    // increment LMEM block to next row, and unwind increments
	    iLocalPixOffset += (iLocalPixPitch - 2);    

	    // SW
	    fVSum[0] = fVSum[0] + (int)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).x*maskLocal[6];    
		fVSum[1] = fVSum[1] + (int)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).y*maskLocal[6];   
		fVSum[2] = fVSum[2] + (int)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset++).z*maskLocal[6]; 

	    // S
		fVSum[0] = fVSum[0] + (int)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).x*maskLocal[7];    
		fVSum[1] = fVSum[1] + (int)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).y*maskLocal[7];   
		fVSum[2] = fVSum[2] + (int)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset++).z*maskLocal[7]; 

	    // SE
		fVSum[0] = fVSum[0] + (int)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).x*maskLocal[8];    
		fVSum[1] = fVSum[1] + (int)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).y*maskLocal[8];   
		fVSum[2] = fVSum[2] + (int)GetDataFromLocalMemory(ucLocalData,iLocalPixOffset).z*maskLocal[8]; 

		
		int sum = 0;
		for( int i = 0 ; i < 9 ; i++)
		{
			sum += maskLocal[i];
		}

		res[0] = fVSum[0] / sum;
		if( res[0] > 255 ) res[0] = 255;
		res[1] = fVSum[1] / sum;
		if( res[1] > 255 ) res[1] = 255;
		res[2] = fVSum[2] / sum;
		if( res[2] > 255 ) res[2] = 255;

	
	    // Write out to GMEM with restored offset
	    if((iDevYPrime < uiDevImageHeight) && (iImagePosX < uiImageWidth))
	    {
		    //setData(ucSource,(char)res[0] ,(char)res[1], (char)res[2], iDevGMEMOffset );
			setData(ucSource,(char)res[0] ,(char)res[1], (char)res[2], iDevGMEMOffset );
	    }
	


}
