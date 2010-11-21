
void GetData(__global uchar* dataIn, __local uchar* dataOut, int iDevGMEMOffset, int iLocalPixOffset, int nChannels)
{
	dataOut[iLocalPixOffset*nChannels] = dataIn[iDevGMEMOffset*nChannels];
	dataOut[iLocalPixOffset*nChannels+1] = dataIn[iDevGMEMOffset*nChannels+1];
	dataOut[iLocalPixOffset*nChannels+2] = dataIn[iDevGMEMOffset*nChannels+2];
	if( nChannels == 4 )
	{
		dataOut[iLocalPixOffset*nChannels+3] = dataIn[iDevGMEMOffset*nChannels+3];
	}
}

void SetZERO(__local uchar* dataOut, int iLocalPixOffset, int nChannels)
{
	dataOut[iLocalPixOffset*nChannels] = (char)0;
	dataOut[iLocalPixOffset*nChannels+1] = (char)0;
	dataOut[iLocalPixOffset*nChannels+2] = (char)0;
	if( nChannels == 4 )
	{
		dataOut[iLocalPixOffset*nChannels+3] = (char)0;
	}
}


uchar4 GetDataFromLocalMemory( __local uchar* data,  int iLocalPixOffset , int nChannels)
{
	uchar4 pix;
	pix.x = data[iLocalPixOffset*nChannels];
	pix.y = data[iLocalPixOffset*nChannels+1];
	pix.z = data[iLocalPixOffset*nChannels+2];
	return pix;
}

void setData(__global char* data, char x , char y, char z, int iDevGMEMOffset , int nChannels)
{
	data[iDevGMEMOffset*nChannels] = x;
	data[iDevGMEMOffset*nChannels+1] = y;
	data[iDevGMEMOffset*nChannels+2] = z;
}

uchar4 GetDataFromGlobalMemory( __global uchar* data,  int iDevGMEMOffset , int nChannels)
{
	uchar4 pix;
	pix.x = data[iDevGMEMOffset*nChannels];
	pix.y = data[iDevGMEMOffset*nChannels+1];
	pix.z = data[iDevGMEMOffset*nChannels+2];
	return pix;
}



void LoadToLocalMemNew(__global uchar* ucSource,__local uchar* ucLocalData, int iLocalPixPitch, 
                      unsigned int uiImageWidth, unsigned int uiDevImageHeight, int nChannels)
{
	    // Get parent image x and y pixel coordinates from global ID, and compute offset into parent GMEM data
	    int iImagePosX = get_global_id(0);
	    int iDevYPrime = get_global_id(1) - 1;  // Shift offset up 1 radius (1 row) for reads
	    int iDevGMEMOffset = mul24(iDevYPrime, (int)get_global_size(0)) + iImagePosX; 

	    // Compute initial offset of current pixel within work group LMEM block
	    int iLocalPixOffset = mul24((int)get_local_id(1), iLocalPixPitch) + get_local_id(0) + 1;

		

	    // Main read of GMEM data into LMEM
	    if((iDevYPrime >= 0) && (iDevYPrime < uiDevImageHeight) && (iImagePosX < uiImageWidth))
	    { 
			GetData(ucSource,ucLocalData,iDevGMEMOffset,iLocalPixOffset,nChannels);
	    }
	    else 
	    {
			SetZERO(ucLocalData,iLocalPixOffset,nChannels);
	    }

	    // Work items with y ID < 2 read bottom 2 rows of LMEM 
	    if (get_local_id(1) < 2)
	    {
			// Increase local offset by 1 workgroup LMEM block height
			// to read in top rows from the next block region down
			iLocalPixOffset += mul24((int)get_local_size(1), iLocalPixPitch);

			// If source offset is within the image boundaries
			if (((iDevYPrime + get_local_size(1)) < uiDevImageHeight) && (iImagePosX < uiImageWidth))
			{
				// Read in top rows from the next block region down
				GetData(ucSource,ucLocalData,iDevGMEMOffset + mul24(get_local_size(1), get_global_size(0)),iLocalPixOffset,nChannels);
			}
			else 
			{
				SetZERO(ucLocalData,iLocalPixOffset,nChannels);
			}
	    }

	    // Work items with x ID at right workgroup edge will read Left apron pixel
	    if (get_local_id(0) == (get_local_size(0) - 1))
	    {
			// set local offset to read data from the next region over
			iLocalPixOffset = mul24((int)get_local_id(1), iLocalPixPitch);

			// If source offset is within the image boundaries and not at the leftmost workgroup
			if ((iDevYPrime >= 0) && (iDevYPrime < uiDevImageHeight) && (get_group_id(0) > 0))
			{
				// Read data into the LMEM apron from the GMEM at the left edge of the next block region over
				GetData(ucSource,ucLocalData,mul24(iDevYPrime, (int)get_global_size(0)) + mul24(get_group_id(0), get_local_size(0)) - 1,iLocalPixOffset,nChannels);
			}
			else 
			{
				SetZERO(ucLocalData,iLocalPixOffset,nChannels);
			}

			// If in the bottom 2 rows of workgroup block 
			if (get_local_id(1) < 2)
			{
				// Increase local offset by 1 workgroup LMEM block height
				// to read in top rows from the next block region down
				iLocalPixOffset += mul24((int)get_local_size(1), iLocalPixPitch);

				// If source offset in the next block down isn't off the image and not at the leftmost workgroup
				if (((iDevYPrime + get_local_size(1)) < uiDevImageHeight) && (get_group_id(0) > 0))
				{
					// read in from GMEM (reaching down 1 workgroup LMEM block height and left 1 pixel)
					GetData(ucSource,ucLocalData,mul24((iDevYPrime + (int)get_local_size(1)), (int)get_global_size(0)) + mul24(get_group_id(0), get_local_size(0)) - 1,iLocalPixOffset,nChannels);
				}
				else 
				{
					SetZERO(ucLocalData,iLocalPixOffset,nChannels);
				}
			}
	    } 
	    else if (get_local_id(0) == 0) // Work items with x ID at left workgroup edge will read right apron pixel
	    {
			// set local offset 
			iLocalPixOffset = mul24(((int)get_local_id(1) + 1), iLocalPixPitch) - 1;

			if ((iDevYPrime >= 0) && (iDevYPrime < uiDevImageHeight) && (mul24(((int)get_group_id(0) + 1), (int)get_local_size(0)) < uiImageWidth))
			{
				// read in from GMEM (reaching left 1 pixel) if source offset is within image boundaries
				GetData(ucSource,ucLocalData,mul24(iDevYPrime, (int)get_global_size(0)) + mul24((get_group_id(0) + 1), get_local_size(0)),iLocalPixOffset,nChannels);
			}
			else 
			{
				SetZERO(ucLocalData,iLocalPixOffset,nChannels);
			}

			// Read bottom 2 rows of workgroup LMEM block
			if (get_local_id(1) < 2)
			{
				// increase local offset by 1 workgroup LMEM block height
				iLocalPixOffset += (mul24((int)get_local_size(1), iLocalPixPitch));

				if (((iDevYPrime + get_local_size(1)) < uiDevImageHeight) && (mul24((get_group_id(0) + 1), get_local_size(0)) < uiImageWidth) )
				{
					// read in from GMEM (reaching down 1 workgroup LMEM block height and left 1 pixel) if source offset is within image boundaries
					GetData(ucSource,ucLocalData,mul24((iDevYPrime + (int)get_local_size(1)), (int)get_global_size(0)) + mul24((get_group_id(0) + 1), get_local_size(0)),iLocalPixOffset,nChannels);
				}
				else 
				{
					SetZERO(ucLocalData,iLocalPixOffset,nChannels);
				}
			}
	    }

        
}
