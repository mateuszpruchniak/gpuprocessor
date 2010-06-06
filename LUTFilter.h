/*!
 * \file LUTFilter.h
 * \brief Lookup table filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "contextfreefilter.h"

/*!
 * \class LUTFilter
 * \brief Lookup table filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class LUTFilter :
	public ContextFreeFilter
{
private:

	/*!
	* Pointer to lookup table.
	*/
	int* lut;

	/*!
	* OpenCL device memory input buffer for LUT table.
	*/
	cl_mem cmDevBufLUT;     

	/*!
	* Load lookup table to buffer.
	*/
	void LoadLookUpTable(int* lut, int count,GPUTransferManager* transfer);

public:

	/*!
	* Constructor.
	*/
	LUTFilter(void);

	/*!
	* Destructor.
	*/
	~LUTFilter(void);

	/*!
	* Constructor.
	*/
	LUTFilter(cl_context GPUContext ,GPUTransferManager* transfer, int* LUTArray): ContextFreeFilter("./OpenCL/LUTFilter.cl",GPUContext,transfer,"ckLUT")
	{
		lut = LUTArray;
		LoadLookUpTable(lut , 256, transfer);
	}

	/*!
	* Start filtering.
	*/
	bool filter(cl_command_queue GPUCommandQueue);


	
};

