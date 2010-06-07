/*!
 * \file LowpassFilter.h
 * \brief Lowpass filters.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "linearfilter.h"

/*!
 * \class LowpassFilter
 * \brief Lowpass filters.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class LowpassFilter :
	public LinearFilter
{
protected:

	/*!
	* Pointer to mask.
	*/
	int* mask;

	/*!
	* Size of mask. (example mask 3x3, size = 9).
	*/
	int maskSize;

	/*!
	* OpenCL device memory input buffer for mask.
	*/
	cl_mem cmDevBufMask;

	/*!
	* Load mask to buffer.
	*/
	void LoadMask(int* mask, int count,GPUTransferManager* transfer);

public:

	/*!
	* Constructor.
	*/
	LowpassFilter(void);

	/*!
	* Destructor.
	*/
	~LowpassFilter(void);

	/*!
	* Constructor.
	*/
	LowpassFilter(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName);

	/*!
	* Start filtering.
	*/
	bool filter(cl_command_queue GPUCommandQueue);


};

