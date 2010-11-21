/*!
 * \file LowpassFilter.h
 * \brief Lowpass filters.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "LinearFilter.h"

/*!
 * \class LowpassFilter
 * \brief Low pass filtering, otherwise known as "smoothing", is employed to remove high spatial frequency noise from a digital image.
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
	* OpenCL device memory input buffer for mask.
	*/
	cl_mem cmDevBufMask;

	/*!
	* Load mask to buffer.
	*/
	void LoadMask(int* mask, int count,GPUTransferManager* transfer);

public:

	/*!
	* Destructor.
	*/
	~LowpassFilter(void);

	/*!
	* Constructor, creates a program object for a context, loads the source code (.cl files) and build the program.
	*/
	LowpassFilter(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName);

	/*!
	* Start filtering. Launching GPU processing.
	*/
	bool filter(cl_command_queue GPUCommandQueue);


};

