/*!
 * \file HighpassFilter.h
 * \brief Highpass filters.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "nonlinearfilter.h"

/*!
 * \class HighpassFilter
 * \brief Highpass filters. A high pass filter tends to retain the high frequency information within an image while reducing the low frequency information.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class HighpassFilter :
	public NonLinearFilter
{
protected:

	/*!
	* Pointer to vertical mask.
	*/
	int* maskV;

	/*!
	* Pointer to horizontal mask.
	*/
	int* maskH;

	/*!
	* OpenCL device memory input buffer for mask (vertical).
	*/
	cl_mem cmDevBufMaskV;

	/*!
	* OpenCL device memory input buffer for mask (horizontal).
	*/
	cl_mem cmDevBufMaskH;

	/*!
	* Load mask to buffer.
	*/
	void LoadMask(cl_mem* cmDevBufMask,int* mask,int count,GPUTransferManager* transfer);

public:
	

	/*!
	* Destructor.
	*/
	~HighpassFilter(void);

	/*!
	* Constructor, creates a program object for a context, loads the source code (.cl files) and build the program. Start GPU processing.
	*/
	HighpassFilter(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName);

	/*!
	* Start filtering. Launching GPU processing.
	*/
	bool filter(cl_command_queue GPUCommandQueue);;

};

