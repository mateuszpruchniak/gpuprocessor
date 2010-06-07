/*!
 * \file HighpassFilter.h
 * \brief Highpass filters.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "linearfilter.h"

/*!
 * \class HighpassFilter
 * \brief Highpass filters.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class HighpassFilter :
	public LinearFilter
{
protected:

	/*!
	* Pointer to first mask.
	*/
	int* maskV;

	/*!
	* Pointer to second mask.
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
	* Constructor.
	*/
	HighpassFilter(void);

	/*!
	* Destructor.
	*/
	~HighpassFilter(void);

	/*!
	* Constructor.
	*/
	HighpassFilter(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName);

	/*!
	* Start filtering.
	*/
	bool filter(cl_command_queue GPUCommandQueue);;

};

