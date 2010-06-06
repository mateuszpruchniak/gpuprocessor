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
	LowpassFilter(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName): LinearFilter(source,GPUContext,transfer,KernelName)
	{}

	/*!
	* Start filtering.
	*/
	bool filter(cl_command_queue GPUCommandQueue);


};

