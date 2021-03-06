/*!
 * \file BinarizationFilter.h
 * \brief Binarization filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "ContextFreeFilter.h"

/*!
 * \class BinarizationFilter
 * \brief Binarization filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class BinarizationFilter :
	public ContextFreeFilter
{
public:

	/*!
	* Constructor.
	*/
	BinarizationFilter(void);

	/*!
	* Destructor.
	*/
	~BinarizationFilter(void);

	/*!
	* Constructor.
	*/
	BinarizationFilter(cl_context GPUContext ,GPUTransferManager* transfer, int bin): ContextFreeFilter("./OpenCL/LUTFilter.cl",GPUContext,transfer,"ckLUT")
	{

	}

	/*!
	* Start filtering.
	*/
	bool filter(cl_command_queue GPUCommandQueue);
};

