/*!
 * \file MedianFilter.h
 * \brief Mean filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "lowpassfilter.h"

/*!
 * \class MeanFilter
 * \brief Mean filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class MeanFilter :
	public LowpassFilter
{

public:

	/*!
	* Constructor.
	*/
	MeanFilter(void);

	/*!
	* Destructor.
	*/
	~MeanFilter(void);

	/*!
	* Constructor.
	*/
	MeanFilter(cl_context GPUContext ,GPUTransferManager* transfer);
	
	/*!
	* Start filtering.
	*/
	bool filter(cl_command_queue GPUCommandQueue);
};

