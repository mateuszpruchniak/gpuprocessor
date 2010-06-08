/*!
 * \file MaxFilter.h
 * \brief Maximum filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "nonlinearfilter.h"

/*!
 * \class MaxFilter
 * \brief Maximum filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class MaxFilter :
	public NonLinearFilter
{
public:

	/*!
	* Constructor.
	*/
	MaxFilter(void);

	/*!
	* Destructor.
	*/
	~MaxFilter(void);

	/*!
	* Constructor.
	*/
	MaxFilter(cl_context GPUContext ,GPUTransferManager* transfer);

	/*!
	* Start filtering.
	*/
	bool filter(cl_command_queue GPUCommandQueue);
};

