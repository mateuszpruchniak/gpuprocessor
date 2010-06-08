/*!
 * \file MinFilter.h
 * \brief Minimal filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "nonlinearfilter.h"

/*!
 * \class MinFilter
 * \brief Minimal filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class MinFilter :
	public NonLinearFilter
{
public:

	/*!
	* Constructor.
	*/
	MinFilter(void);

	/*!
	* Destructor.
	*/
	~MinFilter(void);

	/*!
	* Constructor.
	*/
	MinFilter(cl_context GPUContext ,GPUTransferManager* transfer);

	/*!
	* Start filtering.
	*/
	bool filter(cl_command_queue GPUCommandQueue);
};

