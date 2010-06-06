/*!
 * \file ErodeFilter.h
 * \brief Erode filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "morphologyfilter.h"

/*!
 * \class ErodeFilter
 * \brief Erode filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class ErodeFilter :
	public MorphologyFilter
{
public:

	/*!
	* Constructor.
	*/
	ErodeFilter(void);

	/*!
	* Destructor.
	*/
	~ErodeFilter(void);

	/*!
	* Constructor.
	*/
	ErodeFilter(cl_context GPUContext ,GPUTransferManager* transfer): MorphologyFilter("./OpenCl/ErodeFilter.cl",GPUContext,transfer,"ckErode")
	{}

	/*!
	* Start filtering.
	*/
	bool filter(cl_command_queue GPUCommandQueue);
};

