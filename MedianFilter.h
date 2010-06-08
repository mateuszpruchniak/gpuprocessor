/*!
 * \file MedianFilter.h
 * \brief Median filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "nonlinearfilter.h"

/*!
 * \class MedianFilter
 * \brief Median filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class MedianFilter :
	public NonLinearFilter
{
public:

	/*!
	* Constructor.
	*/
	MedianFilter(void);

	/*!
	* Destructor.
	*/
	~MedianFilter(void);

	/*!
	* Constructor.
	*/
	MedianFilter(cl_context GPUContext ,GPUTransferManager* transfer);

	/*!
	* Start filtering.
	*/
	bool filter(cl_command_queue GPUCommandQueue);
};

