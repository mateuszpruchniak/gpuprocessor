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
 * \brief The Maximum filter enhances bright values in the image by increasing its area. Similar to a dilate function each 3x3 (or other window size) is processed for the brightest surrounding pixel. That brightest pixel then becomes the new pixel value at the center of the window.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class MaxFilter :
	public NonLinearFilter
{
public:

	/*!
	* Destructor.
	*/
	~MaxFilter(void);

	/*!
	* Constructor, creates a program object for a context, loads the source code (.cl files) and build the program.
	*/
	MaxFilter(cl_context GPUContext ,GPUTransferManager* transfer);

	/*!
	* Start filtering. Launching GPU processing.
	*/
	bool filter(cl_command_queue GPUCommandQueue);
};

