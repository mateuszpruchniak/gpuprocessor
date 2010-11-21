/*!
 * \file MinFilter.h
 * \brief File contains class Minimal filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "NonLinearFilter.h"

/*!
 * \class MinFilter
 * \brief The Minimum filter enhances dark values in the image by increasing its area. Similar to a dilate function each 3x3 (or other window size) is processed for the darkest surrounding pixel. That darkest pixel then becomes the new pixel value at the center of the window.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class MinFilter :
	public NonLinearFilter
{
public:

	/*!
	* Destructor.
	*/
	~MinFilter(void);

	/*!
	* Constructor, creates a program object for a context, loads the source code (.cl files) and build the program.
	*/
	MinFilter(cl_context GPUContext ,GPUTransferManager* transfer);

	/*!
	* Start filtering. Launching GPU processing.
	*/
	bool filter(cl_command_queue GPUCommandQueue);
};

