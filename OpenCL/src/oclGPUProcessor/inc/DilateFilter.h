/*!
 * \file DilateFilter.h
 * \brief Dilate filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "MorphologyFilter.h"

/*!
 * \class DilateFilter
 * \brief Dilate filter. The Dilation filter is a morphological filter that changes the shape of objects in an image by dilating (enlarging) the boundaries of bright objects, and reducing the boundaries of dark ones. The dilation filter can be used to increase the size of small bright objects.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class DilateFilter :
	public MorphologyFilter
{
public:

	/*!
	* Constructor.
	*/
	DilateFilter(void);

	/*!
	* Destructor.
	*/
	~DilateFilter(void);

	/*!
	* Constructor, creates a program object for a context, loads the source code (.cl files) and build the program.
	*/
	DilateFilter(cl_context GPUContext ,GPUTransferManager* transfer);

	/*!
	* Start filtering. Launching GPU processing.
	*/
	bool filter(cl_command_queue GPUCommandQueue);
};

