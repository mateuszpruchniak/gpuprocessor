/*!
 * \file ErodeFilter.h
 * \brief Filte contains class Erode filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "MorphologyFilter.h"

/*!
 * \class ErodeFilter
 * \brief Erode filter. The Erosion filter is a morphological filter that changes the shape of objects in an image by eroding (reducing) the boundaries of bright objects, and enlarging the boundaries of dark ones. It is often used to reduce, or eliminate, small bright objects. 
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class ErodeFilter :
	public MorphologyFilter
{
public:

	/*!
	* Destructor.
	*/
	~ErodeFilter(void);

	/*!
	* Constructor, creates a program object for a context, loads the source code (.cl files) and build the program.
	*/
	ErodeFilter(cl_context GPUContext ,GPUTransferManager* transfer);

	/*!
	* Start filtering. Launching GPU processing.
	*/
	bool filter(cl_command_queue GPUCommandQueue);
};

