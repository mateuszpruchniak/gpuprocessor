/*!
 * \file SobelFilter.h
 * \brief Sobel filter, edge detection algorithm.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "highpassfilter.h"

/*!
 * \class SobelFilter
 * \brief Sobel filter, edge detection algorithm. The Sobel operator is used in image processing, particularly within edge detection algorithms. Technically, it is a discrete differentiation operator, computing an approximation of the gradient of the image intensity function. 
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class SobelFilter :
	public HighpassFilter
{
public:

	/*!
	* Destructor.
	*/
	~SobelFilter(void);

	/*!
	* Constructor. Send mask to GPU memory. Creates a program object for a context, loads the source code (.cl files) and build the program.
	*/
	SobelFilter(cl_context GPUContext ,GPUTransferManager* transfer);
	
};

