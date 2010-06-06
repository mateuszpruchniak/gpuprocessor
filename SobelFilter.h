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
 * \brief Sobel filter, edge detection algorithm.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class SobelFilter :
	public HighpassFilter
{
public:

	/*!
	* Constructor.
	*/
	SobelFilter(void);

	/*!
	* Destructor.
	*/
	~SobelFilter(void);

	/*!
	* Constructor.
	*/
	SobelFilter(cl_context GPUContext ,GPUTransferManager* transfer);
	
	/*!
	* Start filtering.
	*/
	bool filter(cl_command_queue GPUCommandQueue);
};

