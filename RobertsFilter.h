/*!
 * \file highpassfilter.h
 * \brief Roberts filter, edge detection algorithm.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "highpassfilter.h"

/*!
 * \class RobertsFilter
 * \brief Roberts filter, edge detection algorithm.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class RobertsFilter :
	public HighpassFilter
{
public:

	/*!
	* Constructor.
	*/
	RobertsFilter(void);

	/*!
	* Destructor.
	*/
	~RobertsFilter(void);

	/*!
	* Constructor.
	*/
	RobertsFilter(cl_context GPUContext ,GPUTransferManager* transfer);


	/*!
	* Start filtering.
	*/
	bool filter(cl_command_queue GPUCommandQueue);
};

