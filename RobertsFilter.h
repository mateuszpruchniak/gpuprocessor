/*!
 * \file RobertsFilter.h
 * \brief Roberts filter, edge detection algorithm.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "highpassfilter.h"

/*!
 * \class RobertsFilter
 * \brief Roberts filter, The Roberts' Cross operator is used in image processing and computer vision for edge detection.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class RobertsFilter :
	public HighpassFilter
{
public:

	/*!
	* Destructor.
	*/
	~RobertsFilter(void);

	/*!
	* Constructor. Send mask to GPU memory. Creates a program object for a context, loads the source code (.cl files) and build the program.
	*/
	RobertsFilter(cl_context GPUContext ,GPUTransferManager* transfer);

};

