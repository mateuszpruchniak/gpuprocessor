/*!
 * \file CornerDetectionFilter.h
 * \brief Corner detection filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "HighpassFilter.h"

/*!
 * \class CornerDetectionFilter
 * \brief Corner detection filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class CornerDetectionFilter :
	public HighpassFilter
{
public:

	/*!
	* Destructor.
	*/
	~CornerDetectionFilter(void);

	/*!
	* Constructor. Send mask to GPU memory. Creates a program object for a context, loads the source code (.cl files) and build the program.
	*/
	CornerDetectionFilter(cl_context GPUContext ,GPUTransferManager* transfer);

};

