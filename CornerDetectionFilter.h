/*!
 * \file CornerDetectionFilter.h
 * \brief Corner detection filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "highpassfilter.h"

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
	* Constructor.
	*/
	CornerDetectionFilter(void);

	/*!
	* Destructor.
	*/
	~CornerDetectionFilter(void);

	/*!
	* Constructor.
	*/
	CornerDetectionFilter(cl_context GPUContext ,GPUTransferManager* transfer);

	/*!
	* Start processing.
	*/
	void process(cl_command_queue GPUCommandQueue);
};

