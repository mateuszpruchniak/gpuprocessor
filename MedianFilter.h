/*!
 * \file MedianFilter.h
 * \brief File contains class Median filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "nonlinearfilter.h"

/*!
 * \class MedianFilter
 * \brief Median filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class MedianFilter :
	public NonLinearFilter
{
public:

	/*!
	* Destructor.
	*/
	~MedianFilter(void);

	/*!
	* Constructor, creates a program object for a context, loads the source code (.cl files) and build the program.
	*/
	MedianFilter(cl_context GPUContext ,GPUTransferManager* transfer);

	/*!
	* Start filtering. Launching GPU processing.
	*/
	bool filter(cl_command_queue GPUCommandQueue);
};

