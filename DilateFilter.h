/*!
 * \file DilateFilter.h
 * \brief Dilate filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "morphologyfilter.h"

/*!
 * \class DilateFilter
 * \brief Dilate filter.
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
	* Constructor.
	*/
	DilateFilter(cl_context GPUContext ,GPUTransferManager* transfer): MorphologyFilter("./OpenCL/DilateFilter.cl",GPUContext,transfer,"ckDilate")
	{}

	/*!
	* Start processing.
	*/
	void process(cl_command_queue GPUCommandQueue);
};

