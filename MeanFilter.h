/*!
 * \file MedianFilter.h
 * \brief Mean filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "lowpassfilter.h"

/*!
 * \class MeanFilter
 * \brief Mean filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class MeanFilter :
	public LowpassFilter
{

public:

	/*!
	* Constructor.
	*/
	MeanFilter(void);

	/*!
	* Destructor.
	*/
	~MeanFilter(void);

	/*!
	* Constructor.
	*/
	MeanFilter(cl_context GPUContext ,GPUTransferManager* transfer): LowpassFilter("./OpenCL/LowpassFilter.cl",GPUContext,transfer,"ckConv")
	{
		maskSize = 9;
		mask = new int[9];
		for(int i = 0 ; i < 9 ; ++i )
		{
			mask[i] = 1;
		}
	}


	
	/*!
	* Start processing.
	*/
	void process(cl_command_queue GPUCommandQueue);
};

