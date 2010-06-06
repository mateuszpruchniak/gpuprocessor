/*!
 * \file MedianFilter.h
 * \brief Mean filter with variable central point.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "lowpassfilter.h"

/*!
 * \class MeanVariableCentralPointFilter
 * \brief Mean filter with variable central point.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class MeanVariableCentralPointFilter :
	public LowpassFilter
{

public:

	/*!
	* Constructor.
	*/
	MeanVariableCentralPointFilter(void);

	/*!
	* Destructor.
	*/
	~MeanVariableCentralPointFilter(void);

	/*!
	* Constructor.
	*/
	MeanVariableCentralPointFilter(cl_context GPUContext ,GPUTransferManager* transfer,int central): LowpassFilter("./OpenCL/LowpassFilter.cl",GPUContext,transfer,"ckConv")
	{
		mask = new int[9];
		for(int i = 0 ; i < 9 ; ++i )
		{
			mask[i] = 1;
		}

		mask[4] = central;
	}

	/*!
	* Start processing.
	*/
	void process(cl_command_queue GPUCommandQueue);
};

