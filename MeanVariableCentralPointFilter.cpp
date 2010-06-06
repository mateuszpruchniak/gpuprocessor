/*!
 * \file MedianFilter.h
 * \brief Mean filter with variable central point.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */


#include "MeanVariableCentralPointFilter.h"


MeanVariableCentralPointFilter::MeanVariableCentralPointFilter(void)
{
}


MeanVariableCentralPointFilter::~MeanVariableCentralPointFilter(void)
{
}

bool MeanVariableCentralPointFilter::filter(cl_command_queue GPUCommandQueue)
{
	return LowpassFilter::filter(GPUCommandQueue);
}

