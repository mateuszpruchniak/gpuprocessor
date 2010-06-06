/*!
 * \file MedianFilter.cpp
 * \brief Mean filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#include "MeanFilter.h"


MeanFilter::MeanFilter(void)
{
}


MeanFilter::~MeanFilter(void)
{
}


bool MeanFilter::filter(cl_command_queue GPUCommandQueue)
{
	return LowpassFilter::filter(GPUCommandQueue);
}

