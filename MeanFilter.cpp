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


void MeanFilter::process(cl_command_queue GPUCommandQueue)
{
	LowpassFilter::process(GPUCommandQueue);
}

