/*!
 * \file CornerDetectionFilter.cpp
 * \brief Corner detection filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#include "CornerDetectionFilter.h"


CornerDetectionFilter::CornerDetectionFilter(void)
{
}


CornerDetectionFilter::~CornerDetectionFilter(void)
{
}


CornerDetectionFilter::CornerDetectionFilter(cl_context GPUContext ,GPUTransferManager* transfer): HighpassFilter("./OpenCL/HighpassFilter.cl",GPUContext,transfer,"ckGradient")
{
	maskH = new int[9];
	maskV = new int[9];
		
	maskV[0] = 1;
	maskV[1] = 1;
	maskV[2] = 1;
	maskV[3] = -1;
	maskV[4] = -2;
	maskV[5] = 1;
	maskV[6] = -1;
	maskV[7] = -1;
	maskV[8] = 1;

	maskH[0] = 0;
	maskH[1] = 0;
	maskH[2] = 0;
	maskH[3] = 0;
	maskH[4] = 0;
	maskH[5] = 0;
	maskH[6] = 0;
	maskH[7] = 0;
	maskH[8] = 0;
}

bool CornerDetectionFilter::filter(cl_command_queue GPUCommandQueue)
{
	return HighpassFilter::filter(GPUCommandQueue);
}
