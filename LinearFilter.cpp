/*!
 * \file LinearFilter.h
 * \brief Linear filters.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#include "LinearFilter.h"


LinearFilter::LinearFilter(void)
{
}


LinearFilter::~LinearFilter(void)
{
}

LinearFilter::LinearFilter(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName): ContextFilter(source,GPUContext,transfer,KernelName)
{

}

bool LinearFilter::filter(cl_command_queue GPUCommandQueue)
{
	return false;
}
