/*!
 * \file ContextFilter.cpp
 * \brief Contex filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#include "ContextFilter.h"


ContextFilter::~ContextFilter(void)
{
}

ContextFilter::ContextFilter(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName): Filter(source,GPUContext,transfer,KernelName)
{

}


bool ContextFilter::filter(cl_command_queue GPUCommandQueue)
{
	return false;
}
