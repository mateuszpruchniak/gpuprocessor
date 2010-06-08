/*!
 * \file MorphologyFilter.cpp
 * \brief Morphology filters.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#include "MorphologyFilter.h"

MorphologyFilter::~MorphologyFilter(void)
{
}

MorphologyFilter::MorphologyFilter(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName): ContextFilter(source,GPUContext,transfer,KernelName)
{

}


bool MorphologyFilter::filter(cl_command_queue GPUCommandQueue)
{
	return false;
}