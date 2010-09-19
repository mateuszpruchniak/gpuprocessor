/*!
 * \file ContextFreeFilter.cpp
 * \brief No contex filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#include "ContextFreeFilter.h"


ContextFreeFilter::~ContextFreeFilter(void)
{
}

ContextFreeFilter::ContextFreeFilter(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName): Filter(source,GPUContext,transfer,KernelName)
{

}
