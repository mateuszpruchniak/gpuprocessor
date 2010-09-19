/*!
 * \file NonLinearFilter.cpp
 * \brief Nonlinear filters.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#include "NonLinearFilter.h"

NonLinearFilter::NonLinearFilter(void)
{

}

NonLinearFilter::~NonLinearFilter(void)
{
}

NonLinearFilter::NonLinearFilter(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName): ContextFilter(source,GPUContext,transfer,KernelName)
{
}
