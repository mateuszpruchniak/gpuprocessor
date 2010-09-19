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

MorphologyFilter::MorphologyFilter(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName): NonLinearFilter(source,GPUContext,transfer,KernelName)
{

}
