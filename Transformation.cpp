/*!
 * \file Transformation.cpp
 * \brief Transformation class.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#include "Transformation.h"


Transformation::Transformation(void)
{
}


Transformation::~Transformation(void)
{
}

Transformation::Transformation(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName): Filter(source,GPUContext,transfer,KernelName)
{

}

bool Transformation::filter(cl_command_queue GPUCommandQueue)
{
	return false;
}