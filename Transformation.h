#pragma once
#include "filter.h"
class Transformation :
	public Filter
{
public:
	Transformation(void);

	~Transformation(void);

	Transformation(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName): Filter(source,GPUContext,transfer,KernelName)
    {
    }
};

