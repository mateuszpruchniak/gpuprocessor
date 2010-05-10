#pragma once
#include "linearfilter.h"
class LowpassFilter :
	public LinearFilter
{
protected:
	int* mask;
public:
	LowpassFilter(void);

	~LowpassFilter(void);

	LowpassFilter(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName): LinearFilter(source,GPUContext,transfer,KernelName)
	{}

	void process(cl_command_queue GPUCommandQueue);


};

