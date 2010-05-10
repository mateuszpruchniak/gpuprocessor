#pragma once
#include "linearfilter.h"
class HighpassFilter :
	public LinearFilter
{
protected:
	int* maskV;
	int* maskH;
public:
	HighpassFilter(void);

	~HighpassFilter(void);

	HighpassFilter(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName);

	void process(cl_command_queue GPUCommandQueue);

};

