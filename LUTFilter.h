#pragma once
#include "contextfreefilter.h"
class LUTFilter :
	public ContextFreeFilter
{
private:
	int* lut;
public:
	LUTFilter(void);

	~LUTFilter(void);

	LUTFilter(cl_context GPUContext ,GPUTransferManager* transfer, int* LUTArray): ContextFreeFilter("./OpenCL/LUTFilter.cl",GPUContext,transfer,"ckLUT")
	{
		lut = LUTArray;
	}

	void process(cl_command_queue GPUCommandQueue);

};

