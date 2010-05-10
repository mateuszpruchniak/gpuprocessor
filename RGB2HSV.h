#pragma once
#include "transformation.h"
class RGB2HSV :
	public Transformation
{
public:
	RGB2HSV(void);

	~RGB2HSV(void);

	RGB2HSV(cl_context GPUContext ,GPUTransferManager* transfer): Transformation("./GPUCode.cl",GPUContext,transfer,"ckRGB2HSV")
	{}

	void process(cl_command_queue GPUCommandQueue);

};

