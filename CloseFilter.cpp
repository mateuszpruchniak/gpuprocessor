#include "CloseFilter.h"


CloseFilter::CloseFilter(void)
{
}


CloseFilter::~CloseFilter(void)
{
	//cout << "~CloseFilter" << endl;
	delete erode;
	delete dilate;
}


CloseFilter::CloseFilter(cl_context GPUContext ,GPUTransferManager* transfer)
{
	erode = new ErodeFilter(GPUContext,transfer);
	dilate = new DilateFilter(GPUContext,transfer);
}

void CloseFilter::process(cl_command_queue GPUCommandQueue)
{
	dilate->process(GPUCommandQueue);
	erode->process(GPUCommandQueue);
}




