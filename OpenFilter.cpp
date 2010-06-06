/*!
 * \file OpenFilter.cpp
 * \brief Open filter, (erode,dilate).
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
#include "OpenFilter.h"


OpenFilter::OpenFilter(void)
{
}


OpenFilter::~OpenFilter(void)
{
	//cout << "~OpenFilter" << endl;
	delete erode;
	delete dilate;
}


OpenFilter::OpenFilter(cl_context GPUContext ,GPUTransferManager* transfer)
{
	//MorphologyFilter("",GPUContext,transfer,"");
	erode = new ErodeFilter(GPUContext,transfer);
	dilate = new DilateFilter(GPUContext,transfer);
}

void OpenFilter::process(cl_command_queue GPUCommandQueue)
{
	erode->process(GPUCommandQueue);
	dilate->process(GPUCommandQueue);
}


