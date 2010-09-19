/*!
 * \file OpenFilter.cpp
 * \brief Open filter, (erode,dilate).
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
#include "OpenFilter.h"


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

bool OpenFilter::filter(cl_command_queue GPUCommandQueue)
{
	if(erode->filter(GPUCommandQueue)) return false;
	if(dilate->filter(GPUCommandQueue)) return false;
	return true;
}


