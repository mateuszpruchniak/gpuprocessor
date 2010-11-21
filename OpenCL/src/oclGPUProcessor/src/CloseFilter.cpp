/*!
 * \file CloseFilter.cpp
 * \brief Close filter, (dilate,erode).
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#include "CloseFilter.h"


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


bool CloseFilter::filter(cl_command_queue GPUCommandQueue)
{
	if(dilate->filter(GPUCommandQueue)) return false;
	if(erode->filter(GPUCommandQueue)) return false;
	return true;
}
