/*!
 * \file MedianFilter.h
 * \brief Filte content class mean filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "LowpassFilter.h"

/*!
 * \class MeanFilter
 * \brief Mean filter. Mean filtering is a simple, intuitive and easy to implement method of smoothing images, i.e. reducing the amount of intensity variation between one pixel and the next. It is often used to reduce noise in images.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class MeanFilter :
	public LowpassFilter
{

public:

	/*!
	* Destructor.
	*/
	~MeanFilter(void);

	/*!
	* Constructor. Send mask to GPU memory. Creates a program object for a context, loads the source code (.cl files) and build the program.
	*/
	MeanFilter(cl_context GPUContext ,GPUTransferManager* transfer);
	
};

