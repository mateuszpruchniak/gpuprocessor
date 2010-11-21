/*!
 * \file ContextFreeFilter.h
 * \brief File caontains contex free filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "Filter.h"

/*!
 * \class ContextFreeFilter
 * \brief Context free transformations process given input image pixel into output image pixel independetly of its neighbors.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class ContextFreeFilter :
	public Filter
{
public:

	/*!
	* Destructor.
	*/
	~ContextFreeFilter(void);

	/*!
	* Constructor.
	*/
	ContextFreeFilter(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName);

};

