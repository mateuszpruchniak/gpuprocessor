/*!
 * \file ContextFreeFilter.h
 * \brief No contex filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "filter.h"

/*!
 * \class ContextFreeFilter
 * \brief No contex filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class ContextFreeFilter :
	public Filter
{
public:

	/*!
	* Constructor.
	*/
	ContextFreeFilter(void);

	/*!
	* Destructor.
	*/
	~ContextFreeFilter(void);

	/*!
	* Constructor.
	*/
	ContextFreeFilter(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName): Filter(source,GPUContext,transfer,KernelName)
    {
    }
};

