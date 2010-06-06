/*!
 * \file ContextFilter.h
 * \brief Contex filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "filter.h"

/*!
 * \class ContextFilter
 * \brief Contex filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class ContextFilter :
	public Filter
{
public:

	/*!
	* Constructor.
	*/
	ContextFilter(void)
	{}

	/*!
	* Destructor.
	*/
	~ContextFilter(void);

	/*!
	* Constructor.
	*/
	ContextFilter(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName): Filter(source,GPUContext,transfer,KernelName)
    {
    }

	/*!
	* Start filtering.
	*/
	bool filter(cl_command_queue GPUCommandQueue);
};

