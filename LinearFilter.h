/*!
 * \file LinearFilter.h
 * \brief Linear filters.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */


#pragma once
#include "contextfilter.h"

/*!
 * \class LinearFilter
 * \brief Linear filters.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class LinearFilter :
	public ContextFilter
{
public:

	/*!
	* Constructor.
	*/
	LinearFilter(void);

	/*!
	* Destructor.
	*/
	~LinearFilter(void);

	/*!
	* Constructor.
	*/
	LinearFilter(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName): ContextFilter(source,GPUContext,transfer,KernelName)
	{}

	/*!
	* Start filtering.
	*/
	bool filter(cl_command_queue GPUCommandQueue);
};

