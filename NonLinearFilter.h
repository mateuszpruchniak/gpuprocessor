/*!
 * \file NonLinearFilter.h
 * \brief Nonlinear filters.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "contextfilter.h"

/*!
 * \class NonLinearFilter
 * \brief Nonlinear filters.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class NonLinearFilter :
	public ContextFilter
{
public:

	/*!
	* Constructor.
	*/
	NonLinearFilter(void);

	/*!
	* Destructor.
	*/
	~NonLinearFilter(void);

	/*!
	* Constructor.
	*/
	NonLinearFilter(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName): ContextFilter(source,GPUContext,transfer,KernelName)
    {
    }

	/*!
	* Start filtering.
	*/
	bool filter(cl_command_queue GPUCommandQueue);
};

