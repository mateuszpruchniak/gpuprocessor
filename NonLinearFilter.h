/*!
 * \file NonLinearFilter.h
 * \brief File contains class Nonlinear filters.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "contextfilter.h"

/*!
 * \class NonLinearFilter
 * \brief Nonlinear filters. In non-linear filters the output pixel value is a result of a non-linear combination on input pixels.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class NonLinearFilter :
	public ContextFilter
{
public:

	/*!
	* Default Constructor.  Nothing doing.
	*/
	NonLinearFilter(void);

	/*!
	* Destructor.
	*/
	~NonLinearFilter(void);

	/*!
	* Constructor.
	*/
	NonLinearFilter(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName);
};

