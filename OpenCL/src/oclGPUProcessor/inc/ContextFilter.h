/*!
 * \file ContextFilter.h
 * \brief Contex filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "Filter.h"

/*!
 * \class ContextFilter
 * \brief Contex filter. Context transformation compute the value of given output image pixel on the base a of its neighbors and a mask.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class ContextFilter :
	public Filter
{
public:

	/*!
	* Default constructor. Nothing doing.
	*/
	ContextFilter(void);

	/*!
	* Destructor.
	*/
	~ContextFilter(void);

	/*!
	* Constructor.
	*/
	ContextFilter(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName);

};

