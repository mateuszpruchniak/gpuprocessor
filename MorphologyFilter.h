/*!
 * \file MorphologyFilter.h
 * \brief Morphology filters.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "contextfilter.h"

/*!
 * \class MorphologyFilter
 * \brief Morphology filters.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class MorphologyFilter :
	public ContextFilter
{
public:
	/*!
	* Constructor.
	*/
	MorphologyFilter(void)
	{}

	/*!
	* Destructor.
	*/
	~MorphologyFilter(void);

	/*!
	* Constructor.
	*/
	MorphologyFilter(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName): ContextFilter(source,GPUContext,transfer,KernelName)
    {
    }
};

