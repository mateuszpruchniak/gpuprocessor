/*!
 * \file MorphologyFilter.h
 * \brief Filter contains class morphology filters.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "NonLinearFilter.h"

/*!
 * \class MorphologyFilter
 * \brief Morphology filters. Morhological transformation change the structure or form of an object in the image.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class MorphologyFilter :
	public NonLinearFilter
{
public:
	/*!
	* Constructor. Doing nothing!
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
	MorphologyFilter(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName);

};

