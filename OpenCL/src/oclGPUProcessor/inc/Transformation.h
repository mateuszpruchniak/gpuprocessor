/*!
 * \file Transformation.h
 * \brief Filte contain color transformation class.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "ContextFreeFilter.h"

/*!
 * \class Transformation
 * \brief Color transformation class.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class Transformation :
	public ContextFreeFilter
{
public:

	/*!
	* Destructor.
	*/
	~Transformation(void);

	/*!
	* Constructor, creates a program object for a context, loads the source code (.cl files) and build the program.
	*/
	Transformation(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName);

	
};
