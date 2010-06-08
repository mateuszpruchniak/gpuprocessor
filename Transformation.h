/*!
 * \file Transformation.h
 * \brief Transformation class.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "filter.h"

/*!
 * \class Transformation
 * \brief Transformation class.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class Transformation :
	public Filter
{
public:

	/*!
	* Default constructor.
	*/
	Transformation(void);

	/*!
	* Destructor.
	*/
	~Transformation(void);

	/*!
	* Constructor.
	*/
	Transformation(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName);

	/*!
	* Start filtering.
	*/
	bool filter(cl_command_queue GPUCommandQueue);
};