/*!
 * \file RGB2HSV.h
 * \brief Color transformation class.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "Transformation.h"

/*!
 * \class RGB2HSV
 * \brief Color transformation class. RGB to HSV.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class RGB2HSV :
	public Transformation
{
public:

	/*!
	* Destructor.
	*/
	~RGB2HSV(void);

	/*!
	* Constructor, creates a program object for a context, loads the source code (.cl files) and build the program.
	*/
	RGB2HSV(cl_context GPUContext ,GPUTransferManager* transfer);

	/*!
	* Start filtering. Launching GPU processing.
	*/
	bool filter(cl_command_queue GPUCommandQueue);

};

