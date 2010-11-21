/*!
 * \file LaplaceFilter.h
 * \brief Laplace filter, edge detection algorithm.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "HighpassFilter.h"

/*!
 * \class LaplaceFilter
 * \brief Laplace filter, The Laplacian of an image highlights regions of rapid intensity change and is therefore often used for edge detection
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class LaplaceFilter :
	public HighpassFilter
{
public:


	/*!
	* Destructor.
	*/
	~LaplaceFilter(void);

	/*!
	* Constructor. Send mask to GPU memory. Creates a program object for a context, loads the source code (.cl files) and build the program.
	*/
	LaplaceFilter(cl_context GPUContext ,GPUTransferManager* transfer);


};

