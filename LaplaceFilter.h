/*!
 * \file LaplaceFilter.h
 * \brief Laplace filter, edge detection algorithm.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "highpassfilter.h"

/*!
 * \class LaplaceFilter
 * \brief Laplace filter, edge detection algorithm.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class LaplaceFilter :
	public HighpassFilter
{
public:

	/*!
	* Constructor.
	*/
	LaplaceFilter(void);

	/*!
	* Destructor.
	*/
	~LaplaceFilter(void);

	/*!
	* Constructor.
	*/
	LaplaceFilter(cl_context GPUContext ,GPUTransferManager* transfer);

	/*!
	* Start processing.
	*/
	void process(cl_command_queue GPUCommandQueue);

};

