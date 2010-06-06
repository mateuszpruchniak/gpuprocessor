/*!
 * \file LUTFilter.h
 * \brief Lookup table filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "contextfreefilter.h"

/*!
 * \class LUTFilter
 * \brief Lookup table filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class LUTFilter :
	public ContextFreeFilter
{
private:

	/*!
	* Pointer to lookup table.
	*/
	int* lut;
public:

	/*!
	* Constructor.
	*/
	LUTFilter(void);

	/*!
	* Destructor.
	*/
	~LUTFilter(void);

	/*!
	* Constructor.
	*/
	LUTFilter(cl_context GPUContext ,GPUTransferManager* transfer, int* LUTArray): ContextFreeFilter("./OpenCL/LUTFilter.cl",GPUContext,transfer,"ckLUT")
	{
		lut = LUTArray;
	}

	/*!
	* Start processing.
	*/
	void process(cl_command_queue GPUCommandQueue);

};

