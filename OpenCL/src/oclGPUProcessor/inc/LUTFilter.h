/*!
 * \file LUTFilter.h
 * \brief Lookup table filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "ContextFreeFilter.h"

/*!
 * \class LUTFilter
 * \brief Color classification, this operation based on Look-Up-Tables.
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

	/*!
	* OpenCL device memory input buffer for LUT table.
	*/
	cl_mem cmDevBufLUT;     

	/*!
	* Load lookup table to buffer.
	*/
	void LoadLookUpTable(int* lut, int count,GPUTransferManager* transfer);

public:


	/*!
	* Destructor.
	*/
	~LUTFilter(void);

	/*!
	* Constructor, creates a program object for a context, loads the source code (.cl files) and build the program.
	*/
	LUTFilter(cl_context GPUContext ,GPUTransferManager* transfer, int* LUTArray);

	/*!
	* Start filtering. Launching GPU processing.
	*/
	bool filter(cl_command_queue GPUCommandQueue);

};

