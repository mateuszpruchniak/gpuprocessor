/*!
 * \file CloseFilter.h
 * \brief Close filter, (dilate,erode).
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "morphologyfilter.h"
#include "DilateFilter.h"
#include "ErodeFilter.h"

/*!
 * \class CloseFilter
 * \brief Close filter, (dilate,erode).
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class CloseFilter :
	public MorphologyFilter
{
private:
	/*!
	* Pointer to DilateFilter.
	*/
	DilateFilter* dilate;

	/*!
	* Pointer to ErodeFilter.
	*/
	ErodeFilter* erode;

public:

	/*!
	* Constructor.
	*/
	CloseFilter(void);

	/*!
	* Destructor.
	*/
	~CloseFilter(void);

	/*!
	* Constructor.
	*/
	CloseFilter(cl_context GPUContext ,GPUTransferManager* transfer);

	/*!
	* Start processing.
	*/
	void process(cl_command_queue GPUCommandQueue);
};


