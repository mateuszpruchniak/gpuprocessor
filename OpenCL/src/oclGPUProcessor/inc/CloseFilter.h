/*!
 * \file CloseFilter.h
 * \brief File contains class Close filter, (dilate,erode).
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "MorphologyFilter.h"
#include "DilateFilter.h"
#include "ErodeFilter.h"

/*!
 * \class CloseFilter
 * \brief Close filter, (dilate,erode) is the submission of dilation and erosion.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class CloseFilter :
	public MorphologyFilter
{
private:
	/*!
	* Pointer to instance of class DilateFilter.
	*/
	DilateFilter* dilate;

	/*!
	* Pointer to instance of class ErodeFilter.
	*/
	ErodeFilter* erode;

public:

	/*!
	* Constructor. Create ErodeFilter and DilateFilter instances.
	*/
	CloseFilter(cl_context GPUContext ,GPUTransferManager* transfer);

	/*!
	* Destructor.
	*/
	~CloseFilter(void);

	/*!
	* Start dilate filtering, and erode filtering. Launching GPU processing.
	*/
	bool filter(cl_command_queue GPUCommandQueue);
};


