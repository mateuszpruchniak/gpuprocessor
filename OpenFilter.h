/*!
 * \file OpenFilter.h
 * \brief Open filter, (erode,dilate).
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "morphologyfilter.h"
#include "DilateFilter.h"
#include "ErodeFilter.h"


/*!
 * \class OpenFilter
 * \brief Open filter, (erode,dilate).
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class OpenFilter :
	public MorphologyFilter
{
private:
	DilateFilter* dilate;
	ErodeFilter* erode;

public:

	/*!
	* Constructor.
	*/
	OpenFilter(void);

	/*!
	* Destructor.
	*/
	~OpenFilter(void);

	/*!
	* Constructor.
	*/
	OpenFilter(cl_context GPUContext ,GPUTransferManager* transfer);

	/*!
	* Start filtering.
	*/
	bool filter(cl_command_queue GPUCommandQueue);
};
