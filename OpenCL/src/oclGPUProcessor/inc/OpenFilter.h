/*!
 * \file OpenFilter.h
 * \brief File contains class Open filter, (erode,dilate).
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "MorphologyFilter.h"
#include "DilateFilter.h"
#include "ErodeFilter.h"


/*!
 * \class OpenFilter
 * \brief Open filter, (erode,dilate)  is the submission of erosion and dilation.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class OpenFilter :
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
	* Constructor. Create ErodeFilter and DilateFilter instances. Creates a program object for a context, loads the source code (.cl files) and build the program.
	*/
	OpenFilter(cl_context GPUContext ,GPUTransferManager* transfer);

	/*!
	* Destructor.
	*/
	~OpenFilter(void);

	/*!
	* Start erode filtering, and dilate filtering. Launching GPU processing.
	*/
	bool filter(cl_command_queue GPUCommandQueue);
};
