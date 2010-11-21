/*!
 * \file PrewittFilter.h
 * \brief Prewitt filter, edge detection algorithm.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */


#pragma once
#include "HighpassFilter.h"

/*!
 * \class PrewittFilter
 * \brief Prewitt filter The Prewitt Edge filter is use to detect edges based applying a horizontal and verticle filter in sequence. Both filters are applied to the image and summed to form the final result. 
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class PrewittFilter :
	public HighpassFilter
{
public:

	/*!
	* Destructor.
	*/
	~PrewittFilter(void);

	/*!
	* Constructor. Send mask to GPU memory. Creates a program object for a context, loads the source code (.cl files) and build the program.
	*/
	PrewittFilter(cl_context GPUContext ,GPUTransferManager* transfer);

};

