/*!
 * \file highpassfilter.h
 * \brief Prewitt filter, edge detection algorithm.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */


#pragma once
#include "highpassfilter.h"

/*!
 * \class PrewittFilter
 * \brief Prewitt filter, edge detection algorithm.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class PrewittFilter :
	public HighpassFilter
{
public:

	/*!
	* Constructor.
	*/
	PrewittFilter(void);

	/*!
	* Destructor.
	*/
	~PrewittFilter(void);

	/*!
	* Constructor.
	*/
	PrewittFilter(cl_context GPUContext ,GPUTransferManager* transfer);

	/*!
	* Start processing.
	*/
	void process(cl_command_queue GPUCommandQueue);
};

