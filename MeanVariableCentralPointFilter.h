/*!
 * \file MedianFilter.h
 * \brief Mean filter with variable central point.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once
#include "lowpassfilter.h"

/*!
 * \class MeanVariableCentralPointFilter
 * \brief Mean filter with variable central point.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class MeanVariableCentralPointFilter :
	public LowpassFilter
{

public:

	/*!
	* Constructor.
	*/
	MeanVariableCentralPointFilter(void);

	/*!
	* Destructor.
	*/
	~MeanVariableCentralPointFilter(void);

	/*!
	* Constructor.
	*/
	MeanVariableCentralPointFilter(cl_context GPUContext ,GPUTransferManager* transfer,int central);

	/*!
	* Start filtering.
	*/
	bool filter(cl_command_queue GPUCommandQueue);
};

