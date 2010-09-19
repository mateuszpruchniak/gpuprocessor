/*!
 * \file MeanVariableCentralPointFilter.h
 * \brief Filte contains class mean filter with variable central point.
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
	* Destructor.
	*/
	~MeanVariableCentralPointFilter(void);

	/*!
	* Constructor, and send mask to GPU memory. Creates a program object for a context, loads the source code (.cl files) and build the program.
	*/
	MeanVariableCentralPointFilter(cl_context GPUContext ,GPUTransferManager* transfer,int central);

};

