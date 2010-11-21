/*!
 * \file Filter.cpp
 * \brief Abstract class for all filters
 *
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#include "Filter.h"

Filter::Filter(void)
{

}

Filter::Filter(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName)
{
    GPUTransfer = transfer;

    iBlockDimX = 16;
    iBlockDimY = 16;
    size_t szKernelLength;	
	size_t szKernelLengthFilter;
	size_t szKernelLengthSum;

    // Load OpenCL kernel
	SourceOpenCL = oclLoadProgSource("/home/mateusz/Pulpit/GIT/gpuprocessor/OpenCL/src/oclGPUProcessor/src/OpenCL/GPUCode.cl", "// My comment\n", &szKernelLength);

    SourceOpenCLFilter = oclLoadProgSource(source, "// My comment\n", &szKernelLengthFilter);

	//strncat (SourceOpenCL, SourceOpenCLFilter,szKernelLengthFilter );
	szKernelLengthSum = szKernelLength + szKernelLengthFilter;
	char* sourceCL = new char[szKernelLengthSum];
	strcpy(sourceCL,SourceOpenCL);
	strcat (sourceCL, SourceOpenCLFilter);

    // creates a program object for a context, and loads the source code specified by the text strings in
    //the strings array into the program object. The devices associated with the program object are the
    //devices associated with context.

    
    GPUProgram = clCreateProgramWithSource( GPUContext , 1, (const char **)&sourceCL, &szKernelLengthSum, &GPUError);

	
    CheckError(GPUError);

    // Build the program with 'mad' Optimization option
    char *flags = "-cl-mad-enable";
//cout << sourceCL << endl;
    GPUError = clBuildProgram(GPUProgram, 0, NULL, flags, NULL, NULL);
    CheckErrorBuildProgram(GPUError);
    cout << GPUError << endl;
    GPUFilter = clCreateKernel(GPUProgram, KernelName, &GPUError);

}

Filter::~Filter()
{
    //cout << "~Filter" <<endl;
	
    if(GPUProgram)clReleaseProgram(GPUProgram);

    if(GPUFilter)clReleaseKernel(GPUFilter);
	
}

void Filter::CheckErrorBuildProgram(int code)
{
	switch(code)
	{
		case CL_INVALID_PROGRAM:
			cout << "CL_INVALID_PROGRAM" << endl;
		case CL_INVALID_VALUE:
			cout << "CL_INVALID_VALUE" << endl;
		case CL_INVALID_DEVICE:
			cout << "CL_INVALID_DEVICE" << endl;
		case CL_INVALID_BINARY:
			cout << "CL_INVALID_BINARY" << endl;
		case CL_INVALID_BUILD_OPTIONS:
			cout << "CL_INVALID_BUILD_OPTIONS" << endl;
		case CL_INVALID_OPERATION:
			cout << "CL_INVALID_OPERATION" << endl;
		case CL_COMPILER_NOT_AVAILABLE:
			cout << "CL_COMPILER_NOT_AVAILABLE" << endl;
		case CL_BUILD_PROGRAM_FAILURE:
			cout << "CL_BUILD_PROGRAM_FAILURE" << endl;
		case CL_OUT_OF_HOST_MEMORY:
			cout << "CL_OUT_OF_HOST_MEMORY" << endl;
	}
	
}

void Filter::CheckError(int code)
{
    switch(code)
    {
    case CL_SUCCESS:
        return;
        break;
    default:
         cout << "OTHERS ERROR Filter" << endl;
    }

    //getchar();
}
