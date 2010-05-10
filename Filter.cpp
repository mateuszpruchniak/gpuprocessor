#include "Filter.h"

Filter::Filter(void)
{
	GPUFilter = NULL;
	GPUProgram= NULL;
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
	SourceOpenCL = oclLoadProgSource("./OpenCL/GPUCode.cl", "// My comment\n", &szKernelLength);
    SourceOpenCLFilter = oclLoadProgSource(source, "// My comment\n", &szKernelLengthFilter);
	//strncat (SourceOpenCL, SourceOpenCLFilter,szKernelLengthFilter );
	szKernelLengthSum = szKernelLength + szKernelLengthFilter;
	char* sourceCL = new char[szKernelLengthSum];
	strcpy(sourceCL,SourceOpenCL);
	strncat (sourceCL, SourceOpenCLFilter,szKernelLength);

    // creates a program object for a context, and loads the source code specified by the text strings in
    //the strings array into the program object. The devices associated with the program object are the
    //devices associated with context.
    GPUProgram = clCreateProgramWithSource( GPUContext , 1, (const char **)&sourceCL, &szKernelLengthSum, &GPUError);
    CheckError(GPUError);

    // Build the program with 'mad' Optimization option
    char *flags = "-cl-fast-relaxed-math";

    GPUError = clBuildProgram(GPUProgram, 0, NULL, flags, NULL, NULL);
    CheckError(GPUError);

    GPUFilter = clCreateKernel(GPUProgram, KernelName, &GPUError);
}

Filter::~Filter()
{
    //cout << "~Filter" <<endl;
	
    if(GPUProgram)clReleaseProgram(GPUProgram);

    if(GPUFilter)clReleaseKernel(GPUFilter);
	
}


void Filter::CheckError(int code)
{
    switch(code)
    {
    case CL_SUCCESS:
        return;
        break;
    default:
         cout << "OTHERS ERROR" << endl;
    }

    //getchar();
}
