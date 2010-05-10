#include "GPUProcessor.h"



GPUProcessor::GPUProcessor()
{
    //cout << "gpu computing konstr" << endl;
    


	GPUError = oclGetPlatformID(&cpPlatform);
    CheckError(GPUError);

	cl_uint uiNumAllDevs = 0;

	// Get the number of GPU devices available to the platform
    GPUError = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &uiNumAllDevs);
    CheckError(GPUError);
    uiDevCount = uiNumAllDevs;

    // Create the device list
    cdDevices = new cl_device_id [uiDevCount];
    GPUError = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, uiDevCount, cdDevices, NULL);
    CheckError(GPUError);

    size_t szParmDataBytes;	
    
    // Create the OpenCL context on a GPU device
    GPUContext = clCreateContext(0, uiNumAllDevs, cdDevices, NULL, NULL, &GPUError);
    CheckError(GPUError);
    
    
    //The command-queue can be used to queue a set of operations (referred to as commands) in order.
    GPUCommandQueue = clCreateCommandQueue(GPUContext, cdDevices[0], 0, &GPUError);
    CheckError(GPUError);

    oclPrintDevName(LOGBOTH, cdDevices[0]);  
}

void GPUProcessor::CheckError(int code)
{
    switch(code)
    {
    case CL_SUCCESS:
        return;
        break;
   
    default:
         cout << "OTHERS ERROR" << endl;
    }
}

GPUProcessor::~GPUProcessor()
{
    int i = (int)filters.size();
    for( int j = 0 ; j < i ; j++)
    {
        delete filters[j];
    }

    if(GPUCommandQueue)clReleaseCommandQueue(GPUCommandQueue);
    if(GPUContext)clReleaseContext(GPUContext);
}

void GPUProcessor::addFilter(Filter* filter)
{
    filters.push_back(filter);
}

void GPUProcessor::process()
{
    int i = (int)filters.size();
    for( int j = 0 ; j < i ; j++)
    {
        filters[j]->process(GPUCommandQueue);
    }
}
