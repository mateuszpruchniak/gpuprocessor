/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

/* This sample queries the properties of the CUDA devices present in the system. */

// standard utilities and systems includes
#include <oclUtils.h>

#include <sstream>
#include <fstream>

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, const char** argv) 
{
    // start logs
    shrSetLogFileName ("oclDeviceQuery.txt");
    shrLog(LOGBOTH, 0, "oclDeviceQuery.exe Starting...\n\n"); 
    bool bPassed = true;
    std::string sProfileString = "oclDeviceQuery, Platform Name = ";

    // Get OpenCL platform ID for NVIDIA if avaiable, otherwise default
    shrLog(LOGBOTH, 0, "OpenCL SW Info:\n\n");
    char cBuffer[1024];
    cl_platform_id clSelectedPlatformID = NULL; 
    cl_int ciErrNum = oclGetPlatformID (&clSelectedPlatformID);
    shrCheckError(ciErrNum, CL_SUCCESS);

    // Get OpenCL platform name and version
    ciErrNum = clGetPlatformInfo (clSelectedPlatformID, CL_PLATFORM_NAME, sizeof(cBuffer), cBuffer, NULL);
    if (ciErrNum == CL_SUCCESS)
    {
        shrLog(LOGBOTH, 0, " CL_PLATFORM_NAME: \t%s\n", cBuffer);
        sProfileString += cBuffer;
    } 
    else
    {
        shrLog(LOGBOTH, 0, " Error %i in clGetPlatformInfo Call !!!\n\n", ciErrNum);
        bPassed = false;
    }
    sProfileString += ", Platform Version = ";

    ciErrNum = clGetPlatformInfo (clSelectedPlatformID, CL_PLATFORM_VERSION, sizeof(cBuffer), cBuffer, NULL);
    if (ciErrNum == CL_SUCCESS)
    {
        shrLog(LOGBOTH, 0, " CL_PLATFORM_VERSION: \t%s\n", cBuffer);
        sProfileString += cBuffer;
    } 
    else
    {
        shrLog(LOGBOTH, 0, " Error %i in clGetPlatformInfo Call !!!\n\n", ciErrNum);
        bPassed = false;
    }
    sProfileString += ", SDK Version = ";

    // Log OpenCL SDK Version # (for convenience:  not specific to OpenCL) 
    shrLog(LOGBOTH, 0, " OpenCL SDK Version: \t%s\n\n\n", oclSDKVERSION);
    sProfileString += oclSDKVERSION;
    sProfileString += ", NumDevs = ";

    // Get and log OpenCL device info 
    cl_uint ciDeviceCount;
    cl_device_id *devices;
    shrLog(LOGBOTH, 0, "OpenCL Device Info:\n\n");
    ciErrNum = clGetDeviceIDs (clSelectedPlatformID, CL_DEVICE_TYPE_ALL, 0, NULL, &ciDeviceCount);

    // check for 0 devices found or errors... 
    if (ciDeviceCount == 0)
    {
        shrLog(LOGBOTH, 0, " No devices found supporting OpenCL (return code %i)\n\n", ciErrNum);
        bPassed = false;
        sProfileString += "0";
    } 
    else if (ciErrNum != CL_SUCCESS)
    {
        shrLog(LOGBOTH, 0, " Error %i in clGetDeviceIDs call !!!\n\n", ciErrNum);
        bPassed = false;
    }
    else
    {
        // Get and log the OpenCL device ID's
        shrLog(LOGBOTH, 0, " %u devices found supporting OpenCL:\n\n", ciDeviceCount);
        char cTemp[2];
        #ifdef WIN32
            sprintf_s(cTemp, 2*sizeof(char), "%u", ciDeviceCount);
        #else
            sprintf(cTemp, "%u", ciDeviceCount);
        #endif
        sProfileString += cTemp;
		if ((devices = (cl_device_id*)malloc(sizeof(cl_device_id) * ciDeviceCount)) == NULL)
		{
			shrLog(LOGBOTH, 0, " Failed to allocate memory for devices !!!\n\n");
			bPassed = false;
		}
        ciErrNum = clGetDeviceIDs (clSelectedPlatformID, CL_DEVICE_TYPE_ALL, ciDeviceCount, devices, &ciDeviceCount);
        if (ciErrNum == CL_SUCCESS)
        {
            for(unsigned int i = 0; i < ciDeviceCount; ++i ) 
            {  
                shrLog(LOGBOTH, 0, " ---------------------------------\n");
                clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);
                shrLog(LOGBOTH, 0.0, " Device %s\n", cBuffer);
                shrLog(LOGBOTH, 0, " ---------------------------------\n");
                oclPrintDevInfo(LOGBOTH, devices[i]);
                sProfileString += ", Device = ";
                sProfileString += cBuffer;
            }
        }
        else
        {
            shrLog(LOGBOTH, 0, " Error %i in clGetDeviceIDs call !!!\n\n", ciErrNum);
            bPassed = false;
        }
    }

    // masterlog info
    sProfileString += "\n";
    shrLog(LOGBOTH | MASTER, 0, sProfileString.c_str());

    // Log system info(for convenience:  not specific to OpenCL) 
    shrLog(LOGBOTH, 0, "\nSystem Info: \n\n");
    #ifdef _WIN32
        SYSTEM_INFO stProcInfo;         // processor info struct
        OSVERSIONINFO stOSVerInfo;      // Win OS info struct
        SYSTEMTIME stLocalDateTime;     // local date / time struct 

        // processor
        SecureZeroMemory(&stProcInfo, sizeof(SYSTEM_INFO));
        GetSystemInfo(&stProcInfo);

        // OS
        SecureZeroMemory(&stOSVerInfo, sizeof(OSVERSIONINFO));
        stOSVerInfo.dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
        GetVersionEx(&stOSVerInfo);

        // date and time
        GetLocalTime(&stLocalDateTime); 

        // write time and date to logs
        shrLog(LOGBOTH, 0, " Local Time/Date = %i:%i:%i, %i/%i/%i\n", 
            stLocalDateTime.wHour, stLocalDateTime.wMinute, stLocalDateTime.wSecond, 
            stLocalDateTime.wMonth, stLocalDateTime.wDay, stLocalDateTime.wYear); 

        // write proc and OS info to logs
        shrLog(LOGBOTH, 0, " CPU Arch: %i\n CPU Level: %i\n # of CPU processors: %u\n Windows Build: %u\n Windows Ver: %u.%u\n\n\n", 
            stProcInfo.wProcessorArchitecture, stProcInfo.wProcessorLevel, stProcInfo.dwNumberOfProcessors, 
            stOSVerInfo.dwBuildNumber, stOSVerInfo.dwMajorVersion, stOSVerInfo.dwMinorVersion);
    #endif

    #ifdef MAC
	#else
    #ifdef UNIX
        char timestr[255];
        time_t now = time(NULL);
        struct tm  *ts;

        ts = localtime(&now);
        
        strftime(timestr, 255, " %H:%M:%S, %m/%d/%Y",ts);
        
        // write time and date to logs
        shrLog(LOGBOTH, 0, " Local Time/Date = %s\n", 
            timestr); 

        // write proc and OS info to logs
        
        // parse /proc/cpuinfo
        std::ifstream cpuinfo( "/proc/cpuinfo" ); // open the file in /proc        
        std::string tmp;

        int cpu_num = 0;
        std::string cpu_name = "none";        

        do
		{
            cpuinfo >> tmp;
            
            if( tmp == "processor" )
                cpu_num++;
            
            if( tmp == "name" )
			{
                cpuinfo >> tmp; // skip :

                std::stringstream tmp_stream("");
                do
				{
                    cpuinfo >> tmp;
                    if (tmp != string("stepping"))
					{
                        tmp_stream << tmp.c_str() << " ";
                    }
                    
                }
				while (tmp != string("stepping"));
                
                cpu_name = tmp_stream.str();
            }

        }
		while ( (! cpuinfo.eof()) );

        // Linux version
        std::ifstream version( "/proc/version" );
        char versionstr[255];

        version.getline(versionstr, 255);

        shrLog(LOGBOTH, 0, " CPU Name: %s\n # of CPU processors: %u\n %s\n\n\n", 
               cpu_name.c_str(),cpu_num,versionstr);
    #endif
    #endif

    // finish
    shrLog(LOGBOTH, 0, "TEST %s\n\n", bPassed ? "PASSED" : "FAILED !!!"); 
    shrEXIT(argc, argv);
}
