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

#include <oclUtils.h>
#include "oclSortingNetworks_common.h"

////////////////////////////////////////////////////////////////////////////////
// Validate sorted keys array (check for integrity and proper order)
// returns 1 if correct/pass, returns 0 if incorrect/fail
////////////////////////////////////////////////////////////////////////////////
extern "C" int validateSortedKeys(
    uint *resKey,
    uint *srcKey,
    uint batch,
    uint N,
    uint numValues,
    uint dir,
    uint *srcHist,
    uint *resHist
){
    shrLog(LOGBOTH, 0, "...validating sorted keys: ");

    if(N < 2){
        shrLog(LOGBOTH, 0, "arrayLength too short, exiting\n");
        return 1;
    }

    for(uint j = 0; j < batch; j++, srcKey += N, resKey += N){
        memset(srcHist, 0, numValues * sizeof(uint));
        memset(resHist, 0, numValues * sizeof(uint));

        //Build histograms for current array
        for(uint i = 0; i < N; i++)
            if( (srcKey[i] < numValues) && (resKey[i] < numValues) ){
                srcHist[srcKey[i]]++;
                resHist[resKey[i]]++;
            }else{
                shrLog(LOGBOTH, 0, "***Set %u key arrays are not limited properly***\n", j);
                return 0;
            }

        //Compare the histograms
        for(uint i = 0; i < numValues; i++)
            if(srcHist[i] != resHist[i]){
                shrLog(LOGBOTH, 0, "***Set %u key histograms do not match***\n", j);
                return 0;
            }

        //Check the ordering
        for(uint i = 0; i < N - 1; i++)
            if( (dir && (resKey[i] > resKey[i + 1])) || (!dir && (resKey[i] < resKey[i + 1])) ){
                shrLog(LOGBOTH, 0, "***Set %u key array is not ordered properly***\n", j);
                return 0;
            }
    }

    //All checks passed
    shrLog(LOGBOTH, 0.0, "OK\n");
    return 1; 
}

////////////////////////////////////////////////////////////////////////////////
// Generate input values from input keys
// Check output values to match output keys by the same translation function
// returns 1 if correct/pass, returns 0 if incorrect/fail
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Value validation / stability check routines
////////////////////////////////////////////////////////////////////////////////
extern "C" void fillValues(
    uint *val,
    uint N
){
    for(uint i = 0; i < N; i++)
        val[i] = i;
}

extern "C" int validateSortedValues(
    uint *resKey,
    uint *resVal,
    uint *srcKey,
    uint batchSize,
    uint arrayLength
){
    int stableFlag = 1;

    shrLog(LOGBOTH, 0.0, "...validating sorted values array: ");
    for(uint j = 0; j < batchSize; j++, resKey += arrayLength, resVal += arrayLength){
        for(uint i = 0; i < arrayLength; i++){
            if( (resVal[i] < j * arrayLength) || (resVal[i] >= (j + 1) * arrayLength) || (resKey[i] != srcKey[resVal[i]]) ){
                shrLog(LOGBOTH, 0, "***corrupted!!!***\n");
                return 0;
            }

            if( (i + 1 < arrayLength) && (resKey[i] == resKey[i + 1]) && (resVal[i] > resVal[i + 1]) )
                stableFlag = 0;
        }
    }

    shrLog(LOGBOTH, 0, "OK\n...stability property: %s\n\n", stableFlag ?  "Stable" : "NOT stable !!!");
    return 1;
}
