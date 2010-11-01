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

// Find the max in local memory array. 
// On return, the max value is in maxValue[0], the index of max element is in maxInd[0]
//*****************************************************************************
void maxOneBlock(__local float maxValue[],
                 __local int   maxInd[],
                 int size)
{
    uint localId   = get_local_id(0);
    uint localSize = get_local_size(0);
    int idx;

    if (localSize < size) 
    {
        for (uint s = localSize; s < size; s += localSize)
        {
            idx = (maxValue[localId] > maxValue[localId + s]) ? localId : localId + s;
            maxValue[localId] = maxValue[idx];
            maxInd[localId] = maxInd[idx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint s = localSize/2; s > 32; s >>= 1)
    {
        if (localId < s) 
        {
            idx = (maxValue[localId] > maxValue[localId + s]) ? localId : localId + s;
            maxValue[localId] = maxValue[idx];
            maxInd[localId] = maxInd[idx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // unroll the final warp to reduce loop and sync overheads
    if (localId < 32)
    {
        idx = (maxValue[localId] > maxValue[localId + 32]) ? localId : localId + 32;
        maxValue[localId] = maxValue[idx];
        maxInd[localId] = maxInd[idx];
        idx = (maxValue[localId] > maxValue[localId + 16]) ? localId : localId + 16;
        maxValue[localId] = maxValue[idx];
        maxInd[localId] = maxInd[idx];
        idx = (maxValue[localId] > maxValue[localId + 8]) ? localId : localId + 8;
        maxValue[localId] = maxValue[idx];
        maxInd[localId] = maxInd[idx];
        idx = (maxValue[localId] > maxValue[localId + 4]) ? localId : localId + 4;
        maxValue[localId] = maxValue[idx];
        maxInd[localId] = maxInd[idx];
        idx = (maxValue[localId] > maxValue[localId + 2]) ? localId : localId + 2;
        maxValue[localId] = maxValue[idx];
        maxInd[localId] = maxInd[idx];
        idx = (maxValue[localId] > maxValue[localId + 1]) ? localId : localId + 1;
        maxValue[localId] = maxValue[idx];
        maxInd[localId] = maxInd[idx];
    }
}

// Performing one step in the Viterbi search main loop
//*****************************************************************************
__kernel void ViterbiOneStep(__global float *maxProbNew,
                             __global int   *path, 
                             __global float *maxProbOld,
                             __global float *mtState,
                             __global float *mtEmit,
                             __local  float maxValue[],
                             __local  int   maxInd[],
                             __local  float maxValueSeg[],
                             __local  int   maxIndSeg[],
                             int nState,
                             int szLmem,
                             int obs,
                             int iObs)
{
    uint numGroups = get_num_groups(0);
    uint groupId   = get_group_id(0);
    uint localId   = get_local_id(0);
    uint localSize = get_local_size(0);

    // loop through all current states, using one work-group for one state
    for (uint s = 0; s < nState; s += numGroups)
    {
        uint iState = groupId + s;

        // loop through all previous states, calculating a segmement of szLmem
        // size per iteration, store the results to temporary array maxValueSeg & maxIndSeg
        for (uint i = 0; i < nState; i += szLmem) 
        {
            // load the data from global to local memory
            if (i + localId < nState) 
            {
                maxValue[localId] = maxProbOld[i + localId] + mtState[iState*nState + i + localId];
                maxInd[localId] = i + localId;
            } else {
                maxValue[localId] = -1;
                maxInd[localId] = -1;
            }
            for (uint offset = localSize; offset < szLmem; offset += localSize)
            {
                if (i + localId + offset < nState)
                {
                    maxValue[localId + offset] = 
                        maxProbOld[i + localId + offset] + mtState[iState*nState + i + localId + offset];
                    maxInd[localId + offset] = i + localId + offset;
                } else {
                    maxValue[localId + offset] = -1;
                    maxInd[localId + offset] = -1;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // find the max on local memory
            maxOneBlock(maxValue, maxInd, szLmem);

            // copy results from local to global memory
            if (localId == 0) 
            {
                maxValueSeg[i/szLmem] = maxValue[0] + mtEmit[obs*nState + iState];
                maxIndSeg[i/szLmem] = maxInd[0];
            }
        }

        // find the max value and index for this group
        if (localId == 0)
        {
            float mValue = -1.0f;
            int mInd = -1;
            int szSeg = (nState % szLmem == 0) ? nState/szLmem : nState/szLmem + 1;
            for (int i = 0; i < szSeg; i++) 
            {
                if (maxValueSeg[i] > mValue)
                {
                    mValue = maxValueSeg[i];
                    mInd = maxIndSeg[i];
                }
            }
            maxProbNew[iState] = mValue;
            path[(iObs-1)*nState + iState] = mInd;
        }
    }
    
}

__kernel void ViterbiPath(__global float *vProb,
                          __global int   *vPath,
                          __global float *maxProbNew,
                          __global int   *path,
                          int nState,
                          int nObs)
{
    // find the final most probable state
    float maxProb = 0.0;
    int maxState = -1;
    for (int i = 0; i < nState; i++) 
    {
        if (maxProbNew[i] > maxProb) 
        {
            maxProb = maxProbNew[i];
            maxState = i;
        }
    }
    *vProb = maxProb;

    // backtrace to find the Viterbi path
    vPath[nObs-1] = maxState;
    for (int t = nObs-2; t >= 0; t--) 
    {
        vPath[t] = path[t*nState + vPath[t+1]];
    }
}