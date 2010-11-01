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

#define BLOCKDIM 256
#define LOOP_UNROLL 4

// Macros to simplify shared memory addressing
#define SX(i) sharedPos[i + mul24(get_local_size(0), get_local_id(1))]

// This macro is only used the multithreadBodies (MT) versions of kernel code below
#ifdef MAC
#define SX_SUM(i,j) sharedPos[i + mul24(get_local_size(0), (uint)j)]    // i + blockDimx * j
#else
#define SX_SUM(i,j) sharedPos[i + mul24(get_local_size(0), j)]    // i + blockDimx * j
#endif

float4 bodyBodyInteraction(float4 ai, float4 bi, float4 bj, float softeningSquared) 
{
    float4 r;

    // r_ij  [3 FLOPS]
    r.x = bi.x - bj.x;
    r.y = bi.y - bj.y;
    r.z = bi.z - bj.z;
    r.w = 0;

    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    distSqr += softeningSquared;

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    float invDist = rsqrt(distSqr);
	float invDistCube =  invDist * invDist * invDist;

    //float distSixth = distSqr * distSqr * distSqr;
    //float invDistCube = 1.0f / sqrt(distSixth);
    
    // s = m_j * invDistCube [1 FLOP]
    float s = bj.w * invDistCube;

    // a_i =  a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}

// This is the "tile_calculation" function from the GPUG3 article.
float4 gravitation(float4 myPos, float4 accel, float softeningSquared, __local float4* sharedPos)
{
    // The CUDA 1.1 compiler cannot determine that i is not going to 
    // overflow in the loop below.  Therefore if int is used on 64-bit linux 
    // or windows (or long instead of long long on win64), the compiler
    // generates suboptimal code.  Therefore we use long long on win64 and
    // long on everything else. (Workaround for Bug ID 347697)
#ifdef _Win64
    unsigned long long i = 0;
#else
    unsigned long i = 0;
#endif

    // Here we unroll the loop

    // Note that having an unsigned int loop counter and an unsigned
    // long index helps the compiler generate efficient code on 64-bit
    // OSes.  The compiler can't assume the 64-bit index won't overflow
    // so it incurs extra integer operations.  This is a standard issue
    // in porting 32-bit code to 64-bit OSes.
    int blockDimx = get_local_size(0);
    for (unsigned int counter = 0; counter < blockDimx; ) 
    {
        accel = bodyBodyInteraction(accel, SX(i++), myPos, softeningSquared); 
	counter++;
#if LOOP_UNROLL > 1
        accel = bodyBodyInteraction(accel, SX(i++), myPos, softeningSquared); 
	counter++;
#endif
#if LOOP_UNROLL > 2
        accel = bodyBodyInteraction(accel, SX(i++), myPos, softeningSquared); 
        accel = bodyBodyInteraction(accel, SX(i++), myPos, softeningSquared); 
	counter += 2;
#endif
#if LOOP_UNROLL > 4
        accel = bodyBodyInteraction(accel, SX(i++), myPos, softeningSquared); 
        accel = bodyBodyInteraction(accel, SX(i++), myPos, softeningSquared); 
        accel = bodyBodyInteraction(accel, SX(i++), myPos, softeningSquared); 
        accel = bodyBodyInteraction(accel, SX(i++), myPos, softeningSquared); 
	counter += 4;
#endif
    }

    return accel;
}

// WRAP is used to force each block to start working on a different 
// chunk (and wrap around back to the beginning of the array) so that
// not all multiprocessors try to read the same memory locations at 
// once.
#define WRAP(x,m) (((x)<m)?(x):(x-m))  // Mod without divide, works on values from 0 up to 2m

float4
computeBodyAccel_MT(float4 bodyPos, __global float4* positions, int numBodies, float softeningSquared, __local float4* sharedPos)
{

    float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
    
    int threadIdxx = get_local_id(0);
    int threadIdxy = get_local_id(1);
    int blockIdxx = get_group_id(0);
    int blockIdxy = get_group_id(1);
    int gridDimx = get_num_groups(0);
    int blockDimx = get_local_size(0);
    int blockDimy = get_local_size(1);
    int p = blockDimx;
    int q = blockDimy;
    int n = numBodies;
    int numTiles = n / mul24(p, q);

    for (int tile = blockIdxy; tile < numTiles + blockIdxy; tile++) 
    {
        sharedPos[threadIdxx + blockDimx * threadIdxy] = 
            positions[WRAP(blockIdxx + mul24(q, tile) + threadIdxy, gridDimx) * p
                      + threadIdxx];
       
        // __syncthreads();
        barrier(CLK_LOCAL_MEM_FENCE);

        // This is the "tile_calculation" function from the GPUG3 article.
        acc = gravitation(bodyPos, acc, softeningSquared, sharedPos);
        
        // __syncthreads();
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // When the numBodies / thread block size is < # multiprocessors (16 on G80), the GPU is 
    // underutilized.  For example, with a 256 threads per block and 1024 bodies, there will only 
    // be 4 thread blocks, so the GPU will only be 25% utilized. To improve this, we use multiple 
    // threads per body.  We still can use blocks of 256 threads, but they are arranged in q rows 
    // of p threads each.  Each thread processes 1/q of the forces that affect each body, and then 
    // 1/q of the threads (those with threadIdx.y==0) add up the partial sums from the other 
    // threads for that body.  To enable this, use the "--p=" and "--q=" command line options to 
    // this example. e.g.: "nbody.exe --n=1024 --p=64 --q=4" will use 4 threads per body and 256 
    // threads per block. There will be n/p = 16 blocks, so a G80 GPU will be 100% utilized.

    // We use a bool template parameter to specify when the number of threads per body is greater 
    // than one, so that when it is not we don't have to execute the more complex code required!
        SX_SUM(threadIdxx, threadIdxy).x = acc.x;
        SX_SUM(threadIdxx, threadIdxy).y = acc.y;
        SX_SUM(threadIdxx, threadIdxy).z = acc.z;

        barrier(CLK_LOCAL_MEM_FENCE);//__syncthreads();

        // Save the result in global memory for the integration step
        if ( get_local_id(0) == 0) 
        {
            for (int i = 1; i < blockDimy; i++) 
            {
                acc.x += SX_SUM(threadIdxx, i).x;
                acc.y += SX_SUM(threadIdxx, i).y;
                acc.z += SX_SUM(threadIdxx, i).z;
            }
        }

    return acc;
}

float4
computeBodyAccel_noMT(float4 bodyPos, __global float4* positions, int numBodies, float softeningSquared, __local float4* sharedPos)
{
    float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
    
    int threadIdxx = get_local_id(0);
    int threadIdxy = get_local_id(1);
    int blockIdxx = get_group_id(0);
    int blockIdxy = get_group_id(1);
    int gridDimx = get_num_groups(0);
    int blockDimx = get_local_size(0);
    int blockDimy = get_local_size(1);
    int p = blockDimx;
    int q = blockDimy;
    int n = numBodies;
    int numTiles = n / mul24(p, q);

    for (int tile = blockIdxy; tile < numTiles + blockIdxy; tile++) 
    {
        sharedPos[threadIdxx + mul24(blockDimx, threadIdxy)] = 
            positions[WRAP(blockIdxx + tile, gridDimx) * p + threadIdxx];
       
        barrier(CLK_LOCAL_MEM_FENCE);// __syncthreads();

        // This is the "tile_calculation" function from the GPUG3 article.
        acc = gravitation(bodyPos, acc, softeningSquared, sharedPos);
        
        barrier(CLK_LOCAL_MEM_FENCE);// __syncthreads();
    }

    // When the numBodies / thread block size is < # multiprocessors (16 on G80), the GPU is 
    // underutilized.  For example, with a 256 threads per block and 1024 bodies, there will only 
    // be 4 thread blocks, so the GPU will only be 25% utilized. To improve this, we use multiple 
    // threads per body.  We still can use blocks of 256 threads, but they are arranged in q rows 
    // of p threads each.  Each thread processes 1/q of the forces that affect each body, and then 
    // 1/q of the threads (those with threadIdx.y==0) add up the partial sums from the other 
    // threads for that body.  To enable this, use the "--p=" and "--q=" command line options to 
    // this example. e.g.: "nbody.exe --n=1024 --p=64 --q=4" will use 4 threads per body and 256 
    // threads per block. There will be n/p = 16 blocks, so a G80 GPU will be 100% utilized.

    // We use a bool template parameter to specify when the number of threads per body is greater 
    // than one, so that when it is not we don't have to execute the more complex code required!

    return acc;
}

__kernel void
integrateBodies_MT(
            __global float4* newPos,
            __global float4* newVel, 
            __global float4* oldPos,
            __global float4* oldVel,
            float deltaTime,
            float damping,
            float softeningSquared,
            int numBodies,
            __local float4* sharedPos)
{
    int threadIdxx = get_local_id(0);
    int threadIdxy = get_local_id(1);
    int blockIdxx = get_group_id(0);
    int blockIdxy = get_group_id(1);
    int gridDimx = get_num_groups(0);
    int blockDimx = get_local_size(0);
    int blockDimy = get_local_size(1);

    int index = mul24(blockIdxx, blockDimx) + threadIdxx;
    float4 pos = oldPos[index];   

    float4 accel = computeBodyAccel_MT(pos, oldPos, numBodies, softeningSquared, sharedPos);

    // acceleration = force \ mass; 
    // new velocity = old velocity + acceleration * deltaTime
    // note we factor out the body's mass from the equation, here and in bodyBodyInteraction 
    // (because they cancel out).  Thus here force == acceleration
    float4 vel = oldVel[index];
       
    vel.x += accel.x * deltaTime;
    vel.y += accel.y * deltaTime;
    vel.z += accel.z * deltaTime;  

    vel.x *= damping;
    vel.y *= damping;
    vel.z *= damping;
        
    // new position = old position + velocity * deltaTime
    pos.x += vel.x * deltaTime;
    pos.y += vel.y * deltaTime;
    pos.z += vel.z * deltaTime;

    // store new position and velocity
    newPos[index] = pos;
    newVel[index] = vel;
}

__kernel void
integrateBodies_noMT(
            __global float4* newPos,
            __global float4* newVel, 
            __global float4* oldPos,
            __global float4* oldVel,
            float deltaTime,
            float damping,
            float softeningSquared,
            int numBodies,
            __local float4* sharedPos)
{
    int threadIdxx = get_local_id(0);
    int threadIdxy = get_local_id(1);
    int blockIdxx = get_group_id(0);
    int blockIdxy = get_group_id(1);
    int gridDimx = get_num_groups(0);
    int blockDimx = get_local_size(0);
    int blockDimy = get_local_size(1);

    int index = mul24(blockIdxx, blockDimx) + threadIdxx;
    float4 pos = oldPos[index];   
    float4 accel = computeBodyAccel_noMT(pos, oldPos, numBodies, softeningSquared, sharedPos);

    // acceleration = force \ mass; 
    // new velocity = old velocity + acceleration * deltaTime
    // note we factor out the body's mass from the equation, here and in bodyBodyInteraction 
    // (because they cancel out).  Thus here force == acceleration
    float4 vel = oldVel[index];
       
    vel.x += accel.x * deltaTime;
    vel.y += accel.y * deltaTime;
    vel.z += accel.z * deltaTime;  

    vel.x *= damping;
    vel.y *= damping;
    vel.z *= damping;
        
    // new position = old position + velocity * deltaTime
    pos.x += vel.x * deltaTime;
    pos.y += vel.y * deltaTime;
    pos.z += vel.z * deltaTime;

    // store new position and velocity
    newPos[index] = pos;
    newVel[index] = vel;
}

