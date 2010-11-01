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

#ifndef __CL_BODYSYSTEMOPENCL_H__
    #define __CL_BODYSYSTEMOPENCL_H__

    #include "CL/cl.h"
    #include "CL/cl_gl.h"
    #include "oclBodySystem.h"

    // OpenCL BodySystem: runs on the GPU
    class BodySystemOpenCL : public BodySystem
    {
        public:
            BodySystemOpenCL(int numBodies, cl_device_id dev, cl_context ctx, cl_command_queue cmdq, unsigned int p, unsigned int q, bool usePBO);
            virtual ~BodySystemOpenCL();

            virtual void update(float deltaTime);

            virtual void setSoftening(float softening);
            virtual void setDamping(float damping);

            virtual float* getArray(BodyArray array);
            virtual void   setArray(BodyArray array, const float* data);

            virtual size_t getCurrentReadBuffer() const 
            {
                if (m_bUsePBO) 
                {
                    return m_pboGL[m_currentRead]; 
                } 
                else 
                {
                    return (size_t) m_hPos;
                }
            }

            virtual void synchronizeThreads() const;

        protected: // methods
            BodySystemOpenCL() {}

            virtual void _initialize(int numBodies);
            virtual void _finalize();
            
        protected: // data
            cl_device_id device;
            cl_context cxContext;
            cl_command_queue cqCommandQueue;

            cl_kernel MT_kernel;
            cl_kernel noMT_kernel;

            // CPU data
            float* m_hPos;
            float* m_hVel;

            // GPU data
            cl_mem m_dPos[2];
            cl_mem m_dVel[2];

            bool m_bUsePBO;

            float m_softeningSq;
            float m_damping;

            unsigned int m_pboGL[2];
            cl_mem       m_pboCL[2];
            unsigned int m_currentRead;
            unsigned int m_currentWrite;

            unsigned int m_p;
            unsigned int m_q;
    };

#endif // __CLH_BODYSYSTEMOPENCL_H__
