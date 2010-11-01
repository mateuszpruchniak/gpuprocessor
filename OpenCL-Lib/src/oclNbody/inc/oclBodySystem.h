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

#ifndef __CLH_BODYSYSTEM_H__
    #define __CLH_BODYSYSTEM_H__
    
    #include <stdlib.h>

    enum NBodyConfig
    {
        NBODY_CONFIG_RANDOM,
        NBODY_CONFIG_SHELL,
        NBODY_CONFIG_EXPAND,
        NBODY_NUM_CONFIGS
    };

    // utility function
    void randomizeBodies(NBodyConfig config, float* pos, float* vel, float* color, float clusterScale, 
		         float velocityScale, int numBodies);

    // BodySystem abstract base class
    class BodySystem
    {
    public: // methods
        BodySystem(int numBodies) : m_numBodies(numBodies), m_bInitialized(false) {}
        virtual ~BodySystem() {}

        virtual void update(float deltaTime) = 0;

        enum BodyArray 
        {
            BODYSYSTEM_POSITION,
            BODYSYSTEM_VELOCITY,
        };

        virtual void setSoftening(float softening) = 0;
        virtual void setDamping(float damping) = 0;

        virtual float* getArray(BodyArray array) = 0;
        virtual void   setArray(BodyArray array, const float* data) = 0;
     
        virtual size_t getCurrentReadBuffer() const = 0;

        virtual int    getNumBodies() const { return m_numBodies; }

        virtual void   synchronizeThreads() const {};

    protected: // methods
        BodySystem() {} // default constructor

        virtual void _initialize(int numBodies) = 0;
        virtual void _finalize() = 0;

    protected: // data
        int m_numBodies;
        bool m_bInitialized;
    };

#endif // __CLH_BODYSYSTEM_H__
