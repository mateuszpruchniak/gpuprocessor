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

#define USE_LOCAL_MEM


// macros to make indexing shared memory easier
#define SMEM(X, Y) sdata[(Y)*tilew+(X)]

int iclamp(int x, int a, int b)
{
    return max(a, min(b, x));
}

// convert floating point rgb color to 8-bit integer
uint rgbToInt(float r, float g, float b)
{
  r = clamp(r, 0.0f, 255.0f);
  g = clamp(g, 0.0f, 255.0f);
  b = clamp(b, 0.0f, 255.0f);
  return (convert_uint(b)<<16) + (convert_uint(g)<<8) + convert_uint(r);
}

// get pixel from 2D image, with clamping to border
uint getPixel(__global uint *data, int x, int y, int width, int height)
{
    x = iclamp(x, 0, width-1);
    y = iclamp(y, 0, height-1);
    return data[y*width+x];
}

/*
    2D convolution using local memory
    - operates on 8-bit RGB data stored in 32-bit uint
    - assumes kernel radius is less than or equal to block size
    - not optimized for performance
     _____________
    |   :     :   |
    |_ _:_____:_ _|
    |   |     |   |
    |   |     |   |
    |_ _|_____|_ _|
  r |   :     :   |
    |___:_____:___|
      r    bw   r
    <----tilew---->
*/

__kernel void postprocess(__global uint* g_data, __global uint* g_odata, int imgw, int imgh, int tilew, int radius, float threshold, float highlight, __local uint* sdata)
{
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int bw = get_local_size(0);
    const int bh = get_local_size(1);
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if( x >= imgw || y >= imgh ) return;

#ifdef USE_LOCAL_MEM   
    // copy tile to shared memory
    // center region
    SMEM(radius + tx, radius + ty) = getPixel(g_data, x, y, imgw, imgh);

    // borders
    if (tx < radius) {
        // left
        SMEM(tx, radius + ty) = getPixel(g_data, x - radius, y, imgw, imgh);
        // right
        SMEM(radius + bw + tx, radius + ty) = getPixel(g_data, x + bw, y, imgw, imgh);
    }
    if (ty < radius) {
        // top
        SMEM(radius + tx, ty) = getPixel(g_data, x, y - radius, imgw, imgh);
        // bottom
        SMEM(radius + tx, radius + bh + ty) = getPixel(g_data, x, y + bh, imgw, imgh);
    }

    // load corners
    if ((tx < radius) && (ty < radius)) {
        // tl
        SMEM(tx, ty) = getPixel(g_data, x - radius, y - radius, imgw, imgh);
        // bl
        SMEM(tx, radius + bh + ty) = getPixel(g_data, x - radius, y + bh, imgw, imgh);
        // tr
        SMEM(radius + bw + tx, ty) = getPixel(g_data, x + bh, y - radius, imgw, imgh);
        // br
        SMEM(radius + bw + tx, radius + bh + ty) = getPixel(g_data, x + bw, y + bh, imgw, imgh);
    }

    // wait for loads to complete
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    // perform convolution
     float rsum = .0f;
     float gsum = 0.0f;
     float bsum = 0.0f;
     float samples = 0.0f;
     

     for(int iy=0; iy<=radius+radius+1; iy++) {
       for(int ix=0; ix<=radius+radius+1; ix++) {
	 int dx = ix - radius;
	 int dy = iy - radius;

#ifdef USE_LOCAL_MEM
	 uint pixel = SMEM(radius+tx+dx, radius+ty+dy);
#else
	 uint pixel = getPixel(g_data, x+dx, y+dy, imgw, imgh);
#endif
	 
	 // only sum pixels within disc-shaped kernel
	 float l = dx*dx + dy*dy;
	 if (l <= radius*radius) {
	   float r = convert_float(pixel&0x0ff);
	   float g = convert_float((pixel>>8)&0x0ff);
	   float b = convert_float((pixel>>16)&0x0ff);
#if 1
	   // brighten highlights
	   float lum = (r + g + b) / (255*3);
	   if (lum > threshold) {
	     r *= highlight;
	     g *= highlight;
	     b *= highlight;
	   }
#endif
	   rsum += r;
	   gsum += g;
	   bsum += b;
	   samples += 1.0f;
	 }
       }
     }

     rsum /= samples;
     gsum /= samples;
     bsum /= samples;

    g_odata[y*imgw+x] = rgbToInt(rsum, gsum, bsum);
}

