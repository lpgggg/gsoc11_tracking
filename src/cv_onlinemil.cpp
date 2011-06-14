//
//
//                        Intel License Agreement
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include "cv_onlinemil.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <new>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cassert>
#include <algorithm> 
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <list>
#include <math.h>

#ifdef _OPENMP
#include "omp.h"
#endif

#include <iostream>
#include <stdio.h>
#include <memory.h>
#include <limits>

#include <cv.h>


// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License 
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

using namespace std;


#ifndef HAVE_IPP
/*****************************************************************************/
IppStatus ippiCopy_8u_C1R(const Ipp8u* pSrc, int srcStep,
  Ipp8u* pDst, int dstStep, IppiSize roiSize)
{
  IppStatus status = ippStsNoErr;

  for (int y = 0; y < roiSize.height; y++)
  {
    const Ipp8u* src = pSrc + y*srcStep;
    Ipp8u* dst = pDst + y*dstStep;
    for (int x = 0; x < roiSize.width; x++, src++, dst++)
    {
      *dst = *src;
    }
  }

  return status;
}

/*****************************************************************************/
IppStatus ippiCopy_32f_C1R(const Ipp32f* pSrc, int srcStep,
  Ipp32f* pDst, int dstStep, IppiSize roiSize)
{
  IppStatus status = ippStsNoErr;

  const int srcStepPixels = srcStep/sizeof(Ipp32f);
  const int dstStepPixels = dstStep/sizeof(Ipp32f);
  for (int y = 0; y < roiSize.height; y++)
  {
    const Ipp32f* src = pSrc + y*srcStepPixels;
    Ipp32f* dst = pDst + y*dstStepPixels;
    for (int x = 0; x < roiSize.width; x++, src++, dst++)
    {
      *dst = *src;
    }
  }

  return status;
}

/*****************************************************************************/
Ipp8u* ippiMalloc_8u_C1(int widthPixels, int heightPixels, int* pStepBytes)
{
  const int numPixels = widthPixels*heightPixels;
  *pStepBytes = widthPixels*sizeof(Ipp8u);

  Ipp8u* data = (Ipp8u*)malloc(numPixels*sizeof(Ipp8u));
  return data;
}

/*****************************************************************************/
Ipp32f* ippiMalloc_32f_C1(int widthPixels, int heightPixels, int* pStepBytes) 
{
  const int numPixels = widthPixels*heightPixels;
  *pStepBytes = widthPixels*sizeof(Ipp32f);

  Ipp32f* data = (Ipp32f*)malloc(numPixels*sizeof(Ipp32f));
  return data;
}

/*****************************************************************************/
void ippiFree(void* ptr)
{
  free(ptr);
  ptr = NULL;
}

/*****************************************************************************/
IppStatus ippiSet_8u_C1R(Ipp8u value, Ipp8u* pDst, int dstStep, IppiSize roiSize)
{
  IppStatus status = ippStsNoErr;

  for (int y = 0; y < roiSize.height; y++)
  {
    Ipp8u* dst = pDst + y*dstStep;
    for (int x = 0; x < roiSize.width; x++, dst++)
    {
      *dst = value;
    }
  }

  return status;
}

/*****************************************************************************/
IppStatus ippiSet_32f_C1R(Ipp32f value, Ipp32f* pDst, int dstStep, IppiSize roiSize)
{
  IppStatus status = ippStsNoErr;

  const int dstStepPixels = dstStep/sizeof(Ipp32f);
  for (int y = 0; y < roiSize.height; y++)
  {
    Ipp32f* dst = pDst + y*dstStepPixels;
    for (int x = 0; x < roiSize.width; x++, dst++)
    {
      *dst = value;
    }
  }

  return status;
}

/*****************************************************************************/
IppStatus ippiAdd_8u_C1RSfs(const Ipp8u* pSrc1, int src1Step, const Ipp8u* pSrc2,
  int src2Step, Ipp8u* pDst, int dstStep, IppiSize roiSize,
  int scaleFactor)
{
  IppStatus status = ippStsNoErr;

  for (int y = 0; y < roiSize.height; y++)
  {
    const Ipp8u* src1 = pSrc1 + y*src1Step;
    const Ipp8u* src2 = pSrc2 + y*src2Step;
    Ipp8u* dst = pDst + y*dstStep;
    for (int x = 0; x < roiSize.width; x++, src1++, src2++, dst++)
    {
      if (scaleFactor > 0)       // * 2^(-scaleFactor) = (>> scaleFactor)
      {
        *dst = static_cast<Ipp8u>( (static_cast<int>(*src1)+static_cast<int>(*src2)) >> scaleFactor );
      }
      else if (scaleFactor < 0)  // * 2(+scaleFactor) = (<< scaleFactor)
      {
        *dst = static_cast<Ipp8u>( (static_cast<int>(*src1)+static_cast<int>(*src2)) << scaleFactor );
      }
      else                       // no scaling
      {
        *dst = *src1 + *src2;
      }
    }
  }

  return status;
}

/*****************************************************************************/
IppStatus ippiAdd_32f_C1R(const Ipp32f* pSrc1, int src1Step, const Ipp32f* pSrc2,
  int src2Step, Ipp32f* pDst, int dstStep, IppiSize roiSize)
{
  IppStatus status = ippStsNoErr;

  const int src1StepPixels = src1Step/sizeof(Ipp32f);
  const int src2StepPixels = src2Step/sizeof(Ipp32f);
  const int dstStepPixels = dstStep/sizeof(Ipp32f);
  for (int y = 0; y < roiSize.height; y++)
  {
    const Ipp32f* src1 = pSrc1 + y*src1StepPixels;
    const Ipp32f* src2 = pSrc2 + y*src2StepPixels;
    Ipp32f* dst = pDst + y*dstStepPixels;
    for (int x = 0; x < roiSize.width; x++, src1++, src2++, dst++)
    {
      *dst = *src1 + *src2;
    }
  }

  return status;
}

/*****************************************************************************/
IppStatus ippiAddC_8u_C1RSfs(const Ipp8u* pSrc, int srcStep, Ipp8u value, Ipp8u* pDst,
  int dstStep, IppiSize roiSize, int scaleFactor)
{
  IppStatus status = ippStsNoErr;

  for (int y = 0; y < roiSize.height; y++)
  {
    const Ipp8u* src = pSrc + y*srcStep;
    Ipp8u* dst = pDst + y*dstStep;
    for (int x = 0; x < roiSize.width; x++, src++, dst++)
    {
      if (scaleFactor > 0)       // * 2^(-scaleFactor) = (>> scaleFactor)
      {
        *dst = static_cast<Ipp8u>( (static_cast<int>(*src)+static_cast<int>(value)) >> scaleFactor );
      }
      else if (scaleFactor < 0)  // * 2(+scaleFactor) = (<< scaleFactor)
      {
        *dst = static_cast<Ipp8u>( (static_cast<int>(*src)+static_cast<int>(value)) << scaleFactor );
      }
      else                       // no scaling
      {
        *dst = *src + value;
      }
    }
  }

  return status;
}

/*****************************************************************************/
IppStatus ippiAddC_32f_C1R(const Ipp32f* pSrc, int srcStep, Ipp32f value, Ipp32f* pDst,
  int dstStep, IppiSize roiSize)
{
  IppStatus status = ippStsNoErr;

  const int srcStepPixels = srcStep/sizeof(Ipp32f);
  const int dstStepPixels = dstStep/sizeof(Ipp32f);
  for (int y = 0; y < roiSize.height; y++)
  {
    const Ipp32f* src = pSrc + y*srcStepPixels;
    Ipp32f* dst = pDst + y*dstStepPixels;
    for (int x = 0; x < roiSize.width; x++, src++, dst++)
    {
      *dst = *src + value;
    }
  }

  return status;
}

/*****************************************************************************/
IppStatus ippiMulC_8u_C1RSfs(const Ipp8u* pSrc, int srcStep, Ipp8u value, Ipp8u* pDst,
  int dstStep, IppiSize roiSize, int scaleFactor)
{
  IppStatus status = ippStsNoErr;

  for (int y = 0; y < roiSize.height; y++)
  {
    const Ipp8u* src = pSrc + y*srcStep;
    Ipp8u* dst = pDst + y*dstStep;
    for (int x = 0; x < roiSize.width; x++, src++, dst++)
    {
      if (scaleFactor > 0)       // * 2^(-scaleFactor) = (>> scaleFactor)
      {
        *dst = static_cast<Ipp8u>( (static_cast<int>(*src)*static_cast<int>(value)) >> scaleFactor );
      }
      else if (scaleFactor < 0)  // * 2(+scaleFactor) = (<< scaleFactor)
      {
        *dst = static_cast<Ipp8u>( (static_cast<int>(*src)*static_cast<int>(value)) << scaleFactor );
      }
      else                       // no scaling
      {
        *dst = *src * value;
      }
    }
  }

  return status;
}

/*****************************************************************************/
IppStatus ippiMulC_32f_C1R(const Ipp32f* pSrc, int srcStep, Ipp32f value, Ipp32f* pDst,
  int dstStep, IppiSize roiSize)
{
  IppStatus status = ippStsNoErr;

  const int srcStepPixels = srcStep/sizeof(Ipp32f);
  const int dstStepPixels = dstStep/sizeof(Ipp32f);
  for (int y = 0; y < roiSize.height; y++)
  {
    const Ipp32f* src = pSrc + y*srcStepPixels;
    Ipp32f* dst = pDst + y*dstStepPixels;
    for (int x = 0; x < roiSize.width; x++, src++, dst++)
    {
      *dst = *src * value;
    }
  }

  return status;
}

/*****************************************************************************/
IppStatus ippiMul_8u_C1RSfs(const Ipp8u* pSrc1, int src1Step, const Ipp8u* pSrc2,
  int src2Step, Ipp8u* pDst, int dstStep, IppiSize roiSize,
  int scaleFactor)
{
  IppStatus status = ippStsNoErr;

  for (int y = 0; y < roiSize.height; y++)
  {
    const Ipp8u* src1 = pSrc1 + y*src1Step;
    const Ipp8u* src2 = pSrc2 + y*src2Step;
    Ipp8u* dst = pDst + y*dstStep;
    for (int x = 0; x < roiSize.width; x++, src1++, src2++, dst++)
    {
      if (scaleFactor > 0)       // * 2^(-scaleFactor) = (>> scaleFactor)
      {
        *dst = static_cast<Ipp8u>( (static_cast<int>(*src1)*static_cast<int>(*src2)) >> scaleFactor );
      }
      else if (scaleFactor < 0)  // * 2(+scaleFactor) = (<< scaleFactor)
      {
        *dst = static_cast<Ipp8u>( (static_cast<int>(*src1)*static_cast<int>(*src2)) << scaleFactor );
      }
      else                       // no scaling
      {
        *dst = *src1 * (*src2);
      }
    }
  }

  return status;
}

/*****************************************************************************/
IppStatus ippiMul_32f_C1R(const Ipp32f* pSrc1, int src1Step, const Ipp32f* pSrc2,
  int src2Step, Ipp32f* pDst, int dstStep, IppiSize roiSize)
{
  IppStatus status = ippStsNoErr;

  const int src1StepPixels = src1Step/sizeof(Ipp32f);
  const int src2StepPixels = src2Step/sizeof(Ipp32f);
  const int dstStepPixels = dstStep/sizeof(Ipp32f);
  for (int y = 0; y < roiSize.height; y++)
  {
    const Ipp32f* src1 = pSrc1 + y*src1StepPixels;
    const Ipp32f* src2 = pSrc2 + y*src2StepPixels;
    Ipp32f* dst = pDst + y*dstStepPixels;
    for (int x = 0; x < roiSize.width; x++, src1++, src2++, dst++)
    {
      *dst = *src1 * (*src2);
    }
  }

  return status;
}

/*****************************************************************************/
IppStatus ippiSqr_8u_C1RSfs(const Ipp8u* pSrc, int srcStep,
  Ipp8u* pDst, int dstStep, IppiSize roiSize, int scaleFactor)
{
  IppStatus status = ippStsNoErr;

  for (int y = 0; y < roiSize.height; y++)
  {
    const Ipp8u* src = pSrc + y*srcStep;
    Ipp8u* dst = pDst + y*dstStep;
    for (int x = 0; x < roiSize.width; x++, src++, dst++)
    {
      if (scaleFactor > 0)       // * 2^(-scaleFactor) = (>> scaleFactor)
      {
        *dst = static_cast<Ipp8u>( (static_cast<int>(*src)*static_cast<int>(*src)) >> scaleFactor );
      }
      else if (scaleFactor < 0)  // * 2(+scaleFactor) = (<< scaleFactor)
      {
        *dst = static_cast<Ipp8u>( (static_cast<int>(*src)*static_cast<int>(*src)) << scaleFactor );
      }
      else                       // no scaling
      {
        *dst = *src*(*src);
      }
    }
  }

  return status;
}

/*****************************************************************************/
IppStatus ippiSqr_32f_C1R(const Ipp32f* pSrc, int srcStep,
  Ipp32f* pDst, int dstStep, IppiSize roiSize)
{
  IppStatus status = ippStsNoErr;

  const int srcStepPixels = srcStep/sizeof(Ipp32f);
  const int dstStepPixels = dstStep/sizeof(Ipp32f);
  for (int y = 0; y < roiSize.height; y++)
  {
    const Ipp32f* src = pSrc + y*srcStepPixels;
    Ipp32f* dst = pDst + y*dstStepPixels;
    for (int x = 0; x < roiSize.width; x++, src++, dst++)
    {
      *dst = *src*(*src);
    }
  }

  return status;
}

/*****************************************************************************/
IppStatus ippiExp_8u_C1RSfs(const Ipp8u* pSrc, int srcStep, Ipp8u* pDst,
  int dstStep, IppiSize roiSize, int scaleFactor)
{
  IppStatus status = ippStsNoErr;

  for (int y = 0; y < roiSize.height; y++)
  {
    const Ipp8u* src = pSrc + y*srcStep;
    Ipp8u* dst = pDst + y*dstStep;
    for (int x = 0; x < roiSize.width; x++, src++, dst++)
    {
      if (scaleFactor > 0)       // * 2^(-scaleFactor) = (>> scaleFactor)
      {
        *dst = static_cast<Ipp8u>( static_cast<int>(exp(static_cast<float>(*src))) >> scaleFactor );
      }
      else if (scaleFactor < 0)  // * 2(+scaleFactor) = (<< scaleFactor)
      {
        *dst = static_cast<Ipp8u>( static_cast<int>(exp(static_cast<float>(*src))) << scaleFactor );
      }
      else                       // no scaling
      {
        *dst = static_cast<Ipp8u>(exp(static_cast<float>(*src)));
      }
    }
  }

  return status;
}

/*****************************************************************************/
IppStatus ippiExp_32f_C1R(const Ipp32f* pSrc, int srcStep, Ipp32f* pDst,
  int dstStep, IppiSize roiSize)
{
  IppStatus status = ippStsNoErr;

  const int srcStepPixels = srcStep/sizeof(Ipp32f);
  const int dstStepPixels = dstStep/sizeof(Ipp32f);
  for (int y = 0; y < roiSize.height; y++)
  {
    const Ipp32f* src = pSrc + y*srcStepPixels;
    Ipp32f* dst = pDst + y*dstStepPixels;
    for (int x = 0; x < roiSize.width; x++, src++, dst++)
    {
      *dst = exp(*src);
    }
  }

  return status;
}

/*****************************************************************************/
IppStatus ippiTranspose_8u_C1R( const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
  IppiSize roiSize)
{
  IppStatus status = ippStsNoErr;

  for (int y = 0; y < roiSize.width; y++)
  {
    Ipp8u* dst = pDst + y*dstStep;
    const Ipp8u* src = pSrc + y;
    for (int x = 0; x < roiSize.height; x++, dst++, src += srcStep)
    {
      *dst = *src;
    }
  }

  return status;
}

/*****************************************************************************/
IppStatus ippiMax_8u_C1R(const Ipp8u* pSrc, int srcStep, IppiSize roiSize, Ipp8u* pMax)
{
  IppStatus status = ippStsNoErr;

  *pMax = 0;
  for (int y = 0; y < roiSize.height; y++)
  {
    const Ipp8u* src = pSrc + y*srcStep;
    for (int x = 0; x < roiSize.width; x++, src++)
    {
      if (*src > *pMax)
      {
        *pMax = *src;
      }
    }
  }

  return status;
}

/*****************************************************************************/
IppStatus ippiMax_32f_C1R(const Ipp32f* pSrc, int srcStep, IppiSize roiSize, Ipp32f* pMax)
{
  IppStatus status = ippStsNoErr;

  const int srcStepPixels = srcStep/sizeof(Ipp32f);
  *pMax = -std::numeric_limits<float>::max();
  for (int y = 0; y < roiSize.height; y++)
  {
    const Ipp32f* src = pSrc + y*srcStepPixels;
    for (int x = 0; x < roiSize.width; x++, src++)
    {
      if (*src > *pMax)
      {
        *pMax = *src;
      }
    }
  }

  return status;
}

/*****************************************************************************/
IppStatus ippiMin_8u_C1R(const Ipp8u* pSrc, int srcStep, IppiSize roiSize, Ipp8u* pMin)
{
  IppStatus status = ippStsNoErr;

  *pMin = 255;
  for (int y = 0; y < roiSize.height; y++)
  {
    const Ipp8u* src = pSrc + y*srcStep;
    for (int x = 0; x < roiSize.width; x++, src++)
    {
      if (*src < *pMin)
      {
        *pMin = *src;
      }
    }
  }

  return status;
}

/*****************************************************************************/
IppStatus ippiMin_32f_C1R(const Ipp32f* pSrc, int srcStep, IppiSize roiSize, Ipp32f* pMin)
{
  IppStatus status = ippStsNoErr;

  const int srcStepPixels = srcStep/sizeof(Ipp32f);
  *pMin = std::numeric_limits<float>::max();
  for (int y = 0; y < roiSize.height; y++)
  {
    const Ipp32f* src = pSrc + y*srcStepPixels;
    for (int x = 0; x < roiSize.width; x++, src++)
    {
      if (*src < *pMin)
      {
        *pMin = *src;
      }
    }
  }

  return status;
}

/*****************************************************************************/
IppStatus ippiMaxIndx_8u_C1R(const Ipp8u* pSrc, int srcStep, IppiSize roiSize, Ipp8u* pMax, 
  int* pIndexX, int* pIndexY)
{
  IppStatus status = ippStsNoErr;

  *pMax = 0;
  for (int y = 0; y < roiSize.height; y++)
  {
    const Ipp8u* src = pSrc + y*srcStep;
    for (int x = 0; x < roiSize.width; x++, src++)
    {
      if (*src > *pMax)
      {
        *pMax = *src;
        *pIndexX = x;
        *pIndexY = y;
      }
    }
  }

  return status;
}

/*****************************************************************************/
IppStatus ippiMaxIndx_32f_C1R(const Ipp32f* pSrc, int srcStep, IppiSize roiSize, Ipp32f* pMax, 
  int* pIndexX, int* pIndexY)
{
  IppStatus status = ippStsNoErr;

  const int srcStepPixels = srcStep/sizeof(Ipp32f);
  *pMax = -std::numeric_limits<float>::max();
  for (int y = 0; y < roiSize.height; y++)
  {
    const Ipp32f* src = pSrc + y*srcStepPixels;
    for (int x = 0; x < roiSize.width; x++, src++)
    {
      if (*src > *pMax)
      {
        *pMax = *src;
        *pIndexX = x;
        *pIndexY = y;
      }
    }
  }

  return status;
}

/*****************************************************************************/
IppStatus ippiMinIndx_8u_C1R(const Ipp8u* pSrc, int srcStep, IppiSize roiSize, Ipp8u* pMin, 
  int* pIndexX, int* pIndexY)
{
  IppStatus status = ippStsNoErr;

  *pMin = 255;
  for (int y = 0; y < roiSize.height; y++)
  {
    const Ipp8u* src = pSrc + y*srcStep;
    for (int x = 0; x < roiSize.width; x++, src++)
    {
      if (*src < *pMin)
      {
        *pMin = *src;
        *pIndexX = x;
        *pIndexY = y;
      }
    }

  }
  return status;
}

/*****************************************************************************/
IppStatus ippiMinIndx_32f_C1R(const Ipp32f* pSrc, int srcStep, IppiSize roiSize, Ipp32f* pMin,
  int* pIndexX, int* pIndexY)
{
  IppStatus status = ippStsNoErr;

  const int srcStepPixels = srcStep/sizeof(Ipp32f);
  *pMin = std::numeric_limits<float>::max();
  for (int y = 0; y < roiSize.height; y++)
  {
    const Ipp32f* src = pSrc + y*srcStepPixels;
    for (int x = 0; x < roiSize.width; x++, src++)
    {
      if (*src < *pMin)
      {
        *pMin = *src;
        *pIndexX = x;
        *pIndexY = y;
      }
    }
  }

  return status;
}

/*****************************************************************************/
IppStatus ippiMean_8u_C1R(const Ipp8u* pSrc, int srcStep, IppiSize roiSize, Ipp64f* pMean)
{
  IppStatus status = ippStsNoErr;

  *pMean = 0.0;
  for (int y = 0; y < roiSize.height; y++)
  {
    const Ipp8u* src = pSrc + y*srcStep;
    for (int x = 0; x < roiSize.width; x++, src++)
    {
      *pMean += static_cast<Ipp64f>(*src);
    }
  }
  *pMean /= static_cast<Ipp64f>(roiSize.height*roiSize.width);

  return status;
}

/*****************************************************************************/
IppStatus ippiMean_32f_C1R(const Ipp32f* pSrc, int srcStep,
  IppiSize roiSize, Ipp64f* pMean, IppHintAlgorithm hint)
{
  IppStatus status = ippStsNoErr;

  const int srcStepPixels = srcStep/sizeof(Ipp32f);
  *pMean = 0.0;
  for (int y = 0; y < roiSize.height; y++)
  {
    const Ipp32f* src = pSrc + y*srcStepPixels;
    for (int x = 0; x < roiSize.width; x++, src++)
    {
      *pMean += static_cast<Ipp64f>(*src);
    }
  }
  *pMean /= static_cast<Ipp64f>(roiSize.height*roiSize.width);

  return status;
}

/*****************************************************************************/
IppStatus ippiMean_StdDev_8u_C1R(const Ipp8u* pSrc, int srcStep, IppiSize roiSize,
  Ipp64f* pMean, Ipp64f* pStdDev)
{
  IppStatus status = ippStsNoErr;

  *pMean = 0.0;
  Ipp64f sumSq = 0.0, val;
  for (int y = 0; y < roiSize.height; y++)
  {
    const Ipp8u* src = pSrc + y*srcStep;
    for (int x = 0; x < roiSize.width; x++, src++)
    {
      val = static_cast<Ipp64f>(*src);
      *pMean += val;
      sumSq += (val*val);
    }
  }
  const Ipp64f N = static_cast<Ipp64f>(roiSize.height*roiSize.width); 
  *pMean /= N;
  *pStdDev = sqrt((sumSq/N) - (*pMean)*(*pMean));

  return status;
}

/*****************************************************************************/
IppStatus ippiMean_StdDev_32f_C1R(const Ipp32f* pSrc, int srcStep, IppiSize roiSize,
  Ipp64f* pMean, Ipp64f* pStdDev)
{
  IppStatus status = ippStsNoErr;

  const int srcStepPixels = srcStep/sizeof(Ipp32f);
  *pMean = 0.0;
  Ipp64f sumSq = 0.0, val;
  for (int y = 0; y < roiSize.height; y++)
  {
    const Ipp32f* src = pSrc + y*srcStepPixels;
    for (int x = 0; x < roiSize.width; x++, src++)
    {
      val = static_cast<Ipp64f>(*src);
      *pMean += val;
      sumSq += (val*val);
    }
  }
  const Ipp64f N = static_cast<Ipp64f>(roiSize.height*roiSize.width); 
  *pMean /= N;
  *pStdDev = sqrt((sumSq/N) - (*pMean)*(*pMean));

  return status;
}

/*****************************************************************************/
IppStatus ippiSum_8u_C1R(const Ipp8u* pSrc, int srcStep, IppiSize roiSize, Ipp64f* pSum)
{
  IppStatus status = ippStsNoErr;

  *pSum = 0.0;
  for (int y = 0; y < roiSize.height; y++)
  {
    const Ipp8u* src = pSrc + y*srcStep;
    for (int x = 0; x < roiSize.width; x++, src++)
    {
      *pSum += static_cast<Ipp64f>(*src);
    }
  }

  return status;
}

/*****************************************************************************/
IppStatus ippiSum_32f_C1R(const Ipp32f* pSrc, int srcStep,
  IppiSize roiSize, Ipp64f* pSum, IppHintAlgorithm hint)
{
  IppStatus status = ippStsNoErr;

  const int srcStepPixels = srcStep/sizeof(Ipp32f);
  *pSum = 0.0;
  for (int y = 0; y < roiSize.height; y++)
  {
    const Ipp32f* src = pSrc + y*srcStepPixels;
    for (int x = 0; x < roiSize.width; x++, src++)
    {
      *pSum += static_cast<Ipp64f>(*src);
    }
  }

  return status;
}

/*****************************************************************************/
IppStatus ippiIntegral_8u32f_C1R(const Ipp8u* pSrc, int srcStep,
  Ipp32f* pDst, int dstStep, IppiSize roiSize, Ipp32f val)
{
  IppStatus status = ippStsNoErr;

  IplImage* src = cvCreateImage(cvSize(roiSize.width,roiSize.height),IPL_DEPTH_8U,1);
  memcpy(src->imageData, pSrc, src->imageSize);

  IplImage* dst = cvCreateImage(cvSize(roiSize.width+1,roiSize.height+1),IPL_DEPTH_32F,1);
  cvIntegral(src, dst);
  memcpy(pDst, dst->imageData, dst->imageSize);

  cvReleaseImage(&src);
  cvReleaseImage(&dst);

  return status;
}

/*****************************************************************************/
IppStatus ippiGetAffineTransform(IppiRect srcRoi, const double quad[4][2], double coeffs[2][3])
{
  IppStatus status = ippStsNoErr;
  //
  // Not used?
  //
  return status;
}

/*****************************************************************************/
IppStatus ippiWarpAffine_8u_C1R(const Ipp8u* pSrc, IppiSize srcSize, int srcStep, IppiRect srcRoi, 
  Ipp8u* pDst, int dstStep, IppiRect dstRoi, const double coeffs[2][3], 
  int interpolation)
{
  IppStatus status = ippStsNoErr;
  //
  // Not used?
  //
  return status;
}

/*****************************************************************************/
IppStatus ippiFilterRow_8u_C1R(const Ipp8u* pSrc, int srcStep,
  Ipp8u* pDst, int dstStep, IppiSize dstRoiSize, const Ipp32s* pKernel,
  int kernelSize, int xAnchor, int divisor)
{
  IppStatus status = ippStsNoErr;
  //
  // Not used?
  //
  return status;
}

/*****************************************************************************/
IppStatus ippiFilterColumn_8u_C1R(const Ipp8u* pSrc,
  int srcStep, Ipp8u* pDst, int dstStep, IppiSize dstRoiSize,
  const Ipp32s* pKernel, int kernelSize, int yAnchor, int divisor)
{
  IppStatus status = ippStsNoErr;
  //
  // Not used?
  //
  return status;
}

/*****************************************************************************/
IppStatus ippiResize_8u_C1R(const Ipp8u* pSrc, IppiSize srcSize, int srcStep, IppiRect srcRoi,
  Ipp8u* pDst, int dstStep, IppiSize dstRoiSize,
  double xFactor, double yFactor, int interpolation)
{
  IppStatus status = ippStsNoErr;

  //
  // Not used?
  //

  IplImage* src = cvCreateImage(cvSize(srcSize.width,srcSize.height),IPL_DEPTH_8U,1);
  memcpy(src->imageData, pSrc, src->imageSize);

  CvSize dstSize = cvSize( cvRound(xFactor*static_cast<double>(srcSize.width)), cvRound(yFactor*static_cast<double>(dstRoiSize.height)) );
  IplImage* dst = cvCreateImage(dstSize, IPL_DEPTH_8U, 1);
  cvResize(src, dst);
  memcpy(pDst, dst->imageData, dst->imageSize);

  cvReleaseImage(&src);
  cvReleaseImage(&dst);

  return status;
}
#endif   // #ifndef HAVE_IPP




namespace cv
{
  namespace mil
  {
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // random functions

    void								randinitalize( const int init )
    {
      rng_state = cvRNG(init);
    }

    int									randint( const int min, const int max )
    {
      return cvRandInt( &rng_state )%(max-min+1) + min;
    }

    float								randfloat( )
    {
      return (float)cvRandReal( &rng_state );
    }

    vectori								randintvec( const int min, const int max, const uint num )
    {
      vectori v(num);
      for( uint k=0; k<num; k++ ) v[k] = randint(min,max);
      return v;
    }
    vectorf								randfloatvec( const uint num )
    {
      vectorf v(num);
      for( uint k=0; k<num; k++ ) v[k] = randfloat();
      return v;
    }
    float								randgaus(const float mean, const float sigma)
    {
      double x, y, r2;

      do{
        x = -1 + 2 * randfloat();
        y = -1 + 2 * randfloat();
        r2 = x * x + y * y;
      }
      while (r2 > 1.0 || r2 == 0);

      return (float) (sigma * y * sqrt (-2.0 * log (r2) / r2)) + mean;
    }

    vectorf								randgausvec(const float mean, const float sigma, const int num)
    {
      vectorf v(num);
      for( int k=0; k<num; k++ ) v[k] = randgaus(mean,sigma);
      return v;
    }

    vectori								sampleDisc(const vectorf &weights, const uint num)
    {
      vectori inds(num,0);
      int maxind = (int)weights.size()-1;

      // normalize weights
      vectorf nw(weights.size());

      nw[0] = weights[0];
      for( uint k=1; k<weights.size(); k++ )
        nw[k] = nw[k-1]+weights[k];

      // get uniform random numbers
      static vectorf r;
      r = randfloatvec(num);

      for( int k=0; k<(int)num; k++ )
        for( uint j=0; j<weights.size(); j++ ){
          if( r[k] > nw[j] && inds[k]<maxind) inds[k]++;
          else break;
        }

        return inds;

    }

    string								int2str( int i, int ndigits )
    {
      ostringstream temp;
      temp << setfill('0') << setw(ndigits) << i;
      return temp.str();
    }


    template<> float				Matrixu::ii ( const int row, const int col, const int depth ) const
    {
      return (float) ((float*)_iidata[depth])[row*_iipixStep + col];
    }
    template<> float				Matrixu::dii_dx(uint x, uint y, uint channel)
    {
      if( !isInitII() ) abortError(__LINE__,__FILE__,"cannot take dii/dx, ii is not init");

      if( (x+1) > (uint)cols() || x < 1 ) return 0.0f;
      //0.5*(GET3(ii,y,(x+1),bin,rows,cols) - GET3(ii,y,(x-1),bin,rows,cols));

      return 0.5f * ( ii(y,(x+1),channel) - ii(y,(x-1),channel) );
    }

    template<> float				Matrixu::dii_dy(uint x, uint y, uint channel)
    {
      if( !isInitII() ) abortError(__LINE__,__FILE__,"cannot take dii/dx, ii is not init");

      if( (y+1) > (uint)rows() || y < 1 ) return 0.0f;
      //0.5*(GET3(ii,y,(x+1),bin,rows,cols) - GET3(ii,y,(x-1),bin,rows,cols));

      return 0.5f * ( ii((y+1),x,channel) - ii((y-1),x,channel) );
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<> void					Matrixu::initII() 
    {
      bool err=false;
      _iidata.resize(_depth);
      for( uint k=0; k<_data.size(); k++ ){
        if( _iidata[k] == NULL )
          _iidata[k] = ippiMalloc_32f_C1(_cols+1,_rows+1,&(_iidataStep));
        if( _iidata[k] == NULL ) abortError(__LINE__,__FILE__,"OUT OF MEMORY!");
        _iipixStep = _iidataStep/sizeof(float);
        IppStatus is = ippiIntegral_8u32f_C1R((unsigned char*)_data[k], _dataStep, (float*)_iidata[k], _iidataStep, _roi, 0);
        assert( is == ippStsNoErr );
        err = err || _data[k] == NULL;
      }
      _ii_init = true;
    }

    template<> float				Matrixu::sumRect(const IppiRect &rect, int channel) const
    {
      // debug checks
      assert(_ii_init);
      assert( rect.x >= 0 && rect.y >= 0 && (rect.y+rect.height) <= _rows 
        && (rect.x+rect.width) <= _cols && channel < _depth);
      int maxy = (rect.y+rect.height)*_iipixStep;
      int maxx = rect.x+rect.width;
      int y = rect.y*_iipixStep;

      float tl = ((float*)_iidata[channel])[y + rect.x];
      float tr = ((float*)_iidata[channel])[y + maxx];
      float br = ((float*)_iidata[channel])[maxy + maxx];
      float bl = ((float*)_iidata[channel])[maxy + rect.x];

      return br + tl - tr - bl;
      //return ii(maxy,maxx,channel) + ii(rect.y,rect.x,channel) 
      //	- ii(rect.y,maxx,channel) - ii(maxy,rect.x,channel);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    template<> void					Matrixu::LoadImage(const char *filename, bool color)
    {
      IplImage *img;
      img = cvLoadImage(filename,(int)color);
      if( img == NULL )
        abortError(__LINE__,__FILE__,"Error loading file");
      Resize(img->height, img->width, img->nChannels);

      if( color ) 
        IplImage2Matrix(img);
      else{
        for( int row=0; row<_rows; row++ )
          for( int k=0; k<_cols; k++ ){
            ((unsigned char*)_data[0])[row*_dataStep+k] = img->imageData[row*img->widthStep+k];
          }

      }
      cvReleaseImage(&img);
    }

    template<> void					Matrixu::SaveImage(const char *filename)
    {
      createIpl();
      int success = cvSaveImage( filename, _iplimg );
      freeIpl();
    }

    template<> bool					Matrixu::CaptureImage(CvCapture* capture, Matrixu &res, int color)
    {
      IplImage *img;
      if( capture == NULL ) return false;
      img = cvQueryFrame( capture );
      if( img == NULL ) return false;
      res.Resize(img->height, img->width, 1+2*color);
      if( color ){
        //res.Resize(img->height, img->width, 3);
        res.IplImage2Matrix(img);
      }
      else{
        static IplImage *img2;
        if( img2 == NULL ) img2 = cvCreateImage( cvSize(res._cols, res._rows), IPL_DEPTH_8U, 1 ); 
        cvCvtColor( img, img2, CV_RGB2GRAY );
        img2->origin = 0;
        res.GrayIplImage2Matrix(img2);
      }
      img = NULL;
      return true;
    }
    template<> bool					Matrixu::WriteFrame(CvVideoWriter* w, Matrixu &img)
    {
      img.createIpl();
      if( w != NULL ){ 
        IplImage* iplimg = img.getIpl();
        iplimg->origin = 1;
        cvWriteFrame( w, iplimg );
        return true;
      }else
        return false;
    }
    template<> void					Matrixu::PlayCam(int color, const char* fname)
    {
      CvCapture* capture = cvCreateCameraCapture( 0 );
      if( capture==NULL ) abortError(__LINE__,__FILE__,"camera not found!");
      CvVideoWriter* w = NULL;


      Matrixu frame;
      frame._keepIpl = false;
      cout << "Press q to quit" << endl;

      StopWatch sw(true);
      double ttime=0.0;
      for( int cnt=0; true; cnt++ )
      {
        CaptureImage(capture,frame,color);

        // initialize video output 
        if( fname != NULL && w == NULL)
          w = cvCreateVideoWriter( fname, CV_FOURCC('I','Y', 'U', 'V'), 10, cvSize(frame.cols(), frame.rows()), 3 );

        // output (both screen and possibly to file)
        frame._keepIpl=true;
        frame.display(1); char q = cvWaitKey(1);
        WriteFrame(w, frame);
        frame._keepIpl=false; frame.freeIpl();

        // check key input
        if( q=='q' ) break;

        // timing
        ttime = sw.Elapsed(true);
        fprintf(stderr,"%s%d Frames/%f sec = %f FPS",ERASELINE,cnt,ttime,((double)cnt)/ttime);
      }

      cvReleaseCapture( &capture );
      if( w != NULL )
        cvReleaseVideoWriter( &w );

    }

    template<> void					Matrixu::PlayCamOpenCV()
    {
      CvCapture* capture = cvCaptureFromCAM( -1 );
      if( capture == NULL )
        abortError(__LINE__,__FILE__,"Error finding cam");

      cout << "Press q to quit" << endl;
      IplImage *img;
      cvNamedWindow( "Cam", 0/*CV_WINDOW_AUTOSIZE*/ );

      StopWatch sw(true);
      double ttime=0.0;
      for( int cnt=0; true; cnt++ )
      {
        img = cvQueryFrame( capture );
        cvShowImage( "Cam", img );
        ttime = sw.Elapsed(true);
        fprintf(stderr,"%s%d Frames/%f sec = %f FPS",ERASELINE,cnt,ttime,((double)cnt)/ttime);
        char q = cvWaitKey(1);
        if( q=='q' ) break;
      }

      cvReleaseCapture( &capture );
      cout << endl << "Ending PlayCam" << endl;
    }

    template<> void					Matrixu::createIpl(bool force)
    {
      if( _iplimg != NULL && !force) return;
      if( _iplimg != NULL ) cvReleaseImage(&_iplimg);
      CvSize sz; sz.width = _cols; sz.height = _rows;

      int depth = 3;
      _iplimg = cvCreateImageHeader( sz, IPL_DEPTH_8U, depth );

      //_iplimg->align = 32;
      //_iplimg->widthStep = (((_iplimg->width * _iplimg->nChannels *
      //     (_iplimg->depth & ~IPL_DEPTH_SIGN) + 7)/8)+ _iplimg->align - 1) & (~(_iplimg->align - 1));
      //_iplimg->widthStep = _dataStep*depth;
      //_iplimg->imageSize = _iplimg->height*_iplimg->widthStep;
      cvCreateData(_iplimg);

      //cvInitImageHeader( _iplimg, sz, IPL_DEPTH_8U, _depth, IPL_ORIGIN_TL, 16 );
      //IplImage *_iplimg = cvCreateImage( sz, IPL_DEPTH_8U, _depth );

      //assert( _depth==1 ? _iplimg->widthStep == _dataStep : _iplimg->widthStep/3 == _dataStep ); // should always be the same (multiple of 32)
      if( _depth == 1 )
        for( int row=0; row<_rows; row++ )
          for( int k=0; k<_cols*3; k+=3 ){
            _iplimg->imageData[row*_iplimg->widthStep+k+2]=((unsigned char*)_data[0])[row*_dataStep+k/3];
            _iplimg->imageData[row*_iplimg->widthStep+k+1]=((unsigned char*)_data[0])[row*_dataStep+k/3];
            _iplimg->imageData[row*_iplimg->widthStep+k  ]=((unsigned char*)_data[0])[row*_dataStep+k/3];
          }
      else
        //for( int k=0; k<_rows*_dataStep*3; k+=3 ){
        //	_iplimg->imageData[k+2] = ((unsigned char*)_data[0])[k/3]; // B
        //	_iplimg->imageData[k+1] = ((unsigned char*)_data[1])[k/3]; // G
        //	_iplimg->imageData[k  ] = ((unsigned char*)_data[2])[k/3]; // R
        //}
        for( int row=0; row<_rows; row++ )
          for( int k=0; k<_cols*3; k+=3 ){
            _iplimg->imageData[row*_iplimg->widthStep+k+2]=((unsigned char*)_data[0])[row*_dataStep+k/3];
            _iplimg->imageData[row*_iplimg->widthStep+k+1]=((unsigned char*)_data[1])[row*_dataStep+k/3];
            _iplimg->imageData[row*_iplimg->widthStep+k  ]=((unsigned char*)_data[2])[row*_dataStep+k/3];
          }

    }

    template<> void					Matrixu::freeIpl()
    {
      if( !_keepIpl && _iplimg != NULL) cvReleaseImage(&_iplimg);
    }

    template<> void					Matrixu::IplImage2Matrix(IplImage *img)
    {
      //Resize(img->height, img->width, img->nChannels);
      bool origin = img->origin==1;

      if( _depth == 1 )
        for( int row=0; row<_rows; row++ )
          for( int k=0; k<_cols*3; k+=3 )
            if( origin )
              ((unsigned char*)_data[0])[(_rows - row - 1)*_dataStep+k/3] = img->imageData[row*img->widthStep+k];
            else
              ((unsigned char*)_data[0])[row*_dataStep+k/3] = img->imageData[row*img->widthStep+k];
      else
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for( int row=0; row<_rows; row++ )
          for( int k=0; k<_cols*3; k+=3 ){
            if( origin ){
              ((unsigned char*)_data[0])[(_rows - row - 1)*_dataStep+k/3] = img->imageData[row*img->widthStep+k+2];
              ((unsigned char*)_data[1])[(_rows - row - 1)*_dataStep+k/3] = img->imageData[row*img->widthStep+k+1];
              ((unsigned char*)_data[2])[(_rows - row - 1)*_dataStep+k/3] = img->imageData[row*img->widthStep+k];
            }
            else{
              ((unsigned char*)_data[0])[row*_dataStep+k/3] = img->imageData[row*img->widthStep+k+2];
              ((unsigned char*)_data[1])[row*_dataStep+k/3] = img->imageData[row*img->widthStep+k+1];
              ((unsigned char*)_data[2])[row*_dataStep+k/3] = img->imageData[row*img->widthStep+k];
            }
          }

          if( _keepIpl )
            _iplimg = img;
    }

    template<> void					Matrixu::GrayIplImage2Matrix(IplImage *img)
    {
      //Resize(img->height, img->width, img->nChannels);
      bool origin = img->origin==1;

      if( _depth == 1 )
        for( int row=0; row<_rows; row++ )
          for( int k=0; k<_cols; k++ )
            if( origin )
              ((unsigned char*)_data[0])[(_rows - row - 1)*_dataStep+k] = img->imageData[row*img->widthStep+k];
            else
              ((unsigned char*)_data[0])[row*_dataStep+k] = img->imageData[row*img->widthStep+k];

    }

    template<> void					Matrixu::display(int fignum, float p)
    {
      assert(size() > 0);
      createIpl();
      char name[1024]; 
      sprintf(name,"Figure %d",fignum);
      cvNamedWindow( name, 0/*CV_WINDOW_AUTOSIZE*/ );
      cvShowImage( name, _iplimg );
      cvResizeWindow( name, std::max<int>(static_cast<int>(static_cast<float>(_cols)*p),200), 
        std::max<int>(static_cast<int>(static_cast<float>(_rows)*p), static_cast<int>(static_cast<float>(_rows)*(200.0f/static_cast<float>(_cols)))) );
      //cvWaitKey(0);//DEBUG
      freeIpl();
    }

    template<> void					Matrixu::drawRect(IppiRect rect, int lineWidth, int R, int G, int B )
    {
      createIpl();
      CvPoint p1, p2;
      p1 = cvPoint(rect.x, rect.y); 
      p2 = cvPoint(rect.x+rect.width,rect.y+rect.height);
      cvDrawRect(_iplimg, p1, p2, CV_RGB(R, G, B), lineWidth);
      IplImage2Matrix(_iplimg);
      freeIpl();
    }

    template<> void					Matrixu::drawRect(float width, float height, float x,float y, float sc, float th, int lineWidth, int R, int G, int B)
    {

      sc = 1.0f/sc;
      th = -th;

      double cth = cos(th)*sc;
      double sth = sin(th)*sc;

      CvPoint p1, p2, p3, p4;

      p1.x = (int)(-cth*width/2 + sth*height/2 + width/2 + x);
      p1.y = (int)(-sth*width/2 - cth*height/2 + height/2 + y);

      p2.x = (int)(cth*width/2 + sth*height/2 + width/2 + x);
      p2.y = (int)(sth*width/2 - cth*height/2 + height/2 + y);

      p3.x = (int)(cth*width/2 - sth*height/2 + width/2 + x);
      p3.y = (int)(sth*width/2 + cth*height/2 + height/2 + y);

      p4.x = (int)(-cth*width/2 - sth*height/2 + width/2 + x);
      p4.y = (int)(-sth*width/2 + cth*height/2 + height/2 + y);

      //cout << p1.x << " " << p1.y << endl;
      //cout << p2.x << " " << p2.y << endl;
      //cout << p3.x << " " << p3.y << endl;
      //cout << p4.x << " " << p4.y << endl;

      createIpl();
      cvLine( _iplimg, p1, p2, CV_RGB( R, G, B), lineWidth, CV_AA );
      cvLine( _iplimg, p2, p3, CV_RGB( R, G, B), lineWidth, CV_AA );
      cvLine( _iplimg, p3, p4, CV_RGB( R, G, B), lineWidth, CV_AA );
      cvLine( _iplimg, p4, p1, CV_RGB( R, G, B), lineWidth, CV_AA );
      IplImage2Matrix(_iplimg);
      freeIpl();
    }



    template<> void					Matrixu::drawEllipse(float height, float width, float x,float y, int lineWidth, int R, int G, int B)
    {
      createIpl();
      CvPoint p = cvPoint((int)x,(int)y);
      CvSize s = cvSize((int)width, (int)height);
      cvEllipse( _iplimg, p, s, 0, 0, 365, CV_RGB( R, G, B), lineWidth );
      IplImage2Matrix(_iplimg);
      freeIpl();
    }
    template<> void					Matrixu::drawEllipse(float height, float width, float x,float y, float startang, float endang, int lineWidth, int R, int G, int B)
    {
      createIpl();
      CvPoint p = cvPoint((int)x,(int)y);
      CvSize s = cvSize((int)width, (int)height);
      cvEllipse( _iplimg, p, s, 0, startang, endang, CV_RGB( R, G, B), lineWidth );
      IplImage2Matrix(_iplimg);
      freeIpl();
    }
    template<> void					Matrixu::drawText(const char* txt, float x, float y, int R, int G, int B)
    {
      createIpl();
      CvFont font;
      cvInitFont( &font, CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 2, 8 );
      CvPoint p = cvPoint((int)x,(int)y);
      cvPutText( _iplimg, txt, p, &font, CV_RGB( R, G, B) );
      IplImage2Matrix(_iplimg);
      freeIpl();
    }
    template<> void					Matrixu::warp(Matrixu &res,uint rows, uint cols, float x, float y, float sc, float th, float sr, float phi)
    {
      res.Resize(rows,cols,_depth);

      double coeffs[2][3];
      double quad[4][2];

      double cth = cos(th)*sc;
      double sth = sin(th)*sc;

      quad[0][0] = -cth*cols/2 + sth*rows/2 + cols/2;
      quad[0][1] = -sth*cols/2 - cth*rows/2 + rows/2;

      quad[1][0] = cth*cols/2 + sth*rows/2 + cols/2;
      quad[1][1] = sth*cols/2 - cth*rows/2 + rows/2;

      quad[2][0] = cth*cols/2 - sth*rows/2 + cols/2;
      quad[2][1] = sth*cols/2 + cth*rows/2 + rows/2;

      quad[3][0] = -cth*cols/2 - sth*rows/2 + cols/2;
      quad[3][1] = -sth*cols/2 + cth*rows/2 + rows/2;

      //cout << quad[0][0]+x << " " << quad[0][1]+y << endl;
      //cout << quad[1][0]+x << " " << quad[1][1]+y << endl;
      //cout << quad[2][0]+x << " " << quad[2][1]+y << endl;
      //cout << quad[3][0]+x << " " << quad[3][1]+y << endl << endl;

      IppiRect r;
      r.x = (int)x;
      r.y = (int)y;
      r.width = cols;
      r.height = rows;

      IppStatus ii = ippiGetAffineTransform(r, quad, coeffs);

      for( int k=0; k<_depth; k++ )
        ippiWarpAffine_8u_C1R((unsigned char*)_data[k],_roi, _dataStep, _roirect, (unsigned char*)res._data[k],res._dataStep, res._roirect, coeffs, IPPI_INTER_LINEAR);

    }

    template<> void					Matrixu::warpAll(uint rows, uint cols, vector<vectorf> params, vector<Matrixu> &res)
    {
      res.resize(params[0].size());

#ifdef _OPENMP
#pragma omp parallel for
#endif
      for( int k=0; k<(int)params[0].size(); k++ )
        warp(res[k],rows,cols,params[0][k],params[1][k],params[2][k],params[3][k]);
    }
    template<> void					Matrixu::computeGradChannels()
    {
      Ipp32s kernel[3] = {-1, 0, 1};

      IppiSize r = _roi;
      r.width-=3;
      r.height-=3;

      ippiFilterRow_8u_C1R((unsigned char*)_data[0], _dataStep, (unsigned char*)_data[_depth-2], _dataStep, r, kernel, 3, 2, -1);
      ippiFilterColumn_8u_C1R((unsigned char*)_data[0], _dataStep, (unsigned char*)_data[_depth-1], _dataStep, r, kernel, 3, 2, -1);
    }
    template<> void					Matrixu::SaveImages(std::vector<Matrixu> imgs, const char *dirname, float resize)
    {
      char fname[1024];

      for( uint k=0; k<imgs.size(); k++ ){
        sprintf(fname,"%s/img%05d.png",dirname,k);
        if( resize == 1.0f )
          imgs[k].SaveImage(fname);
        else{
          imgs[k].imResize(resize).SaveImage(fname);
        }
      }
    }
    template<> Matrixu				Matrixu::imResize(float r, float c)
    {
      float pr, pc; int nr, nc;
      if( c<0 ){
        pr = r; pc = r;
        nr = (int)(r*_rows);
        nc = (int)(r*_cols);
      }else{
        pr = r/_rows; pc = c/_cols;
        nr = (int)r;
        nc = (int)c;
      }

      Matrixu res((int)(nr), (int)(nc), _depth);
      IppStatus ippst;
      for( int k=0; k<_depth; k++ )
        ippst = ippiResize_8u_C1R((unsigned char*)_data[k],_roi, _dataStep, _roirect, (unsigned char*)res._data[k], res._dataStep, res._roi, pc, pr, IPPI_INTER_LINEAR);

      return res;
    }
    template<> void					Matrixu::conv2RGB(Matrixu &res)
    {
      res.Resize(_rows,_cols,3);
      for( int k=0; k<_rows*_dataStep; k++ )
      {
        ((unsigned char*)res._data[0])[k] = ((unsigned char*)_data[0])[k];
        ((unsigned char*)res._data[1])[k] = ((unsigned char*)_data[0])[k];
        ((unsigned char*)res._data[2])[k] = ((unsigned char*)_data[0])[k];
      }
    }
    template<> void					Matrixu::conv2BW(Matrixu &res)
    {
      res.Resize(_rows,_cols,1);
      double t;
      for( int k=0; k<(int)size(); k++ )
      {
        t = (double) ((unsigned char*)_data[0])[k]; 
        t+= (double) ((unsigned char*)_data[1])[k];
        t+= (double) ((unsigned char*)_data[2])[k];
        ((unsigned char*)res._data[0])[k] = (unsigned char) (t/3.0);
      }

      if( res._keepIpl ) res.freeIpl();
    }
    template<> float				Matrixf::Dot(const Matrixf &x)
    {
      assert( this->size() == x.size() );
      float sum = 0.0f;
#ifdef _OPENMP
#pragma omp parallel for reduction(+: sum)
#endif
      for( int i=0; i<(int)size(); i++ )
        sum += (*this)(i)*x(i);

      return sum;
    }


    Sample::Sample(Matrixu *img, int row, int col, int width, int height, float weight) 
    {
      _img	= img;
      _row	= row;
      _col	= col;
      _width	= width;
      _height	= height;
      _weight = weight;
    }



    void		SampleSet::sampleImage(Matrixu *img, int x, int y, int w, int h, float inrad, float outrad, int maxnum)
    {
      int rowsz = img->rows() - h - 1;
      int colsz = img->cols() - w - 1;
      float inradsq = inrad*inrad;
      float outradsq = outrad*outrad;
      int dist;

      uint minrow = max(0,(int)y-(int)inrad);
      uint maxrow = min((int)rowsz-1,(int)y+(int)inrad);
      uint mincol = max(0,(int)x-(int)inrad);
      uint maxcol = min((int)colsz-1,(int)x+(int)inrad);

      //fprintf(stderr,"inrad=%f minrow=%d maxrow=%d mincol=%d maxcol=%d\n",inrad,minrow,maxrow,mincol,maxcol);

      _samples.resize( (maxrow-minrow+1)*(maxcol-mincol+1) );
      int i=0;

      float prob = ((float)(maxnum))/_samples.size();

      for( int r=minrow; r<=(int)maxrow; r++ )
        for( int c=mincol; c<=(int)maxcol; c++ ){
          dist = (y-r)*(y-r) + (x-c)*(x-c);
          if( randfloat()<prob && dist < inradsq && dist >= outradsq ){
            _samples[i]._img = img;
            _samples[i]._col = c;
            _samples[i]._row = r;
            _samples[i]._height = h;
            _samples[i]._width = w;
            i++;
          }
        }

        _samples.resize(min(i,maxnum));

    }

    void		SampleSet::sampleImage(Matrixu *img, uint num, int w, int h)
    {
      int rowsz = img->rows() - h - 1;
      int colsz = img->cols() - w - 1;

      _samples.resize( num );
      for( int i=0; i<(int)num; i++ ){
        _samples[i]._img = img;
        _samples[i]._col = randint(0,colsz);
        _samples[i]._row = randint(0,rowsz);
        _samples[i]._height = h;
        _samples[i]._width = w;
      }
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    HaarFtrParams::HaarFtrParams() 
    {
      _numCh = -1;
      for( int k=0; k<1024; k++ )
        _useChannels[k] = -1;
      _minNumRect	= 2;
      _maxNumRect	= 6;
      _useChannels[0] = 0;
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    HaarFtr::HaarFtr()
    {
      _width = 0;
      _height = 0;
      _channel = 0;
    }


    void			HaarFtr::generate(FtrParams *op)
    {
      HaarFtrParams *p = (HaarFtrParams*)op;
      _width = p->_width;
      _height = p->_height;
      int numrects = randint(p->_minNumRect,p->_maxNumRect);
      _rects.resize(numrects);
      _weights.resize(numrects);
      _rsums.resize(numrects);
      _maxSum = 0.0f;

      for( int k=0; k<numrects; k++ )
      {
        _weights[k] = randfloat()*2-1;
        _rects[k].x = randint(0,(uint)(p->_width-3));
        _rects[k].y = randint(0,(uint)(p->_height-3));
        _rects[k].width = randint(1,(p->_width-_rects[k].x-2));
        _rects[k].height = randint(1 ,(p->_height-_rects[k].y-2));
        _rsums[k] = std::abs(_weights[k]*(_rects[k].width+1)*(_rects[k].height+1)*255);
        //_rects[k].width = randint(1,3);
        //_rects[k].height = randint(1,3);
      }

      if( p->_numCh < 0 ){
        p->_numCh=0;
        for( int k=0; k<1024; k++ )
          p->_numCh += p->_useChannels[k]>=0;
      }

      _channel = p->_useChannels[randint(0,p->_numCh-1)];
    }

    Matrixu			HaarFtr::toViz()
    {
      Matrixu v(_height,_width,3);
      v.Set(0);
      v._keepIpl = true;

      for( uint k=0; k<_rects.size(); k++ )
      {
        if( _weights[k] < 0 )
          v.drawRect(_rects[k],1,(int)(255*std::max<double>(-1*_weights[k],0.5)),0,0);
        else
          v.drawRect(_rects[k],1,0,(int)(255*std::max<double>(_weights[k],0.5)),(int)(255*std::max<double>(_weights[k],0.5)));
      }

      v._keepIpl = false;
      return v;
    }



    //////////////////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    void			Ftr::compute( SampleSet &samples, const vecFtr &ftrs)
    {
      int numftrs = ftrs.size();
      int numsamples = samples.size();
      if( numsamples==0 ) return;

      samples.resizeFtrs(numftrs);

#ifdef _OPENMP
#pragma omp parallel for
#endif
      for( int ftr=0; ftr<numftrs; ftr++ ){
        for( int k=0; k<numsamples; k++ ){
          samples.getFtrVal(k,ftr) = ftrs[ftr]->compute(samples[k]);
        }
      }

    }
    void			Ftr::compute( SampleSet &samples, Ftr *ftr, int ftrind )
    {

      int numsamples = samples.size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
      for( int k=0; k<numsamples; k++ ){
        samples.getFtrVal(k,ftrind) = ftr->compute(samples[k]);
      }

    }
    vecFtr			Ftr::generate( FtrParams *params, uint num )
    {
      vecFtr ftrs;

      ftrs.resize(num);
      for( uint k=0; k<num; k++ ){
        switch( params->ftrType() ){
        case 0: ftrs[k] = new HaarFtr(); break;
        }
        ftrs[k]->generate(params);
      }

      // DEBUG
      if( 0 )
        Ftr::toViz(ftrs,"ftrs");

      return ftrs;
    }

    void			Ftr::deleteFtrs( vecFtr ftrs )
    {
      for( uint k=0; k<ftrs.size(); k ++ )
        delete ftrs[k];
    }
    void			Ftr::toViz( vecFtr &ftrs, const char *dirname )
    {
      char fname[1024];
      Matrixu img;
      for( uint k=0; k<ftrs.size(); k++ ){
        sprintf(fname,"%s/ftr%05d.png",dirname,k);
        img = ftrs[k]->toViz();
        img.SaveImage(fname);
      }
    }



    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    ClfStrong*			ClfStrong::makeClf(ClfStrongParams *clfparams)
    {
      ClfStrong* clf;

      switch(clfparams->clfType()){
      case 0:
        clf = new ClfAdaBoost();
        break;
      case 1:
        clf = new ClfMilBoost();
        break;

      default:
        abortError(__LINE__,__FILE__,"Incorrect clf type!");
      }

      clf->init(clfparams);
      return clf;
    }

    Matrixf				ClfStrong::applyToImage(ClfStrong *clf, Matrixu &img, bool logR)
    {
      img.initII();
      Matrixf resp(img.rows(),img.cols());
      int height = clf->_params->_ftrParams->_height;
      int width = clf->_params->_ftrParams->_width;

      int rowsz = img.rows() - width - 1;
      int colsz = img.cols() - height - 1;

      SampleSet x; x.sampleImage(&img,0,0,width,height,100000); // sample every point
      Ftr::compute(x,clf->_ftrs);
      vectorf rf = clf->classify(x,logR);
      for( int i=0; i<x.size(); i++ )
        resp(x[i]._row,x[i]._col) = rf[i];

      return resp;
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    ClfWeak::ClfWeak()
    {
      _trained=false; _ind=-1;
    }

    ClfWeak::ClfWeak(int id)
    {
      _trained=false; _ind=id;
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    void				ClfOnlineStump::init()
    {
      _mu0	= 0;
      _mu1	= 0;
      _sig0	= 1;
      _sig1	= 1;
      _lRate	= 0.85f;
      _trained = false;
    }

    void				ClfWStump::init()
    {
      _mu0	= 0;
      _mu1	= 0;
      _sig0	= 1;
      _sig1	= 1;
      _lRate	= 0.85f;
      _trained = false;
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    void				ClfAdaBoost::init(ClfStrongParams *params)
    {
      // initialize model
      _params		= params;
      _myParams	= (ClfAdaBoostParams*)params;
      _numsamples = 0;

      if( _myParams->_numSel > _myParams->_numFeat  || _myParams->_numSel < 1 )
        _myParams->_numSel = _myParams->_numFeat/2;

      //_countFP.Resize(_params._numSel,_params._numFeat,1);
      //_countFN.Resize(_params._numSel,_params._numFeat,1);
      //_countTP.Resize(_params._numSel,_params._numFeat,1);
      //_countTN.Resize(_params._numSel,_params._numFeat,1);
      resizeVec(_countFPv,_myParams->_numSel, _myParams->_numFeat,1.0f);
      resizeVec(_countTPv,_myParams->_numSel, _myParams->_numFeat,1.0f);
      resizeVec(_countFNv,_myParams->_numSel, _myParams->_numFeat,1.0f);
      resizeVec(_countTNv,_myParams->_numSel, _myParams->_numFeat,1.0f);

      _alphas.resize(_myParams->_numSel,0);
      _ftrs = Ftr::generate(_myParams->_ftrParams,_myParams->_numFeat);
      _selectors.resize(_myParams->_numSel,0);
      _weakclf.resize(_myParams->_numFeat);
      for( int k=0; k<_myParams->_numFeat; k++ )
        //if (_params._weakLearner == string("kalman"))
        //	_weakclf[k] = new ClfKalmanStump();
        //else 
        if(_myParams->_weakLearner == string("stump")){
          _weakclf[k] = new ClfOnlineStump(k);
          _weakclf[k]->_ftr = _ftrs[k];
          _weakclf[k]->_lRate = _myParams->_lRate;
          _weakclf[k]->_parent = this;
        }
        else if( _myParams->_weakLearner == string("wstump")){
          _weakclf[k] = new ClfWStump(k);
          _weakclf[k]->_ftr = _ftrs[k];
          _weakclf[k]->_lRate = _myParams->_lRate;
          _weakclf[k]->_parent = this;
        }
        else
          abortError(__LINE__,__FILE__,"incorrect weak clf name");
    }
    void				ClfAdaBoost::update(SampleSet &posx, SampleSet &negx)
    {
      _clfsw.Start();
      int numpts = posx.size() + negx.size();

      // compute ftrs
      if( !posx.ftrsComputed() ) Ftr::compute(posx, _ftrs);
      if( !negx.ftrsComputed() ) Ftr::compute(negx, _ftrs);

      //vectorf poslam(posx[0].size(),.5f*numpts/posx[0].size()), neglam(negx[0].size(),.5f*numpts/negx[0].size());
      //vectorf poslam(posx[0].size(),1), neglam(negx[0].size(),1);
      vectorf poslam(posx.size(),.5f/posx.size()), neglam(negx.size(),.5f/negx.size());
      vector<vectorb> pospred(nFtrs()), negpred(nFtrs());
      vectorf errs(nFtrs());
      vectori order(nFtrs());

      _sumAlph=0.0f;
      _selectors.clear();

      // update all weak classifiers and get predicted labels
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for( int k=0; k<nFtrs(); k++ ){
        _weakclf[k]->update(posx,negx);
        pospred[k] = _weakclf[k]->classifySet(posx);
        negpred[k] = _weakclf[k]->classifySet(negx);
      }

      vectori worstinds;

      // loop over selectors
      for( int t=0; t<_myParams->_numSel; t++ ){
        // calculate errors for selector t
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for( int k=0; k<_myParams->_numFeat; k++ ){
          for( int j=0; j<(int)poslam.size(); j++ ){
            //if( poslam[j] > 1e-5 )
            (pospred[k][j])? _countTPv[t][k] += poslam[j] : _countFNv[t][k] += poslam[j];
          }
        }
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for( int k=0; k<_myParams->_numFeat; k++ ){
          for( int j=0; j<(int)neglam.size(); j++ ){
            //if( neglam[j] > 1e-5 )
            (!negpred[k][j])? _countTNv[t][k] += neglam[j] : _countFPv[t][k] += neglam[j];
          }
        }
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for( int k=0; k<_myParams->_numFeat; k++ ){
          //float fp,fn;
          //fp = _countFPv[t][k] / (_countFPv[t][k] + _countTNv[t][k]);
          //fn = _countFNv[t][k] / (_countFNv[t][k] + _countTPv[t][k]);
          //errs[k] = 0.3f*fp + 0.7f*fn;
          errs[k] = (_countFPv[t][k]+_countFNv[t][k])/(_countFPv[t][k]+_countFNv[t][k]+_countTPv[t][k]+_countTNv[t][k]);
        }

        // pick the best weak clf and udpate _selectors and _selectedFtrs
        float minerr;
        uint bestind;

        sort_order_des(errs,order);

        // find best in that isn't already included
        for( uint k=0; k<order.size(); k++ )
          if( count( _selectors.begin(), _selectors.end(), order[k])==0 )	{
            _selectors.push_back(order[k]);
            minerr = errs[k];
            bestind = order[k];
            break;
          }

          //cout << "min err=" << minerr << endl;

          // find worst ind
          worstinds.push_back(order[order.size()-1]);

          // update alpha
          _alphas[t] = std::max<float>(0,std::min<float>(0.5f*log((1-minerr)/(minerr+0.00001f)),10));
          _sumAlph += _alphas[t];

          // update weights
          float corw = 1/(2-2*minerr);
          float incorw = 1/(2*minerr);
#ifdef _OPENMP
#pragma omp parallel for
#endif
          for( int j=0; j<(int)poslam.size(); j++ )
            poslam[j] *= (pospred[bestind][j]==1)? corw : incorw;
#ifdef _OPENMP
#pragma omp parallel for
#endif
          for( int j=0; j<(int)neglam.size(); j++ )
            neglam[j] *= (negpred[bestind][j]==0)? corw : incorw;

      }

      _numsamples += numpts;
      _clfsw.Stop();



      return;
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    void				ClfMilBoost::init(ClfStrongParams *params)
    {
      // initialize model
      _params		= params;
      _myParams	= (ClfMilBoostParams*)params;
      _numsamples = 0;

      _ftrs = Ftr::generate(_myParams->_ftrParams,_myParams->_numFeat);
      if( params->_storeFtrHistory ) Ftr::toViz( _ftrs, "haarftrs" );
      _weakclf.resize(_myParams->_numFeat);
      for( int k=0; k<_myParams->_numFeat; k++ )
        if(_myParams->_weakLearner == string("stump")){
          _weakclf[k] = new ClfOnlineStump(k);
          _weakclf[k]->_ftr = _ftrs[k];
          _weakclf[k]->_lRate = _myParams->_lRate;
          _weakclf[k]->_parent = this;
        }
        else if(_myParams->_weakLearner == string("wstump")){
          _weakclf[k] = new ClfWStump(k);
          _weakclf[k]->_ftr = _ftrs[k];
          _weakclf[k]->_lRate = _myParams->_lRate;
          _weakclf[k]->_parent = this;
        }
        else
          abortError(__LINE__,__FILE__,"incorrect weak clf name");

        if( params->_storeFtrHistory )
          this->_ftrHist.Resize(_myParams->_numFeat,2000);

        _counter=0;
    }
    void				ClfMilBoost::update(SampleSet &posx, SampleSet &negx)
    {
      _clfsw.Start();
      int numneg = negx.size();
      int numpos = posx.size();

      // compute ftrs
      if( !posx.ftrsComputed() ) Ftr::compute(posx, _ftrs);
      if( !negx.ftrsComputed() ) Ftr::compute(negx, _ftrs);

      // initialize H
      static vectorf Hpos, Hneg;
      Hpos.clear(); Hneg.clear();
      Hpos.resize(posx.size(),0.0f), Hneg.resize(negx.size(),0.0f);

      _selectors.clear();
      vectorf posw(posx.size()), negw(negx.size());
      vector<vectorf> pospred(_weakclf.size()), negpred(_weakclf.size());

      // train all weak classifiers without weights
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for( int m=0; m<_myParams->_numFeat; m++ ){
        _weakclf[m]->update(posx,negx);
        pospred[m] = _weakclf[m]->classifySetF(posx);
        negpred[m] = _weakclf[m]->classifySetF(negx);
      }

      // pick the best features
      for( int s=0; s<_myParams->_numSel; s++ ){

        // compute errors/likl for all weak clfs
        vectorf poslikl(_weakclf.size(),1.0f), neglikl(_weakclf.size()), likl(_weakclf.size());
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for( int w=0; w<(int)_weakclf.size(); w++) {
          float lll=1.0f;
          for( int j=0; j<numpos; j++ )
            lll *= ( 1-sigmoid(Hpos[j]+pospred[w][j]) );
          poslikl[w] = (float)-log(1-lll+1e-5);

          lll=0.0f;
          for( int j=0; j<numneg; j++ )
            lll += (float)-log(1e-5f+1-sigmoid(Hneg[j]+negpred[w][j]));
          neglikl[w]=lll;

          likl[w] = poslikl[w]/numpos + neglikl[w]/numneg;
        }

        // pick best weak clf
        vectori order;
        sort_order_des(likl,order);

        // find best weakclf that isn't already included
        for( uint k=0; k<order.size(); k++ )
          if( count( _selectors.begin(), _selectors.end(), order[k])==0 ){
            _selectors.push_back(order[k]);
            break;
          }

          // update H = H + h_m
#ifdef _OPENMP
#pragma omp parallel for
#endif
          for( int k=0; k<posx.size(); k++ )
            Hpos[k] += pospred[_selectors[s]][k];
#ifdef _OPENMP
#pragma omp parallel for
#endif
          for( int k=0; k<negx.size(); k++ )
            Hneg[k] += negpred[_selectors[s]][k];

      }

      if( _myParams->_storeFtrHistory )
        for( uint j=0; j<_selectors.size(); j++ )
          _ftrHist(_selectors[j],_counter) = 1.0f/(j+1);

      _counter++;
      _clfsw.Stop();

      return;
    }


    CvHaarClassifierCascade* Tracker::facecascade = NULL;

    bool			SimpleTracker::init(Matrixu frame, SimpleTrackerParams p, ClfStrongParams *clfparams)
    {
      static Matrixu *img;

      img = &frame;
      frame.initII();

      _clf = ClfStrong::makeClf(clfparams);
      _curState.resize(4);
      for(int i=0;i<4;i++ ) _curState[i] = p._initstate[i];
      SampleSet posx, negx;

      fprintf(stderr,"Initializing Tracker..\n");

      // sample positives and negatives from first frame
      posx.sampleImage(img, (uint)_curState[0],(uint)_curState[1], (uint)_curState[2], (uint)_curState[3], p._init_postrainrad);
      negx.sampleImage(img, (uint)_curState[0],(uint)_curState[1], (uint)_curState[2], (uint)_curState[3], 2.0f*p._srchwinsz, (1.5f*p._init_postrainrad), p._init_negnumtrain);
      if( posx.size()<1 || negx.size()<1 ) return false;

      // train
      _clf->update(posx,negx);
      negx.clear();

      img->FreeII();

      _trparams = p;
      _clfparams = clfparams;
      _cnt = 0;
      return true;
    }


    double			SimpleTracker::track_frame(Matrixu &frame)
    {
      static SampleSet posx, negx, detectx;
      static vectorf prob;
      static vectori order;
      static Matrixu *img;

      double resp;

      img = &frame;
      frame.initII();

      // run current clf on search window
      detectx.sampleImage(img,(uint)_curState[0],(uint)_curState[1],(uint)_curState[2],(uint)_curState[3], (float)_trparams._srchwinsz);
      prob = _clf->classify(detectx,_trparams._useLogR);

      /////// DEBUG /////// display actual probability map
      if( _trparams._debugv ){
        Matrixf probimg(frame.rows(),frame.cols());
        for( uint k=0; k<(uint)detectx.size(); k++ )
          probimg(detectx[k]._row, detectx[k]._col) = prob[k];

        probimg.convert2img().display(2,2);
        cvWaitKey(1);
      }

      // find best location
      int bestind = max_idx(prob);
      resp=prob[bestind];

      _curState[1] = (float)detectx[bestind]._row; 
      _curState[0] = (float)detectx[bestind]._col;

      // train location clf (negx are randomly selected from image, posx is just the current tracker location)

      if( _trparams._negsamplestrat == 0 )
        negx.sampleImage(img, _trparams._negnumtrain, (int)_curState[2], (int)_curState[3]);
      else
        negx.sampleImage(img, (int)_curState[0], (int)_curState[1], (int)_curState[2], (int)_curState[3], 
        (1.5f*_trparams._srchwinsz), _trparams._posradtrain+5, _trparams._negnumtrain);

      if( _trparams._posradtrain == 1 )
        posx.push_back(img, (int)_curState[0], (int)_curState[1], (int)_curState[2], (int)_curState[3]);
      else
        posx.sampleImage(img, (int)_curState[0], (int)_curState[1], (int)_curState[2], (int)_curState[3], _trparams._posradtrain, 0, _trparams._posmaxtrain);

      _clf->update(posx,negx);

      // clean up
      img->FreeII();
      posx.clear(); negx.clear(); detectx.clear();

      _cnt++;

      return resp;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////

    void			Tracker::replayTracker(vector<Matrixu> &vid, string statesfile, string outputvid, uint R, uint G, uint B)
    {
      Matrixf states;
      states.DLMRead(statesfile.c_str());
      Matrixu colorframe;

      // save video file
      CvVideoWriter* w = NULL;
      if( ! outputvid.empty() ){
        w = cvCreateVideoWriter( outputvid.c_str(), CV_FOURCC('I','Y','U','V'), 15, cvSize(vid[0].cols(), vid[0].rows()), 3 );
        if( w==NULL ) abortError(__LINE__,__FILE__,"Error opening video file for output");
      }

      for( uint k=0; k<vid.size(); k++ )
      {	
        vid[k].conv2RGB(colorframe);
        colorframe.drawRect(states(k,2),states(k,3),states(k,0),states(k,1),1,0,2,R,G,B);
        colorframe.drawText(("#"+int2str(k,3)).c_str(),1,25,255,255,0);
        colorframe._keepIpl=true;
        colorframe.display(1,2);
        cvWaitKey(1);
        if( w != NULL )
          cvWriteFrame( w, colorframe.getIpl() );
        colorframe._keepIpl=false; colorframe.freeIpl();
      }

      // clean up
      if( w != NULL )
        cvReleaseVideoWriter( &w );
    }
    void			Tracker::replayTrackers(vector<Matrixu> &vid, vector<string> statesfile, string outputvid, Matrixu colors)
    {
      Matrixu states;
      vector<Matrixu> resvid(vid.size());
      Matrixu colorframe;

      // save video file
      CvVideoWriter* w = NULL;
      if( ! outputvid.empty() ){
        w = cvCreateVideoWriter( outputvid.c_str(), CV_FOURCC('I','Y','U','V'), 15, cvSize(vid[0].cols(), vid[0].rows()), 3 );
        if( w==NULL ) abortError(__LINE__,__FILE__,"Error opening video file for output");
      }

      for( uint k=0; k<vid.size(); k++ ){
        vid[k].conv2RGB(resvid[k]);
        resvid[k].drawText(("#"+int2str(k,3)).c_str(),1,25,255,255,0);
      }

      for( uint j=0; j<statesfile.size(); j++ ){
        states.DLMRead(statesfile[j].c_str());
        for( uint k=0; k<vid.size(); k++ )	
          resvid[k].drawRect(states(k,3),states(k,2),states(k,0),states(k,1),1,0,3,colors(j,0),colors(j,1),colors(j,2));
      }

      for( uint k=0; k<vid.size(); k++ ){
        resvid[k]._keepIpl=true;
        resvid[k].display(1,2);
        cvWaitKey(1);
        if( w!=NULL && k<vid.size()-1)
          Matrixu::WriteFrame(w, resvid[k]);
        resvid[k]._keepIpl=false; resvid[k].freeIpl();
      }

      // clean up
      if( w != NULL )
        cvReleaseVideoWriter( &w );
    }
    bool			Tracker::initFace(TrackerParams* params, Matrixu &frame)
    {
      const char* cascade_name = "haarcascade_frontalface_alt_tree.xml";
      const int minsz = 20;
      if( Tracker::facecascade == NULL )
        Tracker::facecascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 );

      frame.createIpl();
      IplImage *img = frame.getIpl();
      IplImage* gray = cvCreateImage( cvSize(img->width, img->height), IPL_DEPTH_8U, 1 );
      cvCvtColor(img, gray, CV_BGR2GRAY );
      frame.freeIpl();
      cvEqualizeHist(gray, gray);

      CvMemStorage* storage = cvCreateMemStorage(0);
      cvClearMemStorage(storage);
      CvSeq* faces = cvHaarDetectObjects(gray, Tracker::facecascade, storage, 1.05, 3, CV_HAAR_DO_CANNY_PRUNING ,cvSize(minsz, minsz));

      int index = faces->total-1;
      CvRect* r = (CvRect*)cvGetSeqElem( faces, index );



      while(r && (r->width<minsz || r->height<minsz || (r->y+r->height+10)>frame.rows() || (r->x+r->width)>frame.cols() ||
        r->y<0 || r->x<0)){
          r = (CvRect*)cvGetSeqElem( faces, --index);
      }

      //if( r == NULL ){
      //	cout << "ERROR: no face" << endl;
      //	return false;
      //}
      //else 
      //	cout << "Face Found: " << r->x << " " << r->y << " " << r->width << " " << r->height << endl;
      if( r==NULL )
        return false;

      //fprintf(stderr,"x=%f y=%f xmax=%f ymax=%f imgw=%f imgh=%f\n",(float)r->x,(float)r->y,(float)r->x+r->width,(float)r->y+r->height,(float)frame.cols(),(float)frame.rows());

      params->_initstate.resize(4);
      params->_initstate[0]	= (float)r->x;// - r->width;
      params->_initstate[1]	= (float)r->y;// - r->height;
      params->_initstate[2]	= (float)r->width;
      params->_initstate[3]	= (float)r->height+10;


      return true;
    }



    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    TrackerParams::TrackerParams()
    {
      _boxcolor.resize(3);
      _boxcolor[0]	= 204;
      _boxcolor[1]	= 25;
      _boxcolor[2]	= 204;
      _lineWidth		= 2;
      _negnumtrain	= 15;
      _posradtrain	= 1;
      _posmaxtrain	= 100000;
      _init_negnumtrain = 1000;
      _init_postrainrad = 3;
      _initstate.resize(4);
      _debugv			= false;
      _useLogR		= true;
      _disp			= true;
      _initWithFace	= true;
      _vidsave		= "";
      _trsave			= "";
    }

    SimpleTrackerParams::SimpleTrackerParams()
    {
      _srchwinsz		= 30;
      _initstate.resize(4);
      _negsamplestrat	= 1;
    }

  }  // namespace mil
}  // namespace cv
