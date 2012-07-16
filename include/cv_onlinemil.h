/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
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
#ifndef __OPENCV_ONLINE_MIL_H__
#define __OPENCV_ONLINE_MIL_H__

#include <vector>

#include <iostream>
#include <fstream>
#include <float.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#if defined(unix)        || defined(__unix)      || defined(__unix__) \
  || defined(linux)       || defined(__linux)     || defined(__linux__) \
  || defined(sun)         || defined(__sun) \
  || defined(BSD)         || defined(__OpenBSD__) || defined(__NetBSD__) \
  || defined(__FreeBSD__) || defined __DragonFly__ \
  || defined(sgi)         || defined(__sgi) \
  || defined(__MACOSX__)  || defined(__APPLE__) \
  || defined(__CYGWIN__)
#include <typeinfo>
#endif



/** If IPP is included, just use that.  Otherwise, I've included some definitions
 of functions that are used within this code. */
#ifdef HAVE_IPP
#include <ipp.h>
#else
/*****************************************************************************/
typedef enum {
	/* no errors */
	ippStsNoErr                 =   0,   /* No error, it's OK */
} IppStatus;
typedef enum {
	ippAlgHintNone,
	ippAlgHintFast,
	ippAlgHintAccurate
} IppHintAlgorithm;
enum {
	IPPI_INTER_NN     = 1,
	IPPI_INTER_LINEAR = 2,
	IPPI_INTER_CUBIC  = 4,
	IPPI_INTER_CUBIC2P_BSPLINE,     /* two-parameter cubic filter (B=1, C=0) */
	IPPI_INTER_CUBIC2P_CATMULLROM,  /* two-parameter cubic filter (B=0, C=1/2) */
	IPPI_INTER_CUBIC2P_B05C03,      /* two-parameter cubic filter (B=1/2, C=3/10) */
	IPPI_INTER_SUPER  = 8,
	IPPI_INTER_LANCZOS = 16,
	IPPI_SMOOTH_EDGE  = (1 << 31)
};

typedef unsigned char Ipp8u;
typedef float Ipp32f;
typedef double Ipp64f;
typedef signed int Ipp32s;
typedef struct {
	int width;
	int height;
} IppiSize;
typedef struct {
	int x, y, width, height;
} IppiRect;

IppStatus ippiCopy_8u_C1R(const Ipp8u* pSrc, int srcStep,
						  Ipp8u* pDst, int dstStep, IppiSize roiSize);
IppStatus ippiCopy_32f_C1R(const Ipp32f* pSrc, int srcStep,
						   Ipp32f* pDst, int dstStep, IppiSize roiSize);
Ipp8u* ippiMalloc_8u_C1(int widthPixels, int heightPixels, int* pStepBytes);
Ipp32f* ippiMalloc_32f_C1(int widthPixels, int heightPixels, int* pStepBytes) ;
void ippiFree(void* ptr);
IppStatus ippiSet_8u_C1R(Ipp8u value, Ipp8u* pDst, int dstStep, IppiSize roiSize);
IppStatus ippiSet_32f_C1R(Ipp32f value, Ipp32f* pDst, int dstStep, IppiSize roiSize);
IppStatus ippiAdd_8u_C1RSfs(const Ipp8u* pSrc1, int src1Step, const Ipp8u* pSrc2,
							int src2Step, Ipp8u* pDst, int dstStep, IppiSize roiSize,
							int scaleFactor);
IppStatus ippiAdd_32f_C1R(const Ipp32f* pSrc1, int src1Step, const Ipp32f* pSrc2,
						  int src2Step, Ipp32f* pDst, int dstStep, IppiSize roiSize);
IppStatus ippiAddC_8u_C1RSfs(const Ipp8u* pSrc, int srcStep, Ipp8u value, Ipp8u* pDst,
							 int dstStep, IppiSize roiSize, int scaleFactor);
IppStatus ippiAddC_32f_C1R(const Ipp32f* pSrc, int srcStep, Ipp32f value, Ipp32f* pDst,
						   int dstStep, IppiSize roiSize);
IppStatus ippiMulC_8u_C1RSfs(const Ipp8u* pSrc, int srcStep, Ipp8u value, Ipp8u* pDst,
							 int dstStep, IppiSize roiSize, int scaleFactor);
IppStatus ippiMulC_32f_C1R(const Ipp32f* pSrc, int srcStep, Ipp32f value, Ipp32f* pDst,
						   int dstStep, IppiSize roiSize);
IppStatus ippiMul_8u_C1RSfs(const Ipp8u* pSrc1, int src1Step, const Ipp8u* pSrc2,
							int src2Step, Ipp8u* pDst, int dstStep, IppiSize roiSize,
							int scaleFactor);
IppStatus ippiMul_32f_C1R(const Ipp32f* pSrc1, int src1Step, const Ipp32f* pSrc2,
						  int src2Step, Ipp32f* pDst, int dstStep, IppiSize roiSize);
IppStatus ippiSqr_8u_C1RSfs(const Ipp8u* pSrc, int srcStep,
							Ipp8u* pDst, int dstStep, IppiSize roiSize, int scaleFactor);
IppStatus ippiSqr_32f_C1R(const Ipp32f* pSrc, int srcStep,
						  Ipp32f* pDst, int dstStep, IppiSize roiSize);
IppStatus ippiExp_8u_C1RSfs(const Ipp8u* pSrc, int srcStep, Ipp8u* pDst,
							int dstStep, IppiSize roiSize, int scaleFactor);
IppStatus ippiExp_32f_C1R(const Ipp32f* pSrc, int srcStep, Ipp32f* pDst,
						  int dstStep, IppiSize roiSize);
IppStatus ippiTranspose_8u_C1R( const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
							   IppiSize roiSize);
IppStatus ippiMax_8u_C1R(const Ipp8u* pSrc, int srcStep, IppiSize roiSize, Ipp8u* pMax);
IppStatus ippiMax_32f_C1R(const Ipp32f* pSrc, int srcStep, IppiSize roiSize, Ipp32f* pMax);
IppStatus ippiMin_8u_C1R(const Ipp8u* pSrc, int srcStep, IppiSize roiSize, Ipp8u* pMin);
IppStatus ippiMin_32f_C1R(const Ipp32f* pSrc, int srcStep, IppiSize roiSize, Ipp32f* pMin);
IppStatus ippiMaxIndx_8u_C1R(const Ipp8u* pSrc, int srcStep, IppiSize roiSize, Ipp8u* pMax, 
							 int* pIndexX, int* pIndexY);
IppStatus ippiMaxIndx_32f_C1R(const Ipp32f* pSrc, int srcStep, IppiSize roiSize, Ipp32f* pMax, 
							  int* pIndexX, int* pIndexY);
IppStatus ippiMinIndx_8u_C1R(const Ipp8u* pSrc, int srcStep, IppiSize roiSize, Ipp8u* pMin, 
							 int* pIndexX, int* pIndexY);
IppStatus ippiMinIndx_32f_C1R(const Ipp32f* pSrc, int srcStep, IppiSize roiSize, Ipp32f* pMin,
							  int* pIndexX, int* pIndexY);
IppStatus ippiMean_8u_C1R(const Ipp8u* pSrc, int srcStep, IppiSize roiSize, Ipp64f* pMean);
IppStatus ippiMean_32f_C1R(const Ipp32f* pSrc, int srcStep,
						   IppiSize roiSize, Ipp64f* pMean, IppHintAlgorithm hint);
IppStatus ippiMean_StdDev_8u_C1R(const Ipp8u* pSrc, int srcStep, IppiSize roiSize,
								 Ipp64f* pMean, Ipp64f* pStdDev);
IppStatus ippiMean_StdDev_32f_C1R(const Ipp32f* pSrc, int srcStep, IppiSize roiSize,
								  Ipp64f* pMean, Ipp64f* pStdDev);
IppStatus ippiSum_8u_C1R(const Ipp8u* pSrc, int srcStep, IppiSize roiSize, Ipp64f* pSum);
IppStatus ippiSum_32f_C1R(const Ipp32f* pSrc, int srcStep,
						  IppiSize roiSize, Ipp64f* pSum, IppHintAlgorithm hint);
IppStatus ippiIntegral_8u32f_C1R(const Ipp8u* pSrc, int srcStep,
								 Ipp32f* pDst, int dstStep, IppiSize roiSize, Ipp32f val);
IppStatus ippiGetAffineTransform(IppiRect srcRoi, const double quad[4][2], double coeffs[2][3]);
IppStatus ippiWarpAffine_8u_C1R(const Ipp8u* pSrc, IppiSize srcSize, int srcStep, IppiRect srcRoi, 
								Ipp8u* pDst, int dstStep, IppiRect dstRoi, const double coeffs[2][3], 
								int interpolation);
IppStatus ippiFilterRow_8u_C1R(const Ipp8u* pSrc, int srcStep,
							   Ipp8u* pDst, int dstStep, IppiSize dstRoiSize, const Ipp32s* pKernel,
							   int kernelSize, int xAnchor, int divisor);
IppStatus ippiFilterColumn_8u_C1R(const Ipp8u* pSrc,
								  int srcStep, Ipp8u* pDst, int dstStep, IppiSize dstRoiSize,
								  const Ipp32s* pKernel, int kernelSize, int yAnchor, int divisor);
IppStatus ippiResize_8u_C1R(const Ipp8u* pSrc, IppiSize srcSize, int srcStep, IppiRect srcRoi,
							Ipp8u* pDst, int dstStep, IppiSize dstRoiSize,
							double xFactor, double yFactor, int interpolation);
#endif   // #ifdef HAVE_IPP



namespace cv
{
	namespace mil
	{
		typedef unsigned char  uchar;
		typedef unsigned short ushort;
		typedef unsigned int   uint;
		typedef unsigned long  ulong;
		
		typedef std::vector<float>	vectorf;
		typedef std::vector<double>	vectord;
		typedef std::vector<int>		vectori;
		typedef std::vector<long>	vectorl;
		typedef std::vector<uchar>	vectoru;
		typedef std::vector<string>	vectorString;
		typedef std::vector<bool>	vectorb;
		
#define	PI	3.1415926535897931
#define PIINV 0.636619772367581
#define INF 1e99
#define INFf 1e50f
#define EPS 1e-99;
#define EPSf 1e-50f
#define ERASELINE "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b"
		
#define  sign(s)	((s > 0 ) ? 1 : ((s<0) ? -1 : 0))
#define  round(v)   ((int) (v+0.5))
		
		//static CvRNG rng_state = cvRNG((int)time(NULL));
		static CvRNG rng_state = cvRNG(1);
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////
		// random generator stuff
		void				randinitalize( const int init );
		int					randint( const int min=0, const int max=5 );
		vectori				randintvec( const int min=0, const int max=5, const uint num=100 );
		vectorf				randfloatvec( const uint num=100 );
		float				randfloat();
		float				randgaus(const float mean, const float std);
		vectorf				randgausvec(const float mean, const float std, const int num=100);
		vectori				sampleDisc(const vectorf &weights, const uint num=100);
		
		inline float		sigmoid(float x)
		{
			return 1.0f/(1.0f+exp(-x));
		}
		inline double		sigmoid(double x)
		{
			return 1.0/(1.0+exp(-x));
		}
		
		inline vectorf		sigmoid(vectorf x)
		{
			vectorf r(x.size());
			for( uint k=0; k<r.size(); k++ )
				r[k] = sigmoid(x[k]);
			return r;
			
		}
		
		inline int			force_between(int i, int min, int max)
		{
			return std::min<int>(std::max<int>(i,min),max);
		}
		
		string				int2str( int i, int ndigits );
		//////////////////////////////////////////////////////////////////////////////////////////////////////
		// vector functions
		template<class T> class				SortableElement
		{
		public:
			T _val; int _ind;
			SortableElement() {};
			SortableElement( T val, int ind ) { _val=val; _ind=ind; }
			bool operator< ( SortableElement &b ) { return (_val > b._val ); };
		};
		
		template<class T> class				SortableElementRev
		{
		public:
			T _val; int _ind;
			SortableElementRev() {};
			SortableElementRev( T val, int ind ) { _val=val; _ind=ind; }
			bool operator< ( SortableElementRev<T> &b ) { return (_val < b._val ); };
		};
		
		template<class T> void				sort_order( std::vector<T> &v, vectori &order )
		{
			uint n=(uint)v.size();
			std::vector< SortableElement<T> > v2; 
			v2.resize(n); 
			order.clear(); order.resize(n);
			for( uint i=0; i<n; i++ ) {
				v2[i]._ind = i;
				v2[i]._val = v[i];
			}
			std::sort( v2.begin(), v2.end() );
			for( uint i=0; i<n; i++ ) {
				order[i] = v2[i]._ind;
				v[i] = v2[i]._val;
			}
		};
		
		static bool CompareSortableElementRev(const SortableElementRev<float>& i, const SortableElementRev<float>& j)
		{
			return i._val < j._val;
		}
		
		template<class T> void				sort_order_des( std::vector<T> &v, vectori &order )
		{
			uint n=(uint)v.size();
			std::vector< SortableElementRev<T> > v2; 
			v2.resize(n); 
			order.clear(); order.resize(n);
			for( uint i=0; i<n; i++ ) {
				v2[i]._ind = i;
				v2[i]._val = v[i];
			}
			//std::sort( v2.begin(), v2.end() );
			std::sort( v2.begin(), v2.end(), CompareSortableElementRev);
			for( uint i=0; i<n; i++ ) {
				order[i] = v2[i]._ind;
				v[i] = v2[i]._val;
			}
		};
		
		template<class T> void				resizeVec(std::vector<std::vector<T> > &v, int sz1, int sz2, T val=0)
		{
			v.resize(sz1);
			for( int k=0; k<sz1; k++ )
				v[k].resize(sz2,val);
		};
		
		
		
		////template<class T> inline uint		min_idx( const vector<T> &v )
		////{
		////  return (uint)(min_element(v.begin(),v.end())._Myptr-v.begin()._Myptr);
		////}
		template<class T> inline uint		max_idx( const std::vector<T> &v )
		{
#if _MSC_VER <= 1500
			//return (uint)(max_element(v.begin(),v.end())._Myptr-v.begin()._Myptr);
#else
			//return (uint)(max_element(v.begin(),v.end())._Ptr-v.begin()._Ptr);
#endif
			const T* findPtr = &(*max_element(v.begin(),v.end()));
			const T* beginPtr = &(*v.begin());
			return (uint)(findPtr-beginPtr);
		}
		
		template<class T> inline void		normalizeVec( std::vector<T> &v )
		{
			T sum = 0;
			for( uint k=0; k<v.size(); k++ ) sum+=v[k];
			for( uint k=0; k<v.size(); k++ ) v[k]/=sum;
		}
		
		
		template<class T> std::ostream&			operator<<(std::ostream& os, const std::vector<T>& v)
		{  //display vector
			os << "[ " ;
			for (size_t i=0; i<v.size(); i++)
				os << v[i] << " ";
			os << "]";
			return os;
		}
		//////////////////////////////////////////////////////////////////////////////////////////////////////
		// error functions
		inline void							abortError( const int line, const char *file, const char *msg=NULL) 
		{
			if( msg==NULL )
				fprintf(stderr, "%s %d: ERROR\n", file, line );
			else
				fprintf(stderr, "%s %d: ERROR: %s\n", file, line, msg );
			exit(0);
		}
		
		
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////
		// Stop Watch
		class								StopWatch
		{
		public:
			StopWatch() { Reset(); }
			StopWatch(bool start) { Reset(); if(start) Start(); }
			
			inline void Reset(bool restart=false) { 
				totaltime=0; 
				running=false; 
				if(restart) Start();
			}
			
			inline double Elapsed(bool restart=false) { 
				if(running) Stop();
				if(restart) Start();
				return totaltime; 
			}
			
			inline char* ElapsedStr(bool restart=false) { 
				if(running) Stop();
				if( totaltime < 60.0f )
					sprintf( totaltimeStr, "%5.2fs", totaltime );
				else if( totaltime < 3600.0f )
					sprintf( totaltimeStr, "%5.2fm", totaltime/60.0f );
				else 
					sprintf( totaltimeStr, "%5.2fh", totaltime/3600.0f );
				if(restart) Start();
				return totaltimeStr; 
			}
			
			inline void Start() {
				assert(!running); 
				running=true;
				sttime = clock();
			}
			
			inline void Stop() {
				totaltime += ((double) (clock() - sttime)) / CLOCKS_PER_SEC;
				assert(running);
				running=false;
			}
			
		protected:
			bool running;
			clock_t sttime;
			double totaltime;
			char totaltimeStr[100];
		};
		
		
		template<class T> class Matrix;
		typedef Matrix<float>	Matrixf;
		typedef Matrix<uchar>	Matrixu;
		
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		// This is an IPP based matrix class.  It can be used for both matrix math and for multi channel 
		// image manipulation.
		
		template<class T> class Matrix
		{
			
		public:
			//////////////////////////////////////////////////////////////////////////////////////////////////////
			// members
			int				_rows,_cols,_depth;
			std::vector<void*>	_data;
		private:
			// image specific
			int				_dataStep;
			IplImage		*_iplimg;
			// integral images
			vector<float*>	_iidata;
			int				_iidataStep;
			int				_iipixStep;
			bool			_ii_init;
			
			IppiSize		_roi; //whole image roi (needed for some functions)
			IppiRect		_roirect;
			
		public:
			bool			_keepIpl;  // if set to true, calling freeIpl() will have no effect;  this is for speed up only...
			
		public:
			//////////////////////////////////////////////////////////////////////////////////////////////////////
			// constructors
			Matrix();
			Matrix( int rows, int cols );
			Matrix( int rows, int cols, int depth );
			Matrix( const Matrix<T>& x );
			Matrix( const std::vector<T>& v );
			~Matrix();
			static		Matrix<T>	Eye( int sz ); 
			void		Resize( uint rows, uint cols, uint depth=1 );
			void		Resize( uint depth );
			void		Free();
			void		Set(T val);
			void		Set(T val, int channel);
			
			//////////////////////////////////////////////////////////////////////////////////////////////////////
			// access
			T&			operator() ( const int k ) const;
			T&			operator() ( const int row, const int col ) const;
			T&			operator() ( const int row, const int col, const int depth ) const;
			std::vector<T>	operator() ( const vectori rows, const vectori cols );
			std::vector<T>	operator() ( const vectori rows, const vectori cols, const vectori depths );
			float		ii ( const int row, const int col, const int depth ) const;
			Matrix<T>	getCh(uint ch);
			IplImage*	getIpl() { return _iplimg; };
			
			template<typename S>
			inline S* getRow(int row)
			{
				if( typeid(S) == typeid(uchar) )
					return &((S*)_data[0])[row*(_dataStep)];
				else
					return &((S*)_data[0])[row*(_dataStep/sizeof(float))];
			}
			
			int			rows() const { return _rows; };
			int			cols() const { return _cols; };
			int			depth() const { return _depth; };
			uint		size() const { return _cols*_rows; };
			int			length() const { return max(_cols,_rows); };

      void setData(const cv::Mat & image, const int channel);
			
			//Matrix<T>& operator= ( const vector<T> &x )
			//vector<T>	toVec();
			
			//////////////////////////////////////////////////////////////////////////////////////////////////////
			// matrix operations
			Matrix<T>&	operator=  ( const Matrix<T> &x );
			Matrix<T>&	operator=  ( const std::vector<T> &x );
			Matrix<T>	operator+ ( const Matrix<T> &b ) const;
			Matrix<T>	operator+ ( const T &a) const;
			Matrix<T>	operator- ( const Matrix<T> &b ) const;
			Matrix<T>	operator- ( const T &a) const;
			Matrix<T>	operator* ( const T &a) const;
			Matrix<T>	operator& ( const Matrix<T> &b) const;
			Matrixu		operator< ( const T &a) const;
			Matrixu		operator> ( const T &a) const;
			Matrix<T>	normalize() const;
			Matrix<T>	Sqr() const;
			Matrix<T>	Exp() const;
			void		Trans(Matrix<T> &res);
			T			Max(uint channel=0) const;
			T			Min(uint channel=0) const;
			double		Sum(uint channel=0) const;
			void		Max(T &val, uint &row, uint &col, uint channel=0) const;
			void		Min(T &val, uint &row, uint &col, uint channel=0) const;
			float		Mean(uint channel=0) const;
			float		Var(uint channel=0) const;
			float		VarW(const Matrixf &w, T *mu=NULL) const;
			float		MeanW(const vectorf &w)  const;
			float		MeanW(const Matrixf &w)  const;
			float		Dot(const Matrixf &x);
			
			
			//////////////////////////////////////////////////////////////////////////////////////////////////////
			// image operations
			//		Note: many of these functions use OpenCV.  To do this, createIpl() is called to create an Ipl version of the image (OpenCV format).
			//		At the end of these functions freeIpl() is called to erase the Ipl version.  For speed up purposes, you can see Matrix._keepIpl=true
			//		to prevent these functions from erasing the Ipl image.  However, care must be taken to erase the image and create a new one when the
			//		Matrix changes or gets updated somehow (otherwise the Matrix will change, but the Ipl will stay the same).
			
			void		initII();
			bool		isInitII() const { return _ii_init; };
			void		FreeII();
			float		sumRect(const IppiRect &rect, int channel) const;
			void		drawRect(IppiRect rect, int lineWidth=3, int R=255, int G=0, int B=0);
			void		drawRect(float width, float height, float x,float y, float sc, float th, int lineWidth=3, int R=255, int G=0, int B=0);
			void		drawEllipse(float height, float width, float x,float y, int lineWidth=3, int R=255, int G=0, int B=0);
			void		drawEllipse(float height, float width, float x,float y, float startang, float endang, int lineWidth=3, int R=255, int G=0, int B=0);
			void		drawText(const char* txt, float x, float y, int R=255, int G=255, int B=0);
			void		warp(Matrixu &res, uint rows, uint cols, float x, float y, float sc=1.0f, float th=0.0f, float sr=1.0f, float phi=0.0f);
			void		warpAll(uint rows, uint cols, std::vector<vectorf> params, std::vector<Matrixu> &res);
			void		computeGradChannels();
			Matrixu		imResize(float p, float x=-1);
			void		conv2RGB(Matrixu &res);
			void		conv2BW(Matrixu &res);
			float		dii_dx(uint x, uint y, uint channel=0);
			float		dii_dy(uint x, uint y, uint channel=0);
			
			void		createIpl(bool force=false);  
			void		freeIpl();
			
			void		LoadImage(const char *filename, bool color=false); 
			void		SaveImage(const char *filename);
			static void	SaveImages(std::vector<Matrixu> imgs, const char *dirname, float resize=1.0f);
			static std::vector<Matrixu> LoadVideo(const char *dirname, const char *basename, const char *ext, int start, int end, int digits, bool color=false);
			static std::vector<Matrixu> LoadVideo(const char *fname, bool color=false, int maxframes=10000);
			static bool	CaptureImage(CvCapture* capture, Matrixu &res, int color=0);
			static bool WriteFrame(CvVideoWriter* w, Matrixu &img);
			static void	PlayCam(int color=0, const char* fname=NULL);
			static void PlayCamOpenCV();
			
			static Matrix<T>		vecMat2Mat(const std::vector<Matrix<T> > &x);
			static std::vector<Matrix<T> >	vecMatTranspose(const std::vector<Matrix<T> > &x);
			
			bool		DLMRead( const char *fname, const char *delim="," ); // compatible with Matlab dlmread & dlmwrite
			bool		DLMWrite( const char *fname, const char *delim="," );
			
			void		display(int fignum, float p=1.0f);
			
			///////////////////////////////////////////////////////////////////////////////////////////////////////
			// misc
			Matrixu		convert2img(float min=0.0f, float max=0.0f);
			
			
			void		IplImage2Matrix(IplImage *img);
			void		GrayIplImage2Matrix(IplImage *img);
			
			
		};
		
		template<class T> std::ostream&			operator<< ( std::ostream& os, const Matrix<T>& x );
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		// constructors
		template<class T>					Matrix<T>::Matrix() 
		{
			_rows		= 0;
			_cols		= 0;
			_depth		= 0;
			_iplimg		= NULL;
			_keepIpl	= false;
		}
		
		template<class T>					Matrix<T>::Matrix(int rows, int cols) 
		{
			_rows		= 0;
			_cols		= 0;
			_depth		= 0;
			_iplimg		= NULL;
			_keepIpl	= false;
			_ii_init	= false;
			Resize(rows,cols,1);	
		}
		
		template<class T>					Matrix<T>::Matrix(int rows, int cols, int depth) 
		{
			_rows		= 0;
			_cols		= 0;
			_depth		= 0;
			_iplimg		= NULL;
			_keepIpl	= false;
			_ii_init	= false;
			Resize(rows,cols,depth);	
		}
		
		template<class T>					Matrix<T>::Matrix(const Matrix<T> &a)
		{
			_rows		= 0;
			_cols		= 0;
			_depth		= 0;
			_iplimg		= NULL;
			_keepIpl	= (typeid(T) == typeid(uchar)) && a._keepIpl;
			_ii_init	= false;
			Resize(a._rows, a._cols, a._depth);
			if( typeid(T) == typeid(uchar) )
				for( uint k=0; k<_data.size(); k++ )
					ippiCopy_8u_C1R((unsigned char*)a._data[k], a._dataStep, (unsigned char*)_data[k], _dataStep, _roi );
			else
				for( uint k=0; k<_data.size(); k++ )
					ippiCopy_32f_C1R((float*)a._data[k], a._dataStep, (float*)_data[k], _dataStep, _roi );
			if( a._ii_init ){
				_iidata.resize(a._iidata.size());
				
				for( uint k=0; k<_iidata.size(); k++ ){
					if( _iidata[k] != NULL ) ippiFree(_iidata[k]);
					_iidata[k] = ippiMalloc_32f_C1(_cols+1,_rows+1,&(_iidataStep));
					_iipixStep = _iidataStep/sizeof(float);
					ippiCopy_32f_C1R((float*)a._iidata[k], a._iidataStep, (float*)_iidata[k], _iidataStep, _roi );
				}
				_ii_init = true;
			}
			
			if( a._iplimg != NULL && typeid(T) == typeid(uchar))
			{
				((Matrixu*)this)->createIpl();
				cvCopy(a._iplimg, _iplimg);
			}
		}
		
		template<class T> Matrix<T>			Matrix<T>::Eye( int sz )
		{
			Matrix<T> res(sz,sz);
			for( int k=0; k<sz; k++ )
				res(k,k) = 1;
			return res;
		}
    
    template<class T>
    void Matrix<T>::setData(const cv::Mat & image, const int channel)
    {
      if( typeid(T) == typeid(uchar) )
      {
        ippiCopy_8u_C1R((unsigned char*)image.data, _dataStep, (unsigned char*)_data[channel], _dataStep, _roi);
      }
      else
      {
        ippiCopy_32f_C1R((float*)image.data, _dataStep, (float*)_data[channel], _dataStep, _roi);
      }
    }

		template<class T> void				Matrix<T>::Resize(uint rows, uint cols, uint depth) 
		{
			if( rows<0 || cols<0 )
				abortError(__LINE__, __FILE__,"NEGATIVE MATRIX SIZE");
			
			if( _rows == (int)rows && _cols == (int)cols && _depth == (int)depth ) return;
			bool err = false;
			Free();
			_rows = rows;
			_cols = cols;
			_depth = depth;
			
			_data.resize(depth);
			
			for( uint k=0; k<_data.size(); k++ ){
				if( typeid(T) == typeid(uchar) ){
					_data[k] = (void*)ippiMalloc_8u_C1(cols,rows,&(_dataStep));
				}
				else{
					_data[k] = (void*)ippiMalloc_32f_C1(cols,rows,&(_dataStep));//malloc((uint)rows*cols*sizeof(float));
				}
				err = err || _data[k] == NULL;
			}
			
			_roi.width = cols;
			_roi.height = rows;
			_roirect.width = cols;
			_roirect.height = rows;
			_roirect.x = 0;
			_roirect.y = 0;
			Set(0);
			
			//free ipl
			if( _iplimg != NULL )
				cvReleaseImage(&_iplimg);
			
			if( err )
				abortError(__LINE__, __FILE__,"OUT OF MEMORY");
		}
		
		template<class T> void				Matrix<T>::Resize(uint depth) 
		{
			
			if( _depth == depth ) return;
			bool err=false;
			
			
			_data.resize(depth);
			
			for( uint k=_depth; k<depth; k++ ){
				if( typeid(T) == typeid(uchar) ){
					_data[k] = (void*)ippiMalloc_8u_C1(_cols,_rows,&(_dataStep));
				}
				else{
					_data[k] = (void*)ippiMalloc_32f_C1(_cols,_rows,&(_dataStep));//malloc((uint)rows*cols*sizeof(float));
				}
				err = err || _data[k] == NULL;
				Set(0,k);
			}
			_depth = depth;
			
			
			if( err )
				abortError(__LINE__, __FILE__,"OUT OF MEMORY");
		}
		
		template<class T> void				Matrix<T>::Free() 
		{
			if( _ii_init ) FreeII();
			if( _iplimg != NULL ) cvReleaseImage(&_iplimg);
			_ii_init = false;
			
			for( uint k=0;  k<_data.size(); k++ )
				if( _data[k] != NULL )
					if( typeid(T) == typeid(uchar) ){
						ippiFree((unsigned char*)_data[k]);
					}
					else{
						ippiFree((float*)_data[k]);
					}
			
			_rows = 0;
			_cols = 0;
			_depth = 0;
			_data.resize(0);
		}
		
		template<class T> 
		float				Matrix<T>::sumRect(const IppiRect &rect, int channel) const
		{
			if (typeid(T) != typeid(unsigned char)) return 0;
			
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
		
		template<class T> void				Matrix<T>::Set(T val) 
		{
			for( uint k=0; k<_data.size(); k++ )
				if( typeid(T) == typeid(uchar) ){
					ippiSet_8u_C1R((unsigned char)val,(unsigned char*)_data[k], _dataStep,_roi);
				}
				else{
					ippiSet_32f_C1R((float)val,(float*)_data[k], _dataStep,_roi);
				}
			//for( uint j=0; j<(uint)_rows*_dataStep; j++ )
			//	((float*)_data[k])[j] = val;
		}
		template<class T> void				Matrix<T>::Set(T val, int k) 
		{
			if( typeid(T) == typeid(uchar) ){
				ippiSet_8u_C1R((unsigned char)val,(unsigned char*)_data[k], _dataStep,_roi);
			}
			else{
				ippiSet_32f_C1R((float)val,(float*)_data[k], _dataStep,_roi);
			}
		}
		
		
		template<class T> void				Matrix<T>::FreeII() 
		{
			for( uint k=0;  k<_iidata.size(); k++ )
				ippiFree(_iidata[k]);
			_iidata.resize(0);
			_ii_init = false;
		}
		
		template<class T>					Matrix<T>::~Matrix()
		{
			Free();
		}
		
		
		template<class T> 
		void					Matrix<T>::createIpl(bool force)
		{
			if (typeid(T) != typeid(unsigned char)) return;
			
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
		
		template<class T> 
		void					Matrix<T>::freeIpl()
		{
			if (typeid(T) != typeid(unsigned char)) return;
			if( !_keepIpl && _iplimg != NULL) cvReleaseImage(&_iplimg);
		}
		
		template<class T> 
		void					Matrix<T>::IplImage2Matrix(IplImage *img)
		{
			if (typeid(T) != typeid(unsigned char))
			{
				return;
			}
			
			//Resize(img->height, img->width, img->nChannels);
			bool origin = img->origin==1;
			
			if( _depth == 1 )
				for( int row=0; row<_rows; row++ )
					for( int k=0; k<_cols*3; k+=3 )
						if( origin )
							((T*)_data[0])[(_rows - row - 1)*_dataStep+k/3] = img->imageData[row*img->widthStep+k];
						else
							((T*)_data[0])[row*_dataStep+k/3] = img->imageData[row*img->widthStep+k];
						else
#ifdef _OPENMP
#pragma omp parallel for
#endif
							for( int row=0; row<_rows; row++ )
								for( int k=0; k<_cols*3; k+=3 ){
									if( origin ){
										((T*)_data[0])[(_rows - row - 1)*_dataStep+k/3] = img->imageData[row*img->widthStep+k+2];
										((T*)_data[1])[(_rows - row - 1)*_dataStep+k/3] = img->imageData[row*img->widthStep+k+1];
										((T*)_data[2])[(_rows - row - 1)*_dataStep+k/3] = img->imageData[row*img->widthStep+k];
									}
									else{
										((T*)_data[0])[row*_dataStep+k/3] = img->imageData[row*img->widthStep+k+2];
										((T*)_data[1])[row*_dataStep+k/3] = img->imageData[row*img->widthStep+k+1];
										((T*)_data[2])[row*_dataStep+k/3] = img->imageData[row*img->widthStep+k];
									}
								}
			
			if( _keepIpl )
				_iplimg = img;
		}
		
		template<class T> 
		void					Matrix<T>::GrayIplImage2Matrix(IplImage *img)
		{
			if (typeid(T) != typeid(unsigned char))
			{
				return;
			}
			
			//Resize(img->height, img->width, img->nChannels);
			bool origin = img->origin==1;
			
			if( _depth == 1 )
				for( int row=0; row<_rows; row++ )
					for( int k=0; k<_cols; k++ )
						if( origin )
							((T*)_data[0])[(_rows - row - 1)*_dataStep+k] = img->imageData[row*img->widthStep+k];
						else
							((T*)_data[0])[row*_dataStep+k] = img->imageData[row*img->widthStep+k];
			
		}
		
		template<class T> 
		void					Matrix<T>::display(int fignum, float p)
		{
			if (typeid(T) != typeid(unsigned char)) return;
			
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
		
		template<class T> 
		Matrixu				Matrix<T>::imResize(float r, float c)
		{
			if (typeid(T) != typeid(unsigned char)) return Matrix<T>(0,0);
			
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
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		// operators
		
		template<class T> inline T&			Matrix<T>::operator() ( const int row, const int col, const int depth ) const
		{
			if( typeid(T) == typeid(uchar) )
				return (T&) ((unsigned char*)_data[depth])[row*(_dataStep) + col];
			else
				return (T&) ((float*)_data[depth])[row*(_dataStep/sizeof(float)) + col];
		}
		
		template<class T> inline T&			Matrix<T>::operator() ( const int row, const int col ) const
		{
			return (*this)(row,col,0);
		}
		template<class T> inline T&			Matrix<T>::operator() ( const int k ) const
		{
			return (*this)(k/_cols,k%_cols,0);
		}
		template<class T> inline std::vector<T>	Matrix<T>::operator() ( const vectori rows, const vectori cols )
		{
			assert(rows.size() == cols.size());
			std::vector<T> res;
			res.resize(rows.size());
			for( uint k=0; k<rows.size(); k++ )
				res[k] = (*this)(rows[k],cols[k]);
			return res;
		}
		
		template<class T> inline std::vector<T>	Matrix<T>::operator() ( const vectori rows, const vectori cols, const vectori depths )
		{
			assert(rows.size() == cols.size() && cols.size() == depths.size());
			std::vector<T> res;
			res.resize(rows.size());
			for( uint k=0; k<rows.size(); k++ )
				res[k] = (*this)(rows[k],cols[k],depths[k]);
			return res;
		}
		
		template<class T> inline Matrix<T>	Matrix<T>::getCh(uint ch)
		{
			Matrix<T> a(_rows, _cols, 1);
			if( typeid(T) == typeid(uchar) )
				ippiCopy_8u_C1R((unsigned char*)_data[ch], _dataStep, (unsigned char*)a._data[0], a._dataStep, _roi );
			else
				ippiCopy_32f_C1R((float*)_data[ch], _dataStep, (float*)a._data[0], a._dataStep, _roi );
			return a;
		}
		template<class T> Matrix<T>&		Matrix<T>::operator= ( const Matrix<T> &a )
		{
			if( this != &a ){
				Resize(a._rows, a._cols, a._depth);
				if( typeid(T) == typeid(uchar) )
					for( uint k=0; k<_data.size(); k++ )
						ippiCopy_8u_C1R((unsigned char*)a._data[k], a._dataStep, (unsigned char*)_data[k], _dataStep, _roi );
				else
					for( uint k=0; k<_data.size(); k++ )
						ippiCopy_32f_C1R((float*)a._data[k], a._dataStep, (float*)_data[k], _dataStep, _roi );
				//ippmCopy_va_32f_SS((float*)a._data[k],sizeof(float)*_cols,sizeof(float),(float*)_data[k],sizeof(float)*_cols,sizeof(float),_cols,_rows);
				if( a._ii_init ){
					_iidata.resize(a._iidata.size());
					
					for( uint k=0; k<_iidata.size(); k++ ){
						if( _iidata[k] != NULL ) ippiFree(_iidata[k]);
						_iidata[k] = ippiMalloc_32f_C1(_cols+1,_rows+1,&(_iidataStep));
						_iipixStep = _iidataStep/sizeof(float);
						ippiCopy_32f_C1R((float*)a._iidata[k], a._iidataStep, (float*)_iidata[k], _iidataStep, _roi );
					}
					_ii_init = true;
				}
				
			}
			return (*this);
		}
		
		
		
		
		
		template<class T> Matrix<T>&		Matrix<T>::operator= ( const std::vector<T> &a )
		{
			Resize(1,a.size(),1);
			for( uint k=0; k<a.size(); k++ )
				(*this)(k) = a[k];
			
			return (*this);
		}
		
		
		
		
		
		template<class T> Matrix<T>			Matrix<T>::operator+ ( const Matrix<T> &a ) const
		{
			Matrix<T> res(rows(),cols());
			assert(rows()==a.rows() && cols()==a.cols());
			
			if( typeid(T) == typeid(uchar) )
				for( uint k=0; k<_data.size(); k++ )
					ippiAdd_8u_C1RSfs((unsigned char*)a._data[k], a._dataStep, (unsigned char*)_data[k], _dataStep,(unsigned char*)res._data[k], res._dataStep,
									  _roi, 0);
			else
				for( uint k=0; k<_data.size(); k++ )
					ippiAdd_32f_C1R((float*)a._data[k], a._dataStep, (float*)_data[k], _dataStep,(float*)res._data[k], res._dataStep,
									_roi);
			
			return res;
		}
		
		template<class T> Matrix<T>			Matrix<T>::operator+ ( const T &a ) const
		{
			Matrix<T> res;
			res.Resize(rows(),cols(),depth());
			
			if( typeid(T) == typeid(uchar) )
				for( uint k=0; k<_data.size(); k++ )
					ippiAddC_8u_C1RSfs((unsigned char*)_data[k], _dataStep,(unsigned char)a,(unsigned char*)res._data[k], res._dataStep,
									   _roi, 0);
			else
				for( uint k=0; k<_data.size(); k++ )
					ippiAddC_32f_C1R((float*)_data[k], _dataStep,(float)a,(float*)res._data[k], res._dataStep,
									 _roi);
			
			return res;
		}
		template<class T> Matrix<T>			Matrix<T>::operator- ( const Matrix<T> &a ) const
		{
			return (*this) + (a*-1);
		}
		
		template<class T> Matrix<T>			Matrix<T>::operator- ( const T &a ) const
		{
			return (*this) + (a*-1);
		}
		template<class T> Matrix<T>			Matrix<T>::operator* ( const T &a ) const
		{
			Matrix<T> res(rows(),cols());
			
			if( typeid(T) == typeid(uchar) )
				for( uint k=0; k<_data.size(); k++ )
					ippiMulC_8u_C1RSfs((unsigned char*)_data[k], _dataStep,(unsigned char)a,(unsigned char*)res._data[k], res._dataStep,
									   _roi, 0);
			else
				for( uint k=0; k<_data.size(); k++ )
					ippiMulC_32f_C1R((float*)_data[k], _dataStep,(float)a,(float*)res._data[k], res._dataStep,
									 _roi);
			
			return res;
		}
		template<class T> Matrix<T>			Matrix<T>::operator& ( const Matrix<T> &b) const
		{
			Matrix<T> res(rows(),cols());
			assert(rows()==b.rows() && cols()==b.cols() && depth()==b.depth());
			
			if( typeid(T) == typeid(uchar) )
				for( uint k=0; k<_data.size(); k++ )
					ippiMul_8u_C1RSfs((unsigned char*)_data[k], _dataStep,(unsigned char*)b._data[k], b._dataStep,(unsigned char*)res._data[k], res._dataStep,
									  _roi, 1);
			else
				for( uint k=0; k<_data.size(); k++ )
					ippiMul_32f_C1R((float*)_data[k], _dataStep,(float*)b._data[k], b._dataStep,(float*)res._data[k], res._dataStep,
									_roi);
			
			return res;
		}
		
		template<class T> Matrixu			Matrix<T>::operator< ( const T &b) const
		{
			Matrixu res(rows(),cols());
			
			for( uint i=0; i<size(); i++ )
				res(i) = (uint) ((*this)(i) < b);
			
			return res;
		}
		
		template<class T> Matrixu			Matrix<T>::operator> ( const T &b) const
		{
			Matrixu res(rows(),cols());
			
			for( uint i=0; i<size(); i++ )
				res(i) = (uint) ((*this)(i) > b);
			
			return res;
		}
		template<class T> Matrix<T>			Matrix<T>::normalize() const
		{
			double sum = this->Sum();
			return (*this) * (T)(1.0/(sum+1e-6));
		}
		template<class T> Matrix<T>			Matrix<T>::Sqr ( ) const
		{
			Matrix<T> res(rows(),cols());
			
			if( typeid(T) == typeid(uchar) )
				for( uint k=0; k<_data.size(); k++ )
					ippiSqr_8u_C1RSfs((unsigned char*)_data[k], _dataStep,(unsigned char*)res._data[k], res._dataStep,
									  _roi, 0);
			else
				for( uint k=0; k<_data.size(); k++ )
					ippiSqr_32f_C1R((float*)_data[k], _dataStep,(float*)res._data[k], res._dataStep,
									_roi);
			
			return res;
		}
		template<class T> Matrix<T>			Matrix<T>::Exp ( ) const
		{
			Matrix<T> res(rows(),cols());
			
			if( typeid(T) == typeid(uchar) )
				for( uint k=0; k<_data.size(); k++ )
					ippiExp_8u_C1RSfs((unsigned char*)_data[k], _dataStep,(unsigned char*)res._data[k], res._dataStep,
									  _roi, 0);
			else
				for( uint k=0; k<_data.size(); k++ )
					ippiExp_32f_C1R((float*)_data[k], _dataStep,(float*)res._data[k], res._dataStep,
									_roi);
			
			return res;
		}
		template<class T> std::ostream&			operator<<(std::ostream& os, const Matrix<T>& x)
		{  //display matrix
			os << "[ ";
			char tmp[1024];
			for (int j=0; j<x.rows(); j++) {
				if( j>0 ) os << "  ";
				for (int i=0; i<x.cols(); i++) {
					if( typeid(T) == typeid(uchar) )
						sprintf(tmp,"%3d",(int)x(j,i));
					else
						sprintf(tmp,"%02.2f",(float)x(j,i));
					os << tmp << " ";
				}
				if( j!=x.rows()-1 )
					os << "\n";
			}
			os << "]";
			return os;
		}
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////	
		template<class T> void				Matrix<T>::Trans(Matrix<T> &res) 
		{
			res.Resize(_cols,_rows,_depth);
			for( uint k=0; k<res._data.size(); k++ )
				if( typeid(T) == typeid(uchar) )
					ippiTranspose_8u_C1R((unsigned char*)_data[k], _dataStep,(unsigned char*)res._data[k], res._dataStep, _roi);
				else
					abortError(__LINE__,__FILE__,"Trans not implemented for floats");
			//ippmTranspose_m_32f((float*)_data[k], sizeof(float)*_cols, sizeof(float), _rows, _cols, (float*)res._data[k], sizeof(float)*_rows, sizeof(float));
			
		}
		
		template<class T> T					Matrix<T>::Max(uint channel) const
		{
			T max;
			if( typeid(T) == typeid(uchar) )
				ippiMax_8u_C1R((unsigned char*)_data[channel], _dataStep, _roi, (unsigned char*)&max);
			else{
				ippiMax_32f_C1R((float*)_data[channel], _dataStep, _roi, (float*)&max);
			}
			
			return max;
		}
		
		template<class T> T					Matrix<T>::Min(uint channel)  const
		{
			T min;
			if( typeid(T) == typeid(uchar) )
				ippiMin_8u_C1R((unsigned char*)_data[channel], _dataStep, _roi, (unsigned char*)&min);
			else
				ippiMin_32f_C1R((float*)_data[channel], _dataStep, _roi, (float*)&min);
			
			return min;
		}
		
		
		
		
		template<class T> void				Matrix<T>::Max(T &val, uint &row, uint &col, uint channel) const
		{
			if( typeid(T) == typeid(uchar) )
				ippiMaxIndx_8u_C1R((unsigned char*)_data[channel], _dataStep, _roi, (unsigned char*)&val, (int*)&col, (int*)&row);
			else{
				ippiMaxIndx_32f_C1R((float*)_data[channel], _dataStep, _roi, (float*)&val, (int*)&col, (int*)&row);
			}
			
		}
		
		template<class T> void				Matrix<T>::Min(T &val, uint &row, uint &col, uint channel)  const
		{
			if( typeid(T) == typeid(uchar) )
				ippiMinIndx_8u_C1R((unsigned char*)_data[channel], _dataStep, _roi, (unsigned char*)&val, (int*)&col, (int*)&row);
			else
				ippiMinIndx_32f_C1R((float*)_data[channel], _dataStep, _roi, (float*)&val, (int*)&col, (int*)&row);
			
		}
		
		
		
		
		template<class T> float				Matrix<T>::Mean(uint channel)  const
		{
			double mean;
			if( typeid(T) == typeid(uchar) )
				ippiMean_8u_C1R((unsigned char*)_data[channel], _dataStep, _roi, &mean );
			else
				ippiMean_32f_C1R((float*)_data[channel], _dataStep, _roi, &mean, ippAlgHintFast);
			
			return (float)mean;
		}
		
		template<class T> float				Matrix<T>::Var(uint channel)  const
		{
			double mean,var;
			if( typeid(T) == typeid(uchar) )
				ippiMean_StdDev_8u_C1R((unsigned char*)_data[channel], _dataStep, _roi, &mean, &var);
			else
				ippiMean_StdDev_32f_C1R((float*)_data[channel], _dataStep, _roi, &mean, &var);
			
			return (float)(var*var);
		}
		
		template<class T> float				Matrix<T>::VarW(const Matrixf &w, T *mu)  const
		{
			T mm;
			if( mu == NULL )
				mm = (*this).MeanW(w);
			else
				mm = *mu;
			return ((*this)-mm).Sqr().MeanW(w);
		}
		
		template<class T> float				Matrix<T>::MeanW(const vectorf &w)  const
		{
			float mean=0.0f;
			assert(w.size() == this->size());
			for( uint k=0; k<w.size(); k++ )
				mean += w[k]*(*this)(k);
			
			return mean;
		}
		
		template<class T> float				Matrix<T>::MeanW(const Matrixf &w)  const
		{	
			return (float)((*this)&w).Sum();
		}
		template<class T> double			Matrix<T>::Sum(uint channel)  const
		{
			double sum;
			if( typeid(T) == typeid(uchar) )
				ippiSum_8u_C1R((unsigned char*)_data[channel], _dataStep, _roi, &sum);
			else
				ippiSum_32f_C1R((float*)_data[channel], _dataStep, _roi, &sum, ippAlgHintFast);
			
			return sum;
		}
		
		
		
		
		
		
		template<class T> Matrixu			Matrix<T>::convert2img(float min, float max)
		{
			if( max==min ){
				max = Max();
				min = Min();
			}
			
			Matrixu res(rows(),cols());
			// scale to 0 to 255
			Matrix<T> tmp;
			tmp = (*this);
			tmp = (tmp-(T)min)*(255/((T)max-(T)min));
			
			for( int d=0; d<depth(); d++ )
				for( int row=0; row<rows(); row++ )
					for( int col=0; col<cols(); col++ )
						res(row,col) = (uchar)tmp(row,col);
			
			return res;
			
		}
		template<class T> std::vector<Matrixu>	Matrix<T>::LoadVideo(const char *dirname, const char *basename, const char *ext, int start, int end, int digits, bool color)
		{
			std::vector<Matrixu> res(end-start+1);
			
			char format[1024];
			char fname[1024];
			for( int k=start; k<=end; k++ ){
				sprintf(format,"%s/%s%%0%ii.%s",dirname,basename,digits,ext);
				sprintf(fname,format,k);
				res[k-start].LoadImage(fname, color);
			}
			
			return res;
		}
		template<class T> std::vector<Matrixu>	Matrix<T>::LoadVideo(const char *fname, bool color, int maxframes)
		{
			CvCapture* capture = cvCaptureFromFile( fname );
			if( capture == NULL )
				abortError(__LINE__,__FILE__,"Error reading in video file");
			
			std::vector<Matrixu> vid;
			Matrixu tmp;
			bool c=true;
			while(c && vid.size()<(uint)maxframes){
				c=CaptureImage(capture, tmp, color);
				vid.push_back(tmp);
				fprintf(stderr,"%sLoading video: %d frames",ERASELINE,vid.size());
			}
			
			fprintf(stderr, "\n");
			return vid;
		}
		
		template<class T> Matrix<T>			Matrix<T>::vecMat2Mat(const std::vector<Matrix<T> > &x)
		{
			Matrix<T> t(x.size(),x[0].size());
			
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for( int k=0; k<(int)t.rows(); k++ )
				for( int j=0; j<t.cols(); j++ )
					t(k,j) = x[k](j);
			
			return t;
		}
		template<class T> std::vector<Matrix<T> >	Matrix<T>::vecMatTranspose(const std::vector<Matrix<T> > &x)
		{
			std::vector<Matrix<T> > t(x[0].size());
			
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for( int k=0; k<(int)t.size(); k++ )
				t[k].Resize(1,x.size());
			
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for( int k=0; k<(int)t.size(); k++ )
				for( uint j=0; j<x.size(); j++ )
					t[k](j) = x[j](k);
			
			
			return t;
		}
		template<class T> bool				Matrix<T>::DLMWrite( const char *fname, const char *delim )
		{
			remove( fname ); 
			std::ofstream strm; strm.open(fname, std::ios::out);
			if (strm.fail()) { abortError(  __LINE__, __FILE__,"unable to write" ); return false; }
			
			for( int r=0; r<rows(); r++ ) {
				for( int c=0; c<cols(); c++ ) {
					strm << (float)(*this)(r,c);
					if( c<(cols()-1)) strm << delim;
				}
				strm << std::endl;
			}
			
			strm.close();
			return true;
		}
		template<class T> bool				Matrix<T>::DLMRead( const char *fname, const char *delim )
		{
		  std::ifstream strm; strm.open(fname, std::ios::in);
			if( strm.fail() ) return false;
			char * tline = new char[40000000];
			
			// get number of cols
			strm.getline( tline, 40000000 );
			int ncols = ( strtok(tline," ,")==NULL ) ? 0 : 1;
			while( strtok(NULL," ,")!=NULL ) ncols++;
			
			// read in each row
			strm.seekg( 0, std::ios::beg ); 
			Matrix<T> *rowVec; std::vector<Matrix<T>*> allRowVecs;
			while(!strm.eof() && strm.peek()>=0) {
				strm.getline( tline, 40000000 );
				rowVec = new Matrix<T>(1,ncols);
				(*rowVec)(0,0) = (T) atof( strtok(tline,delim) );
				for( int col=1; col<ncols; col++ )
					(*rowVec)(0,col) = (T) atof( strtok(NULL,delim) );
				allRowVecs.push_back( rowVec );
			}
			int mrows = allRowVecs.size();
			
			// finally create matrix
			Resize(mrows,ncols);
			for( int row=0; row<mrows; row++ ) {
				rowVec = allRowVecs[row];
				for( int col=0; col<ncols; col++ )
					(*this)(row,col) = (*rowVec)(0,col);
				delete rowVec;
			}
			allRowVecs.clear();
			delete [] tline;
			strm.close();
			return true;
		}
		
		
		class Sample
    {
    public:
      Sample(const cv::Mat & img, const std::vector<cv::Mat_<float> > & ii_imgs, int row, int col, int width = 0,
             int height = 0, float weight = 1.0);
      Sample()
      {
        _row = _col = _height = _width = 0;
        _weight = 1.0f;
      }
			Sample&				operator= ( const Sample &a );
			
		public:
      cv::Mat _img;
      std::vector<cv::Mat_<float> > _ii_imgs;
			int					_row, _col, _width, _height;
			float				_weight;
			
		};
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		class SampleSet
		{
		public:
			SampleSet() {};
			SampleSet(const Sample &s) { _samples.push_back(s); };
			
			int					size() const { return _samples.size(); };
			void				push_back(const Sample &s) { _samples.push_back(s); };
      void
      push_back(const cv::Mat & img, const std::vector<cv::Mat_<float> > & ii_img, int x, int y, int width = 0,
                int height = 0, float weight = 1.0f);
			void				resize(int i) { _samples.resize(i); };
			void				resizeFtrs(int i);
			float &				getFtrVal(int sample,int ftr) { return _ftrVals[ftr](sample); };
			float				getFtrVal(int sample,int ftr) const { return _ftrVals[ftr](sample); };
			Sample &			operator[] (const int sample)  { return _samples[sample]; };
			Sample				operator[] (const int sample) const { return _samples[sample]; };
      const cv::Mat_<float> &
      ftrVals(int ftr) const
      {
        return _ftrVals[ftr];
      }
			bool				ftrsComputed() const { return !_ftrVals.empty() && !_samples.empty() && !_ftrVals[0].empty(); };
			void				clear() { _ftrVals.clear(); _samples.clear(); };
			
			
			// densely sample the image in a donut shaped region: will take points inside circle of radius inrad,
      // but outside of the circle of radius outrad.  when outrad=0 (default), then just samples points inside a circle
      void
      sampleImage(const cv::Mat & img, const std::vector<cv::Mat_<float> > & ii_imgs, int x, int y, int w, int h,
                  float inrad, float outrad = 0, int maxnum = 1000000);
      void
      sampleImage(const cv::Mat & img, const std::vector<cv::Mat_<float> > & ii_imgs, uint num, int w, int h);
			
			
			
		private:
			std::vector<Sample>		_samples;
      std::vector<cv::Mat_<float> > _ftrVals; // [ftr][sample]
    };
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		inline Sample&			Sample::operator= ( const Sample &a )
		{
			_img	= a._img;
			_row	= a._row;
			_col	= a._col;
			_width	= a._width;
			_height	= a._height;
			
			return (*this);
		}
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		inline void				SampleSet::resizeFtrs(int nftr)
		{
			_ftrVals.resize(nftr);
			int nsamp = _samples.size();
			
			if( nsamp>0 )
				for(int k=0; k<nftr; k++) 
					_ftrVals[k].create(1, nsamp);
		}

    inline void
    SampleSet::push_back(const cv::Mat & img, const std::vector<cv::Mat_<float> > & ii_imgs, int x, int y, int width,
                         int height, float weight)
    {
      Sample s(img, ii_imgs, y, x, width, height, weight);
			push_back(s); 
		}
		
		
		class Ftr;
		typedef std::vector<Ftr*> vecFtr;
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		class FtrParams
		{
		public:
			uint				_width, _height;
			
		public:
			virtual int			ftrType()=0;
		};
		
		class HaarFtrParams : public FtrParams
		{
		public:
			HaarFtrParams();
			uint				_maxNumRect, _minNumRect;
			int					_useChannels[1024];
			int					_numCh;
			
		public:
			virtual int			ftrType() { return 0; };
		};
		
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		class Ftr
		{
		public:
			uint					_width, _height;
			
			virtual float			compute( const Sample &sample ) const =0;
			virtual void			generate( FtrParams *params ) = 0;
			virtual Matrixu			toViz() {Matrixu empty; return empty;};
			virtual bool			update(const SampleSet &posx, const SampleSet &negx, const Matrixf &posw, const Matrixf &negw){return false;};
			
			
			static void				compute( SampleSet &samples, const vecFtr &ftrs );
			static void				compute( SampleSet &samples, Ftr *ftr, int ftrind );
			static vecFtr			generate( FtrParams *params, uint num );
			static void 			deleteFtrs( vecFtr ftrs );
			static void				toViz( vecFtr &ftrs, const char *dirname );
			
			virtual int				ftrType()=0;
		};
		
		class HaarFtr : public Ftr
		{
		public:
			uint					_channel;
			vectorf					_weights;
			std::vector<IppiRect>		_rects;
			vectorf					_rsums;
			double					_maxSum;
			static StopWatch		_sw;
			
		public:
			//HaarFtr( HaarFtrParams &params );
			HaarFtr();
			
			
			HaarFtr&				operator= ( const HaarFtr &a );
			
			float					expectedValue() const;
			
			virtual float			compute( const Sample &sample ) const;
			virtual void			generate( FtrParams *params );
			virtual Matrixu			toViz();
			virtual int				ftrType() { return 0; };
			
			
			
		};
		
		
		
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		inline float				HaarFtr::compute( const Sample &sample ) const
		{
      if (sample._ii_imgs.empty())
        abortError(__LINE__, __FILE__, "Integral image not initialized before called compute()");
			IppiRect r;
			float sum = 0.0f;
			
			for( int k=0; k<(int)_rects.size(); k++ )
			{
				r = _rects[k];
				r.x += sample._col; r.y += sample._row;
        sum +=
            _weights[k] * (sample._ii_imgs[_channel](r.y + r.height, r.x + r.width)
                + sample._ii_imgs[_channel](r.y, r.x)
                           - sample._ii_imgs[_channel](r.y + r.height, r.x)
                           - sample._ii_imgs[_channel](r.y, r.x + r.width)); ///_rsums[k];
      }
			
			r.x = sample._col;
			r.y = sample._row;
			r.width = (int)sample._weight;
			r.height = (int)sample._height;
			
			return (float)(sum);
			//return (float) (100*sum/sample._img->sumRect(r,_channel));
		}
		
		
		inline HaarFtr&				HaarFtr::operator= ( const HaarFtr &a )
		{
			_width		= a._width;
			_height		= a._height;
			_channel	= a._channel;
			_weights	= a._weights;
			_rects		= a._rects;
			_maxSum		= a._maxSum;
			
			return (*this);
		}
		
		inline float				HaarFtr::expectedValue() const
		{
			float sum=0.0f;
			for( int k=0; k<(int)_rects.size(); k++ ){
				sum += _weights[k]*_rects[k].height*_rects[k].width*125;
			}
			return sum;
		}
		
		
		class ClfWeak;
		class ClfStrong;
		class ClfAdaBoost;
		class ClfMilBoost;
		
		
		
		
		class ClfStrongParams
		{
		public:
			ClfStrongParams(){_weakLearner = "stump"; _lRate=0.85f; _storeFtrHistory=false;};
			virtual int			clfType()=0; // [0] Online AdaBoost (Oza/Grabner) [1] Online StochBoost_LR [2] Online StochBoost_MIL
		public:
			FtrParams			*_ftrParams;
			string				_weakLearner; // "stump" or "wstump"; current code only uses "stump"
			float				_lRate; // learning rate for weak learners;
			bool				_storeFtrHistory;
		};
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		class ClfStrong
		{
		public:
			ClfStrongParams		*_params;
			vecFtr				_ftrs;
			vecFtr				_selectedFtrs;
			StopWatch			_clfsw;
			Matrixf				_ftrHist;
			uint				_counter;
			
		public:
			int					nFtrs() {return _ftrs.size();};
			
			// abstract functions
			virtual void		init(ClfStrongParams *params)=0;
			virtual void		update(SampleSet &posx, SampleSet &negx)=0;
			virtual vectorf		classify(SampleSet &x, bool logR=true)=0;
			
			static ClfStrong*	makeClf(ClfStrongParams *clfparams);
			static Matrixf		applyToImage(ClfStrong *clf, const cv::Mat & img, bool logR=true); // returns a probability map (or log odds ratio map if logR=true)
			
			static void			eval(vectorf ppos, vectorf pneg, float &err, float &fp, float &fn, float thresh=0.5f);
			static float		likl(vectorf ppos, vectorf pneg);
		};
		
		
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		// IMEPLEMENTATIONS - PARAMS
		
		class ClfAdaBoostParams : public ClfStrongParams
		{
		public:
			int					_numSel, _numFeat;
			
		public:
			ClfAdaBoostParams(){_numSel=50;_numFeat=250;};
			virtual int			clfType() { return 0; };
		};
		
		
		class ClfMilBoostParams : public ClfStrongParams
		{
		public:
			int					_numFeat,_numSel;
			
		public:
			ClfMilBoostParams(){_numSel=50;_numFeat=250;};
			virtual int			clfType() { return 1; };
		};
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		// IMEPLEMENTATIONS - CLF
		
		class ClfAdaBoost : public ClfStrong
		{
		private:
			vectorf				_alphas;
			vectori				_selectors;
			std::vector<ClfWeak*>	_weakclf;
			uint				_numsamples;
			float				_sumAlph;
			std::vector<vectorf>		_countFPv, _countFNv, _countTPv, _countTNv; //[selector][feature]
			ClfAdaBoostParams	*_myParams;
		public:
			ClfAdaBoost(){};
			virtual void		init(ClfStrongParams *params);
			virtual void		update(SampleSet &posx, SampleSet &negx);
			virtual vectorf		classify(SampleSet &x, bool logR=true);
		};
		
		
		class ClfMilBoost : public ClfStrong
		{
		private:
			vectori				_selectors;
			std::vector<ClfWeak*>	_weakclf;
			uint				_numsamples;
			ClfMilBoostParams	*_myParams;
			
		public:
			ClfMilBoost(){};
			virtual void		init(ClfStrongParams *params);
			virtual void		update(SampleSet &posx, SampleSet &negx);
			virtual vectorf		classify(SampleSet &x, bool logR=true);
			
		};
		
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		// WEAK CLF
		
		class ClfWeak
		{
		public:
			ClfWeak();
			ClfWeak(int id);
			
			virtual void		init()=0;
      virtual void
      update(SampleSet &posx, SampleSet &negx, const cv::Mat_<float> & posw = cv::Mat_<float>(),
             const cv::Mat_<float> & negw = cv::Mat_<float>())=0;
			virtual bool		classify(SampleSet &x, int i)=0;
			virtual float		classifyF(SampleSet &x, int i)=0;
			virtual void		copy(const ClfWeak* c)=0;
			
			virtual vectorb		classifySet(SampleSet &x);
			virtual vectorf		classifySetF(SampleSet &x);
			
			float				ftrcompute(const Sample &x) {return _ftr->compute(x);};
			float				getFtrVal(const SampleSet &x,int i) { return (x.ftrsComputed()) ? x.getFtrVal(i,_ind) : _ftr->compute(x[i]); };
			
		protected:
			bool				_trained;
			Ftr					*_ftr;
			vecFtr				*_ftrs;
			int					_ind;
			float				_lRate;
			ClfStrong			*_parent;
			
			friend class ClfAdaBoost;
			friend class ClfMilBoost;
		};
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		class ClfOnlineStump : public ClfWeak
		{
		public:
			//////////////////////////////////////////////////////////////////////////////////////////////////////
			// members
			float				_mu0, _mu1, _sig0, _sig1;
			float				_q;
			int					_s;
			float				_n1, _n0;
			float				_e1, _e0;
		public:
			//////////////////////////////////////////////////////////////////////////////////////////////////////
			// functions
			ClfOnlineStump() : ClfWeak() {init();};
			ClfOnlineStump(int ind) : ClfWeak(ind) {init();};
			virtual void		init();
      virtual void
      update(SampleSet &posx, SampleSet &negx, const cv::Mat_<float> & posw, const cv::Mat_<float> & negw);
			virtual bool		classify(SampleSet &x, int i);
			virtual float		classifyF(SampleSet &x, int i);
			virtual void		copy(const ClfWeak* c);
			
			
		};
		
		class ClfWStump : public ClfWeak
		{
		public:
			//////////////////////////////////////////////////////////////////////////////////////////////////////
			// members
			float				_mu0, _mu1, _sig0, _sig1;
			float				_q;
			int					_s;
			float				_n1, _n0;
			float				_e1, _e0;
		public:
			//////////////////////////////////////////////////////////////////////////////////////////////////////
			// functions
			ClfWStump() : ClfWeak() {init();};
			ClfWStump(int ind) : ClfWeak(ind) {init();};
			virtual void		init();
      virtual void
      update(SampleSet &posx, SampleSet &negx, const cv::Mat_<float> & posw, const cv::Mat_<float> & negw);
			virtual bool		classify(SampleSet &x, int i){return classifyF(x,i)>0;};
			virtual float		classifyF(SampleSet &x, int i);
			virtual void		copy(const ClfWeak* c);
		};
		
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		inline vectorb			ClfWeak::classifySet(SampleSet &x)
		{
			vectorb res(x.size());
			
			for( int k=0; k<(int)res.size(); k++ ){
				res[k] = classify(x,k);
			}
			return res;
		}
		inline vectorf			ClfWeak::classifySetF(SampleSet &x)
		{
			vectorf res(x.size());
			
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for( int k=0; k<(int)res.size(); k++ ){
				res[k] = classifyF(x,k);
			}
			return res;
		}
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
    inline void
    ClfOnlineStump::update(SampleSet &posx, SampleSet &negx, const cv::Mat_<float> & posw, const cv::Mat_<float> & negw)
    {
			float posmu=0.0,negmu=0.0;
      if (posx.size() > 0)
        posmu = cv::mean(posx.ftrVals(_ind))[0];
      if (negx.size() > 0)
        negmu = cv::mean(negx.ftrVals(_ind))[0];
			
			if( _trained ){
				if( posx.size()>0 ){
					_mu1	= ( _lRate*_mu1  + (1-_lRate)*posmu );
          cv::Mat diff = posx.ftrVals(_ind) - _mu1;
          _sig1 = _lRate * _sig1 + (1 - _lRate) * cv::mean(diff.mul(diff))[0];
				}
				if( negx.size()>0 ){
					_mu0	= ( _lRate*_mu0  + (1-_lRate)*negmu );
          cv::Mat diff = negx.ftrVals(_ind) - _mu0;
          _sig0 = _lRate * _sig0 + (1 - _lRate) * cv::mean(diff.mul(diff))[0];
				}
				
				_q = (_mu1-_mu0)/2;
				_s = sign(_mu1-_mu0);
				_n0 = 1.0f/pow(_sig0,0.5f);
				_n1 = 1.0f/pow(_sig1,0.5f);
				//_e1 = -1.0f/(2.0f*_sig1+1e-99f);
				//_e0 = -1.0f/(2.0f*_sig0+1e-99f);
				_e1 = -1.0f/(2.0f*_sig1+FLT_MIN);
				_e0 = -1.0f/(2.0f*_sig0+FLT_MIN);

			}
			else{
				_trained = true;
				if( posx.size()>0 ){
					_mu1 = posmu;
          cv::Scalar scal_mean, scal_std_dev;
          cv::meanStdDev(posx.ftrVals(_ind), scal_mean, scal_std_dev);
          _sig1 = scal_std_dev[0] * scal_std_dev[0] + 1e-9f;
				}
				
				if( negx.size()>0 ){
					_mu0 = negmu;
          cv::Scalar scal_mean, scal_std_dev;
          cv::meanStdDev(negx.ftrVals(_ind), scal_mean, scal_std_dev);
          _sig0 = scal_std_dev[0] * scal_std_dev[0] + 1e-9f;
				}
				
				_q = (_mu1-_mu0)/2;
				_s = sign(_mu1-_mu0);
				_n0 = 1.0f/pow(_sig0,0.5f);
				_n1 = 1.0f/pow(_sig1,0.5f);
				//_e1 = -1.0f/(2.0f*_sig1+1e-99f);
				//_e0 = -1.0f/(2.0f*_sig0+1e-99f);				
				_e1 = -1.0f/(2.0f*_sig1+FLT_MIN);
				_e0 = -1.0f/(2.0f*_sig0+FLT_MIN);
			}
		}
		
		inline bool				ClfOnlineStump::classify(SampleSet &x, int i)
		{
			float xx = getFtrVal(x,i);
			double p0 = exp( (xx-_mu0)*(xx-_mu0)*_e0 )*_n0;
			double p1 = exp( (xx-_mu1)*(xx-_mu1)*_e1 )*_n1;
			bool r = p1>p0;
			return r;
			
			//return (_s*sign(x-_q))>0? 1 : 0 ;
		}
		inline float			ClfOnlineStump::classifyF(SampleSet &x, int i)
		{
			float xx = getFtrVal(x,i);
			double p0 = exp( (xx-_mu0)*(xx-_mu0)*_e0 )*_n0;
			double p1 = exp( (xx-_mu1)*(xx-_mu1)*_e1 )*_n1;
			float r = (float)(log(1e-5+p1)-log(1e-5+p0));
			//r = (float)p1>p0;
			return r;
			
			//return (_s*sign(x-_q))>0? 1 : 0 ;
		}
		inline void				ClfOnlineStump::copy(const ClfWeak* c)
		{
			ClfOnlineStump *cc = (ClfOnlineStump*)c;
			_mu0	= cc->_mu0;
			_mu1	= cc->_mu1;
			_sig0	= cc->_sig0;
			_sig1	= cc->_sig1;
			_lRate	= cc->_lRate;
			_e0		= cc->_e0;
			_e1		= cc->_e1;
			_n0		= cc->_n0;
			_n1		= cc->_n1;
			
			return;
		}
		
		
		////////////////////////////////////////////////////////////////////////////////////////////////////////
    inline void
    ClfWStump::update(SampleSet &posx, SampleSet &negx, const cv::Mat_<float> & posw, const cv::Mat_<float> & negw)
		{
			cv::Mat_<float> poswn, negwn;
			if( (posx.size() != posw.size().area()) || (negx.size() != negw.size().area()) )
				abortError(__LINE__,__FILE__,"ClfWStump::update - number of samples and number of weights mismatch");
			
			float posmu=0.0, negmu=0.0;
			if( posx.size()>0 ) {
        poswn = posw / (cv::sum(posw)[0] + 1e-6);
        posmu = cv::mean(posx.ftrVals(_ind).mul(poswn))[0];
			}
			if( negx.size()>0 ) {
        negwn = negw / (cv::sum(negw)[0] + 1e-6);
        negmu = cv::mean(negx.ftrVals(_ind).mul(negwn))[0];
			}
			
			if( _trained ){
				if( posx.size()>0 ){
					_mu1	= ( _lRate*_mu1  + (1-_lRate)*posmu );
          cv::Scalar scal_mean, scal_std_dev;
          cv::meanStdDev(posx.ftrVals(_ind).mul(poswn), scal_mean, scal_std_dev);
          _sig1 = _lRate * _sig1 + (1 - _lRate) * scal_std_dev[0] * scal_std_dev[0];
				}
				if( negx.size()>0 ){
					_mu0	= ( _lRate*_mu0  + (1-_lRate)*negmu );
          cv::Scalar scal_mean, scal_std_dev;
          cv::meanStdDev(negx.ftrVals(_ind).mul(negwn), scal_mean, scal_std_dev);
          _sig0 = _lRate * _sig0 + (1 - _lRate) * scal_std_dev[0] * scal_std_dev[0];
				}
			}
			else{
				_trained = true;
				_mu1 = posmu;
				_mu0 = negmu;
        cv::Scalar scal_mean, scal_std_dev;
        if (negx.size() > 0)
        {
          cv::meanStdDev(negx.ftrVals(_ind).mul(negwn), scal_mean, scal_std_dev);
          _sig0 = scal_std_dev[0] * scal_std_dev[0] + 1e-9f;
        }
        if (posx.size() > 0)
        {
          cv::meanStdDev(posx.ftrVals(_ind).mul(poswn), scal_mean, scal_std_dev);
          _sig1 = scal_std_dev[0] * scal_std_dev[0] + 1e-9f;
        }
      }
			
			_n0 = 1.0f/pow(_sig0,0.5f);
			_n1 = 1.0f/pow(_sig1,0.5f);
			_e1 = -1.0f/(2.0f*_sig1);
			_e0 = -1.0f/(2.0f*_sig0);
		}
		
		inline float			ClfWStump::classifyF(SampleSet &x, int i)
		{
			float xx = getFtrVal(x,i);
			double p0 = exp( (xx-_mu0)*(xx-_mu0)*_e0 )*_n0;
			double p1 = exp( (xx-_mu1)*(xx-_mu1)*_e1 )*_n1;
			float r = (float)(log(1e-5+p1)-log(1e-5+p0));
			//r = (float)(r>0);
			return r;
		}
		inline void				ClfWStump::copy(const ClfWeak* c)
		{
			ClfWStump *cc = (ClfWStump*)c;
			_mu0	= cc->_mu0;
			_mu1	= cc->_mu1;
			
			return;
		}
		
		
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		inline vectorf			ClfAdaBoost::classify(SampleSet &x, bool logR)
		{
			int numsamples = x.size();
			vectorf res(numsamples);
			vectorb tr;
			
			// for each selector, accumate in the res vector
			for( int sel=0; sel<(int)_selectors.size(); sel++ ){
				tr = _weakclf[_selectors[sel]]->classifySet(x);
#ifdef _OPENMP
#pragma omp parallel for
#endif
				for( int j=0; j<numsamples; j++ ){
					res[j] += tr[j] ?  _alphas[sel] : -_alphas[sel];
				}
				
			}
			
			// return probabilities or log odds ratio
			if( !logR ){
#ifdef _OPENMP
#pragma omp parallel for
#endif
				for( int j=0; j<(int)res.size(); j++ ){
					res[j] = sigmoid(2*res[j]);
				}
			}
			
			return res;
		}
		
		
		inline vectorf			ClfMilBoost::classify(SampleSet &x, bool logR)
		{
			int numsamples = x.size();
			vectorf res(numsamples);
			vectorf tr;
			
			for( uint w=0; w<_selectors.size(); w++ ){
				tr = _weakclf[_selectors[w]]->classifySetF(x);
#ifdef _OPENMP
#pragma omp parallel for
#endif
				for( int j=0; j<numsamples; j++ ){
					res[j] += tr[j];
				}
			}
			
			// return probabilities or log odds ratio
			if( !logR ){
#ifdef _OPENMP
#pragma omp parallel for
#endif
				for( int j=0; j<(int)res.size(); j++ ){
					res[j] = sigmoid(res[j]);
				}
			}
			
			return res;
		}
		
		
		
		
		
		////////////////////////////////////////////////////////////////////////////////////////////////////////
		inline void				ClfStrong::eval(vectorf ppos, vectorf pneg, float &err, float &fp, float &fn, float thresh)
		{
			fp=0; fn=0;
			for( uint k=0; k<ppos.size(); k++ )
				(ppos[k] < thresh) ? fn++ : fn;
			
			for( uint k=0; k<pneg.size(); k++ )
				(pneg[k] >= thresh) ? fp++ : fp;
			
			fn /= ppos.size();
			fp /= pneg.size();
			
			err = 0.5f*fp + 0.5f*fn;
		}
		inline float			ClfStrong::likl(vectorf ppos, vectorf pneg)
		{
			float likl=0, posw = 1.0f/ppos.size(), negw = 1.0f/pneg.size();
			
			for( uint k=0; k<ppos.size(); k++ )
				likl += log(ppos[k]+1e-5f)*posw;
			
			for( uint k=0; k<pneg.size(); k++ )
				likl += log(1-pneg[k]+1e-5f)*negw;
			
			return likl;
		}
		
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		class TrackerParams
		{
		public:
			TrackerParams();
			
			vectori			_boxcolor;						// for outputting video
			uint			_lineWidth;						// line width 
			uint			_negnumtrain,_init_negnumtrain; // # negative samples to use during training, and init
			float			_posradtrain,_init_postrainrad; // radius for gathering positive instances
			uint			_posmaxtrain;					// max # of pos to train with
			bool			_debugv;						// displays response map during tracking [kinda slow, but help in debugging]
			vectorf			_initstate;						// [x,y,scale,orientation] - note, scale and orientation currently not used
			bool			_useLogR;						// use log ratio instead of probabilities (tends to work much better)
			bool			_initWithFace;					// initialize with the OpenCV tracker rather than _initstate
			bool			_disp;							// display video with tracker state (colored box)
			
			string			_vidsave;						// filename - save video with tracking box
			string			_trsave;						// filename - save file containing the coordinates of the box (txt file with [x y width height] per row)
			
		};
		
		
		
		class SimpleTrackerParams : public TrackerParams
		{
		public:
			SimpleTrackerParams();
			
			uint			_srchwinsz;						// size of search window
			uint			_negsamplestrat;				// [0] all over image [1 - default] close to the search window
		};
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		class Tracker
		{
		public:
			
			static bool		initFace(TrackerParams* params, Matrixu &frame);
			static void		replayTracker(vector<Matrixu> &vid, const string states, string outputvid="",uint R=255, uint G=0, uint B=0);
			static void		replayTrackers(vector<Matrixu> &vid, vector<string> statesfile, string outputvid, Matrixu colors);
			
		protected:
			static cv::CascadeClassifier facecascade;
		};
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		class SimpleTracker : public Tracker
		{
		public:
			SimpleTracker(){};
			~SimpleTracker(){ if( _clf!=NULL ) delete _clf; };
      double
      track_frame(const cv::Mat & frame); // track object in a frame;  requires init() to have been called.
      bool
      init(const cv::Mat & frame, SimpleTrackerParams p, ClfStrongParams *clfparams);
			Matrixf &		getFtrHist() { return _clf->_ftrHist; }; // only works if _clf->_storeFtrHistory is set to true.. mostly for debugging
			
			inline void getTrackBox(cv::Rect & roi)
			{
				roi.width = cvRound(_curState[2]);
				roi.height = cvRound(_curState[3]);
				roi.x = cvRound(_curState[0]);
				roi.y = cvRound(_curState[1]);
			}
			
			
		private:
			ClfStrong			*_clf;
			vectorf				_curState;
			SimpleTrackerParams	_trparams;
			ClfStrongParams		*_clfparams;
			int					_cnt;
		};
		
	}  // namespace mil
}  // namespace cv


#endif  // #ifndef __OPENCV_ONLINE_MIL_H__

/* End of file. */
