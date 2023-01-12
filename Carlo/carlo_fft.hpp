#pragma once

namespace Casino::IPP
{

    
    template<typename T>
    void FFTInitR(void ** ppFFTSpec, int order, int flag, IppHintAlgorithm hint, Ipp8u * pSpec, Ipp8u* pSpecBuffer)
    {
        assert(1==0);
    }
    template<>
    void FFTInitR<Ipp32f>(void ** ppFFTSpec, int order, int flag, IppHintAlgorithm hint, Ipp8u * pSpec, Ipp8u* pSpecBuffer)
    {
        IppStatus status = ippsFFTInit_R_32f((IppsFFTSpec_R_32f**)ppFFTSpec,order,flag,hint,pSpec,pSpecBuffer);
        checkStatus(status);
    }
    template<>
    void FFTInitR<Ipp64f>(void ** ppFFTSpec, int order, int flag, IppHintAlgorithm hint, Ipp8u * pSpec, Ipp8u* pSpecBuffer)
    {
        IppStatus status = ippsFFTInit_R_64f((IppsFFTSpec_R_64f**)ppFFTSpec,order,flag,hint,pSpec,pSpecBuffer);
        checkStatus(status);
    }

    template<typename T>
    void FFTInitC(void ** ppFFTSpec, int order, int flag, IppHintAlgorithm hint, Ipp8u * pSpec, Ipp8u* pSpecBuffer)
    {
        assert(1==0);
    }
    
    template<>
    void FFTInitC<Ipp32f>(void ** ppFFTSpec, int order, int flag, IppHintAlgorithm hint, Ipp8u * pSpec, Ipp8u* pSpecBuffer)
    {
        IppStatus status = ippsFFTInit_C_32f((IppsFFTSpec_C_32f**)ppFFTSpec,order,flag,hint,pSpec,pSpecBuffer);
        checkStatus(status);
    }
    template<>
    void FFTInitC<Ipp64f>(void ** ppFFTSpec, int order, int flag, IppHintAlgorithm hint, Ipp8u * pSpec, Ipp8u* pSpecBuffer)
    {
        IppStatus status = ippsFFTInit_C_64f((IppsFFTSpec_C_64f**)ppFFTSpec,order,flag,hint,pSpec,pSpecBuffer);
        checkStatus(status);
    }
    
    template<>
    void FFTInitC<Ipp32fc>(void ** ppFFTSpec, int order, int flag, IppHintAlgorithm hint, Ipp8u * pSpec, Ipp8u* pSpecBuffer)
    {
        IppStatus status = ippsFFTInit_C_32fc((IppsFFTSpec_C_32fc**)ppFFTSpec,order,flag,hint,pSpec,pSpecBuffer);
        checkStatus(status);
    }
    template<>
    void FFTInitC<Ipp64fc>(void ** ppFFTSpec, int order, int flag, IppHintAlgorithm hint, Ipp8u * pSpec, Ipp8u* pSpecBuffer)
    {
        IppStatus status = ippsFFTInit_C_64fc((IppsFFTSpec_C_64fc**)ppFFTSpec,order,flag,hint,pSpec,pSpecBuffer);
        checkStatus(status);
    }

    template<typename T>
    void FFTFwd_RToPack(const T * pSrc, T * pDst, void * pFFTSpec, Ipp8u * pBuffer)
    {
        assert(1==0);
    }
    template<>
    void FFTFwd_RToPack<Ipp32f>(const Ipp32f * pSrc, Ipp32f * pDst, void * pFFTSpec, Ipp8u * pBuffer)
    {
        IppStatus status = ippsFFTFwd_RToPack_32f(pSrc,pDst,(IppsFFTSpec_R_32f*)pFFTSpec,pBuffer);
        checkStatus(status);
    }
    template<>
    void FFTFwd_RToPack<Ipp64f>(const Ipp64f * pSrc, Ipp64f * pDst, void * pFFTSpec, Ipp8u * pBuffer)
    {
        IppStatus status = ippsFFTFwd_RToPack_64f(pSrc,pDst,(IppsFFTSpec_R_64f*)pFFTSpec,pBuffer);
        checkStatus(status);
    }


    template<typename T>
    void FFTFwd_RToPerm(const T * pSrc, T * pDst, void * pFFTSpec, Ipp8u * pBuffer)
    {
        assert(1==0);
    }
    template<>
    void FFTFwd_RToPerm<Ipp32f>(const Ipp32f * pSrc, Ipp32f * pDst, void * pFFTSpec, Ipp8u * pBuffer)
    {
        IppStatus status = ippsFFTFwd_RToPerm_32f(pSrc,pDst,(IppsFFTSpec_R_32f*)pFFTSpec,pBuffer);
        checkStatus(status);
    }
    template<>
    void FFTFwd_RToPerm<Ipp64f>(const Ipp64f * pSrc, Ipp64f * pDst, void * pFFTSpec, Ipp8u * pBuffer)
    {
        IppStatus status = ippsFFTFwd_RToPerm_64f(pSrc,pDst,(IppsFFTSpec_R_64f*)pFFTSpec,pBuffer);
        checkStatus(status);
    }

    template<typename T>
    void FFTFwd_RToCCS(const T * pSrc, T * pDst, void * pFFTSpec, Ipp8u * pBuffer)
    {
        assert(1==0);
    }
    template<>
    void FFTFwd_RToCCS<Ipp32f>(const Ipp32f * pSrc, Ipp32f * pDst, void * pFFTSpec, Ipp8u * pBuffer)
    {
        IppStatus status = ippsFFTFwd_RToCCS_32f(pSrc,pDst,(IppsFFTSpec_R_32f*)pFFTSpec,pBuffer);
        checkStatus(status);
    }
    template<>
    void FFTFwd_RToCCS<Ipp64f>(const Ipp64f * pSrc, Ipp64f * pDst, void * pFFTSpec, Ipp8u * pBuffer)
    {
        IppStatus status = ippsFFTFwd_RToCCS_64f(pSrc,pDst,(IppsFFTSpec_R_64f*)pFFTSpec,pBuffer);
        checkStatus(status);
    }
    
    
    template<typename T>
    void FFTInv_PackToR(const T * pSrc, T * pDst, void * pFFTSpec, Ipp8u * pBuffer)
    {
        assert(1==0);
    }
    
    template<>
    void FFTInv_PackToR<Ipp32f>(const Ipp32f * pSrc, Ipp32f * pDst, void * pFFTSpec, Ipp8u * pBuffer)
    {
        IppStatus status = ippsFFTInv_PackToR_32f(pSrc,pDst,(IppsFFTSpec_R_32f*)pFFTSpec,pBuffer);
        checkStatus(status);
    }
    template<>
    void FFTInv_PackToR<Ipp64f>(const Ipp64f * pSrc, Ipp64f * pDst, void * pFFTSpec, Ipp8u * pBuffer)
    {
        IppStatus status = ippsFFTInv_PackToR_64f(pSrc,pDst,(IppsFFTSpec_R_64f*)pFFTSpec,pBuffer);
        checkStatus(status);
    }

    template<typename T>
    void FFTInv_PermToR(const T * pSrc, T * pDst, void * pFFTSpec, Ipp8u * pBuffer)
    {
        assert(1==0);
    }
    template<>
    void FFTInv_PermToR<Ipp32f>(const Ipp32f * pSrc, Ipp32f * pDst, void * pFFTSpec, Ipp8u * pBuffer)
    {
        IppStatus status = ippsFFTInv_PermToR_32f(pSrc,pDst,(IppsFFTSpec_R_32f*)pFFTSpec,pBuffer);
        checkStatus(status);
    }
    template<>
    void FFTInv_PermToR<Ipp64f>(const Ipp64f * pSrc, Ipp64f * pDst, void * pFFTSpec, Ipp8u * pBuffer)
    {
        IppStatus status = ippsFFTInv_PermToR_64f(pSrc,pDst,(IppsFFTSpec_R_64f*)pFFTSpec,pBuffer);
        checkStatus(status);
    }

    template<typename T>
    void FFTInv_CCSToR(const T * pSrc, T * pDst, void * pFFTSpec, Ipp8u * pBuffer)
    {
        assert(1==0);
    }
    template<>
    void FFTInv_CCSToR<Ipp32f>(const Ipp32f * pSrc, Ipp32f * pDst, void * pFFTSpec, Ipp8u * pBuffer)
    {
        IppStatus status = ippsFFTInv_CCSToR_32f(pSrc,pDst,(IppsFFTSpec_R_32f*)pFFTSpec,pBuffer);
        checkStatus(status);
    }
    template<>
    void FFTInv_CCSToR<Ipp64f>(const Ipp64f * pSrc, Ipp64f * pDst, void * pFFTSpec, Ipp8u * pBuffer)
    {
        IppStatus status = ippsFFTInv_CCSToR_64f(pSrc,pDst,(IppsFFTSpec_R_64f*)pFFTSpec,pBuffer);
        checkStatus(status);
    }
    template<typename T>
    void FFTGetSizeR(int order, int flag, IppHintAlgorithm hint, int * pSpecSize, int * pSpecBufferSize, int * pBufferSize)
    {
        assert(1==0);
    }
    template<>
    void FFTGetSizeR<Ipp32f>(int order, int flag, IppHintAlgorithm hint, int * pSpecSize, int * pSpecBufferSize, int * pBufferSize)
    {
        IppStatus status = ippsFFTGetSize_R_32f(order,flag,hint,pSpecSize,pSpecBufferSize,pBufferSize);
        checkStatus(status);
    }
    template<>
    void FFTGetSizeR<Ipp64f>(int order, int flag, IppHintAlgorithm hint, int * pSpecSize, int * pSpecBufferSize, int * pBufferSize)
    {
        IppStatus status = ippsFFTGetSize_R_64f(order,flag,hint,pSpecSize,pSpecBufferSize,pBufferSize);
        checkStatus(status);
    }

    template<typename T>
    void FFTGetSizeC(int order, int flag, IppHintAlgorithm hint, int * pSpecSize, int * pSpecBufferSize, int * pBufferSize)
    {
        assert(1==0);
    }
    template<>
    void FFTGetSizeC<Ipp32f>(int order, int flag, IppHintAlgorithm hint, int * pSpecSize, int * pSpecBufferSize, int * pBufferSize)
    {
        IppStatus status = ippsFFTGetSize_C_32f(order,flag,hint,pSpecSize,pSpecBufferSize,pBufferSize);
        checkStatus(status);
    }
    template<>
    void FFTGetSizeC<Ipp64f>(int order, int flag, IppHintAlgorithm hint, int * pSpecSize, int * pSpecBufferSize, int * pBufferSize)
    {
        IppStatus status = ippsFFTGetSize_C_64f(order,flag,hint,pSpecSize,pSpecBufferSize,pBufferSize);
        checkStatus(status);
    }
    template<>
    void FFTGetSizeC<Ipp32fc>(int order, int flag, IppHintAlgorithm hint, int * pSpecSize, int * pSpecBufferSize, int * pBufferSize)
    {
        IppStatus status = ippsFFTGetSize_C_32fc(order,flag,hint,pSpecSize,pSpecBufferSize,pBufferSize);
        checkStatus(status);
    }
    template<>
    void FFTGetSizeC<Ipp64fc>(int order, int flag, IppHintAlgorithm hint, int * pSpecSize, int * pSpecBufferSize, int * pBufferSize)
    {
        IppStatus status = ippsFFTGetSize_C_64fc(order,flag,hint,pSpecSize,pSpecBufferSize,pBufferSize);
        checkStatus(status);
    }

    template<typename T>
    void FFTFwd_C2C(const T * pSrcRe, const T * pSrcIm,  T* pDstRe, T* pDstIm, void * pFFTSpec, Ipp8u* pBuffer)
    {
        // sometime later will try to put some more useful exceptions here
        // it should never be called but I dont think there is a virtual function without a class
        assert(1==0);
    }
    
    template<>
    void FFTFwd_C2C<Ipp32f>(const Ipp32f * pSrcRe, const Ipp32f * pSrcIm, Ipp32f* pDstRe, Ipp32f* pDstIm, void * pFFTSpec, Ipp8u* pBuffer)
    {
        IppStatus status = ippsFFTFwd_CToC_32f(pSrcRe,pSrcIm,pDstRe,pDstIm,(IppsFFTSpec_C_32f*)pFFTSpec,pBuffer);        
        checkStatus(status);
    }
    template<>
    void FFTFwd_C2C<Ipp64f>(const Ipp64f * pSrcRe, const Ipp64f * pSrcIm, Ipp64f* pDstRe, Ipp64f* pDstIm, void * pFFTSpec, Ipp8u* pBuffer)
    {
        IppStatus status = ippsFFTFwd_CToC_64f(pSrcRe,pSrcIm,pDstRe,pDstIm,(IppsFFTSpec_C_64f*)pFFTSpec,pBuffer);        
        checkStatus(status);
    }

    template<typename T>
    void FFTFwd_C2C(const T * pSrc, T* pDst, void * pFFTSpec, Ipp8u* pBuffer)
    {
        assert(1==0);
    }

    template<>
    void FFTFwd_C2C<Ipp32fc>(const Ipp32fc * pSrc,Ipp32fc* pDst, void * pFFTSpec, Ipp8u* pBuffer)
    {
        IppStatus status = ippsFFTFwd_CToC_32fc(pSrc,pDst,(IppsFFTSpec_C_32fc*)pFFTSpec,pBuffer);        
        checkStatus(status);
    }
    template<>
    void FFTFwd_C2C<Ipp64fc>(const Ipp64fc * pSrc,Ipp64fc* pDst, void * pFFTSpec, Ipp8u* pBuffer)
    {
        IppStatus status = ippsFFTFwd_CToC_64fc(pSrc,pDst,(IppsFFTSpec_C_64fc*)pFFTSpec,pBuffer);        
        checkStatus(status);
    }

    template<typename T>
    void FFTInv_C2C(const T * pSrcRe, const T * pSrcIm,  T* pDstRe, T* pDstIm, void * pFFTSpec, Ipp8u* pBuffer)
    {
        // sometime later will try to put some more useful exceptions here
        // it should never be called but I dont think there is a virtual function without a class
        assert(1==0);
    }
    template<>
    void FFTInv_C2C<Ipp32f>(const Ipp32f * pSrcRe, const Ipp32f * pSrcIm, Ipp32f* pDstRe, Ipp32f* pDstIm, void * pFFTSpec, Ipp8u* pBuffer)
    {
        IppStatus status = ippsFFTInv_CToC_32f(pSrcRe,pSrcIm,pDstRe,pDstIm,(IppsFFTSpec_C_32f*)pFFTSpec,pBuffer);        
        checkStatus(status);
    }

    template<>
    void FFTInv_C2C<Ipp64f>(const Ipp64f * pSrcRe, const Ipp64f * pSrcIm, Ipp64f* pDstRe, Ipp64f* pDstIm, void * pFFTSpec, Ipp8u* pBuffer)
    {
        IppStatus status = ippsFFTInv_CToC_64f(pSrcRe,pSrcIm,pDstRe,pDstIm,(IppsFFTSpec_C_64f*)pFFTSpec,pBuffer);        
        checkStatus(status);
    }

    template<typename T>
    void FFTInv_C2C(const T * pSrc, T* pDst, void * pFFTSpec, Ipp8u* pBuffer)
    {
        assert(1==0);
    }

    template<>
    void FFTInv_C2C<Ipp32fc>(const Ipp32fc * pSrc,Ipp32fc* pDst, void * pFFTSpec, Ipp8u* pBuffer)
    {
        IppStatus status = ippsFFTInv_CToC_32fc(pSrc,pDst,(IppsFFTSpec_C_32fc*)pFFTSpec,pBuffer);        
        checkStatus(status);
    }
    template<>
    void FFTInv_C2C<Ipp64fc>(const Ipp64fc * pSrc,Ipp64fc* pDst, void * pFFTSpec, Ipp8u* pBuffer)
    {
        IppStatus status = ippsFFTInv_CToC_64fc(pSrc,pDst,(IppsFFTSpec_C_64fc*)pFFTSpec,pBuffer);        
        checkStatus(status);
    }

    template<typename T>
    struct CFFT
    {
        Ipp8u * pBuffer;
        Ipp8u * pSpec;
        Ipp8u * pSpecBuffer;
        void * fft;
        
        CFFT(size_t n) {
            int size,specbuffer,spec;            
            int flag = IPP_FFT_DIV_FWD_BY_N;
            FFTGetSizeC<T>(n,flag,ippAlgHintNone,&spec,&specbuffer,&size);            
            pSpec = Malloc<Ipp8u>(spec);
            pSpecBuffer = specbuffer > 0? Malloc<Ipp8u>(specbuffer) : NULL;
            pBuffer = Malloc<Ipp8u>(size);
            int order = std::log2(n);
            FFTInitC<T>(&fft,order, flag,ippAlgHintNone,pSpec,pSpecBuffer);                        
        }
        ~CFFT() {
            if(pSpec) Free(pSpec);
            if(pSpecBuffer) Free(pSpecBuffer);
            if(pBuffer) Free(pBuffer);
        }
        void Forward(const T* pSrc, T * pDst)
        {                               
            FFTFwd_C2C<T>(pSrc,pDst, fft, pBuffer);                                    
        }
        void Inverse(const T* pSrc, T * pDst)
        {                               
            FFTInv_C2C<T>(pSrc,pDst, fft, pBuffer);            
        }        
    };


    template<typename T>
    struct RFFT
    {
        Ipp8u * pBuffer;
        Ipp8u * pSpec;
        Ipp8u * pSpecBuffer;
        void  * fft;
                        
        RFFT(size_t n) {
            int size,specbuffer,spec;            
            int flag = IPP_FFT_DIV_FWD_BY_N;
            FFTGetSizeR<T>(n,flag,ippAlgHintNone,&spec,&specbuffer,&size);            
            pSpec = Malloc<Ipp8u>(spec);
            pSpecBuffer = specbuffer > 0? Malloc<Ipp8u>(specbuffer) : NULL;
            pBuffer = Malloc<Ipp8u>(size);
            int order = std::log2(n);
            FFTInitR<T>(&fft,order, flag,ippAlgHintNone,pSpec,pSpecBuffer);                        
        }
        ~RFFT() {
            if(pSpec) Free(pSpec);
            if(pSpecBuffer) Free(pSpecBuffer);
            if(pBuffer) Free(pBuffer);            
        }
        
        void Forward(const Ipp32f* pSrc, Ipp32f * pDst)
        {                   
            FFTFwd_RToPack<T>(pSrc,pDst,fft,pBuffer);            
        }
        void Inverse(const Ipp32f* pSrc, Ipp32f * pDst)
        {                   
            FFTInv_PackToR<T>(pSrc,pDst,fft,pBuffer);            
        }
        void Unpack(int len, Ipp32f * pSrc, Ipp32fc * pDst) {
            ConjPack<Ipp32f,Ipp32fc>(pSrc,pDst,len);
        }
    };

    void fft(size_t n, const Ipp32fc * pSrc, Ipp32fc * pDst)
    {
        CFFT<Ipp32fc> f(n);
        f.Forward(pSrc,pDst);
    }
    void fft(CFFT<Ipp32fc> &f,const Ipp32fc * pSrc, Ipp32fc * pDst)
    {        
        f.Forward(pSrc,pDst);
    }
    void ifft(size_t n, const Ipp32fc * pSrc, Ipp32fc * pDst)
    {
        CFFT<Ipp32fc> f(n);
        f.Inverse(pSrc,pDst);
    }
    void ifft(CFFT<Ipp32fc> &f,const Ipp32fc * pSrc, Ipp32fc * pDst)
    {        
        f.Inverse(pSrc,pDst);
    }
    
    /*
    void fft(size_t n, const Ipp32f * pSrcRe, const Ipp32f* pSrcIm, Ipp32f * pDstRe, Ipp32f * pDstIm)
    {
        CFFT<Ipp32f> f(n);
        f.Forward(pSrcRe,pSrcIm,pDstRe,pDstIm);
    }    
    void fft(CFFT<Ipp32f> &f, const Ipp32f * pSrcRe, const Ipp32f* pSrcIm, Ipp32f * pDstRe, Ipp32f * pDstIm)
    {    
        f.Forward(pSrcRe,pSrcIm,pDstRe,pDstIm);
    }    
    void ifft(size_t n, const Ipp32f * pSrcRe, const Ipp32f * pSrcIm, Ipp32f * pDstRe, Ipp32f * pDstIm)    
    {
        CFFT<Ipp32f> f(n);
        f.Inverse(pSrcRe, pSrcIm, pDstRe, pDstIm);
    }
    void ifft(CFF<Ipp32f> &f, const Ipp32f * pSrcRe, const Ipp32f * pSrcIm, Ipp32f * pDstRe, Ipp32f * pDstIm)    
    {        
        f.Inverse(pSrcRe, pSrcIm, pDstRe, pDstIm);
    }
    */

    void fft(size_t n, const Ipp64fc * pSrc, Ipp64fc * pDst)
    {
        CFFT<Ipp64fc> f(n);
        f.Forward(pSrc,pDst);
    }
    void fft(CFFT<Ipp64fc> &f,const Ipp64fc * pSrc, Ipp64fc * pDst)
    {        
        f.Forward(pSrc,pDst);
    }
    void ifft(size_t n, const Ipp64fc * pSrc, Ipp64fc * pDst)
    {
        CFFT<Ipp64fc> f(n);
        f.Inverse(pSrc,pDst);
    }
    void ifft(CFFT<Ipp64fc> &f,const Ipp64fc * pSrc, Ipp64fc * pDst)
    {        
        f.Inverse(pSrc,pDst);
    }

    /*
    void fft(size_t n, const Ipp64f * pSrcRe, const Ipp64f* pSrcIm, Ipp64f * pDstRe, Ipp64f * pDstIm)
    {
        CFFT<Ipp64f> f(n);
        f.Forward(pSrcRe,pSrcIm,pDstRe,pDstIm);
    }
    void fft(CFFT<ipp64f> &f, const Ipp64f * pSrcRe, const Ipp64f* pSrcIm, Ipp64f * pDstRe, Ipp64f * pDstIm)
    {    
        f.Forward(pSrcRe,pSrcIm,pDstRe,pDstIm);
    }
    void ifft(size_t n, const Ipp64f * pSrcRe, const Ipp64f * pSrcIm, Ipp64f * pDstRe, Ipp64f * pDstIm)    
    {
        CFFT<Ipp64f> f(n);
        f.Inverse(pSrcRe, pSrcIm, pDstRe, pDstIm);
    }
    void ifft(CFFT<Ipp64f> &f, const Ipp64f * pSrcRe, const Ipp64f * pSrcIm, Ipp64f * pDstRe, Ipp64f * pDstIm)    
    {        
        f.Inverse(pSrcRe, pSrcIm, pDstRe, pDstIm);
    }
    */

}