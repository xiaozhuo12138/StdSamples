#pragma once

namespace Casino::IPP
{

    template<typename T>
    void DFTInitR(int length, int flag, IppHintAlgorithm hint, void * pDFTSpec, Ipp8u * pMemInit)
    {
        assert(1==0);
    }
    template<>
    void DFTInitR<Ipp32f>(int length, int flag, IppHintAlgorithm hint, void * pDFTSpec, Ipp8u * pMemInit)
    {
        IppStatus status = ippsDFTInit_R_32f(length,flag,hint,(IppsDFTSpec_R_32f*)pDFTSpec,pMemInit);        
        checkStatus(status);            
    }
    template<>
    void DFTInitR<Ipp64f>(int length, int flag, IppHintAlgorithm hint, void * pDFTSpec, Ipp8u * pMemInit)
    {
        IppStatus status = ippsDFTInit_R_64f(length,flag,hint,(IppsDFTSpec_R_64f*)pDFTSpec,pMemInit);        
        checkStatus(status);            
    }
    template<typename T>
    void DFTInitC(int length, int flag, IppHintAlgorithm hint, void * pDFTSpec, Ipp8u * pMemInit)
    {
        assert(1==0);
    }
    template<>
    void DFTInitC<Ipp32f>(int length, int flag, IppHintAlgorithm hint, void * pDFTSpec, Ipp8u * pMemInit)
    {
        IppStatus status = ippsDFTInit_C_32f(length,flag,hint,(IppsDFTSpec_C_32f*)pDFTSpec,pMemInit);        
        checkStatus(status);            
    }
    template<>
    void DFTInitC<Ipp64f>(int length, int flag, IppHintAlgorithm hint, void * pDFTSpec, Ipp8u * pMemInit)
    {
        IppStatus status = ippsDFTInit_C_64f(length,flag,hint,(IppsDFTSpec_C_64f*)pDFTSpec,pMemInit);        
        checkStatus(status);            
    }
    template<>
    void DFTInitC<Ipp32fc>(int length, int flag, IppHintAlgorithm hint, void * pDFTSpec, Ipp8u * pMemInit)
    {
        IppStatus status = ippsDFTInit_C_32fc(length,flag,hint,(IppsDFTSpec_C_32fc*)pDFTSpec,pMemInit);        
        checkStatus(status);            
    }
    template<>
    void DFTInitC<Ipp64fc>(int length, int flag, IppHintAlgorithm hint, void * pDFTSpec, Ipp8u * pMemInit)
    {
        IppStatus status = ippsDFTInit_C_64fc(length,flag,hint,(IppsDFTSpec_C_64fc*)pDFTSpec,pMemInit);        
        checkStatus(status);            
    }

    template<typename T>
    void DFTGetSizeR(int length, int flag, IppHintAlgorithm hint, int * pSizeSpec, int * pSizeInt, int * pSizeBuf)
    {
        assert(1==0);
    }
    template<>
    void DFTGetSizeR<Ipp32f>(int length, int flag, IppHintAlgorithm hint, int * pSizeSpec, int * pSizeInt, int * pSizeBuf)
    {
        IppStatus status = ippsDFTGetSize_R_32f(length,flag,hint,pSizeSpec,pSizeInt,pSizeBuf);
        checkStatus(status);            
    }
    template<>
    void DFTGetSizeR<Ipp64f>(int length, int flag, IppHintAlgorithm hint, int * pSizeSpec, int * pSizeInt, int * pSizeBuf)
    {
        IppStatus status = ippsDFTGetSize_R_64f(length,flag,hint,pSizeSpec,pSizeInt,pSizeBuf);
        checkStatus(status);            
    }
    template<typename T>
    void DFTGetSizeC(int length, int flag, IppHintAlgorithm hint, int * pSizeSpec, int * pSizeInt, int * pSizeBuf)
    {
        assert(1==0);
    }
    template<>
    void DFTGetSizeC<Ipp32f>(int length, int flag, IppHintAlgorithm hint, int * pSizeSpec, int * pSizeInt, int * pSizeBuf)
    {
        IppStatus status = ippsDFTGetSize_C_32f(length,flag,hint,pSizeSpec,pSizeInt,pSizeBuf);
        checkStatus(status);            
    }
    template<>
    void DFTGetSizeC<Ipp64f>(int length, int flag, IppHintAlgorithm hint, int * pSizeSpec, int * pSizeInt, int * pSizeBuf)
    {
        IppStatus status = ippsDFTGetSize_C_64f(length,flag,hint,pSizeSpec,pSizeInt,pSizeBuf);
        checkStatus(status);            
    }
    template<>
    void DFTGetSizeC<Ipp32fc>(int length, int flag, IppHintAlgorithm hint, int * pSizeSpec, int * pSizeInt, int * pSizeBuf)
    {
        IppStatus status = ippsDFTGetSize_C_32fc(length,flag,hint,pSizeSpec,pSizeInt,pSizeBuf);
        checkStatus(status);            
    }
    template<>
    void DFTGetSizeC<Ipp64fc>(int length, int flag, IppHintAlgorithm hint, int * pSizeSpec, int * pSizeInt, int * pSizeBuf)
    {
        IppStatus status = ippsDFTGetSize_C_64fc(length,flag,hint,pSizeSpec,pSizeInt,pSizeBuf);
        checkStatus(status);            
    }

    template<typename T>
    void DFTFwd_C2C(const T * pSrcRe, const T * pSrcIm,  T* pDstRe, T* pDstIm, void * pDFTSpec, Ipp8u* pBuffer)
    {
        // sometime later will try to put some more useful exceptions here
        // it should never be called but I dont think there is a virtual function without a class
        assert(1==0);
    }
    
    template<>
    void DFTFwd_C2C<Ipp32f>(const Ipp32f * pSrcRe, const Ipp32f * pSrcIm, Ipp32f* pDstRe, Ipp32f* pDstIm, void * pDFTSpec, Ipp8u* pBuffer)
    {
        IppStatus status = ippsDFTFwd_CToC_32f(pSrcRe,pSrcIm,pDstRe,pDstIm,(IppsDFTSpec_C_32f*)pDFTSpec,pBuffer);        
        checkStatus(status);
    }
    template<>
    void DFTFwd_C2C<Ipp64f>(const Ipp64f * pSrcRe, const Ipp64f * pSrcIm, Ipp64f* pDstRe, Ipp64f* pDstIm, void * pDFTSpec, Ipp8u* pBuffer)
    {
        IppStatus status = ippsDFTFwd_CToC_64f(pSrcRe,pSrcIm,pDstRe,pDstIm,(IppsDFTSpec_C_64f*)pDFTSpec,pBuffer);        
        checkStatus(status);
    }

    template<typename T>
    void DFTFwd_C2C(const T * pSrc, T* pDst, void * pDFTSpec, Ipp8u* pBuffer)
    {
        assert(1==0);
    }

    template<>
    void DFTFwd_C2C<Ipp32fc>(const Ipp32fc * pSrc,Ipp32fc* pDst, void * pDFTSpec, Ipp8u* pBuffer)
    {
        IppStatus status = ippsDFTFwd_CToC_32fc(pSrc,pDst,(IppsDFTSpec_C_32fc*)pDFTSpec,pBuffer);        
        checkStatus(status);
    }
    template<>
    void DFTFwd_C2C<Ipp64fc>(const Ipp64fc * pSrc,Ipp64fc* pDst, void * pDFTSpec, Ipp8u* pBuffer)
    {
        IppStatus status = ippsDFTFwd_CToC_64fc(pSrc,pDst,(IppsDFTSpec_C_64fc*)pDFTSpec,pBuffer);        
        checkStatus(status);
    }

    template<typename T>
    void DFTInv_C2C(const T * pSrcRe, const T * pSrcIm,  T* pDstRe, T* pDstIm, void * pDFTSpec, Ipp8u* pBuffer)
    {
        // sometime later will try to put some more useful exceptions here
        // it should never be called but I dont think there is a virtual function without a class
        assert(1==0);
    }
    template<>
    void DFTInv_C2C<Ipp32f>(const Ipp32f * pSrcRe, const Ipp32f * pSrcIm, Ipp32f* pDstRe, Ipp32f* pDstIm, void * pDFTSpec, Ipp8u* pBuffer)
    {
        IppStatus status = ippsDFTInv_CToC_32f(pSrcRe,pSrcIm,pDstRe,pDstIm,(IppsDFTSpec_C_32f*)pDFTSpec,pBuffer);        
        checkStatus(status);
    }

    template<>
    void DFTInv_C2C<Ipp64f>(const Ipp64f * pSrcRe, const Ipp64f * pSrcIm, Ipp64f* pDstRe, Ipp64f* pDstIm, void * pDFTSpec, Ipp8u* pBuffer)
    {
        IppStatus status = ippsDFTInv_CToC_64f(pSrcRe,pSrcIm,pDstRe,pDstIm,(IppsDFTSpec_C_64f*)pDFTSpec,pBuffer);        
        checkStatus(status);
    }

    template<typename T>
    void DFTInv_C2C(const T * pSrc, T* pDst, void * pDFTSpec, Ipp8u* pBuffer)
    {
        assert(1==0);
    }

    template<>
    void DFTInv_C2C<Ipp32fc>(const Ipp32fc * pSrc,Ipp32fc* pDst, void * pDFTSpec, Ipp8u* pBuffer)
    {
        IppStatus status = ippsDFTInv_CToC_32fc(pSrc,pDst,(IppsDFTSpec_C_32fc*)pDFTSpec,pBuffer);        
        checkStatus(status);
    }
    template<>
    void DFTInv_C2C<Ipp64fc>(const Ipp64fc * pSrc,Ipp64fc* pDst, void * pDFTSpec, Ipp8u* pBuffer)
    {
        IppStatus status = ippsDFTInv_CToC_64fc(pSrc,pDst,(IppsDFTSpec_C_64fc*)pDFTSpec,pBuffer);        
        checkStatus(status);
    }

    template<typename T>
    void DFTFwd_RToPack(const T * pSrc, T * pDst, void * pDFTSpec, Ipp8u * pBuffer)
    {
        assert(1==0);
    }
    template<>
    void DFTFwd_RToPack<Ipp32f>(const Ipp32f * pSrc, Ipp32f * pDst, void * pDFTSpec, Ipp8u * pBuffer)
    {
        IppStatus status = ippsDFTFwd_RToPack_32f(pSrc,pDst,(IppsDFTSpec_R_32f*)pDFTSpec,pBuffer);
        checkStatus(status);
    }
    template<>
    void DFTFwd_RToPack<Ipp64f>(const Ipp64f * pSrc, Ipp64f * pDst, void * pDFTSpec, Ipp8u * pBuffer)
    {
        IppStatus status = ippsDFTFwd_RToPack_64f(pSrc,pDst,(IppsDFTSpec_R_64f*)pDFTSpec,pBuffer);
        checkStatus(status);
    }

    template<typename T>
    void DFTFwd_RToPerm(const T * pSrc, T * pDst, void * pDFTSpec, Ipp8u * pBuffer)
    {
        assert(1==0);
    }
    template<>
    void DFTFwd_RToPerm<Ipp32f>(const Ipp32f * pSrc, Ipp32f * pDst, void * pDFTSpec, Ipp8u * pBuffer)
    {
        IppStatus status = ippsDFTFwd_RToPerm_32f(pSrc,pDst,(IppsDFTSpec_R_32f*)pDFTSpec,pBuffer);
        checkStatus(status);
    }
    template<>
    void DFTFwd_RToPerm<Ipp64f>(const Ipp64f * pSrc, Ipp64f * pDst, void * pDFTSpec, Ipp8u * pBuffer)
    {
        IppStatus status = ippsDFTFwd_RToPerm_64f(pSrc,pDst,(IppsDFTSpec_R_64f*)pDFTSpec,pBuffer);
        checkStatus(status);
    }

    
    template<typename T>
    void DFTInv_PackToR(const T * pSrc, T * pDst, void * pDFTSpec, Ipp8u * pBuffer)
    {
        assert(1==0);
    }
    template<>
    void DFTInv_PackToR<Ipp32f>(const Ipp32f * pSrc, Ipp32f * pDst, void * pDFTSpec, Ipp8u * pBuffer)
    {
        IppStatus status = ippsDFTInv_PackToR_32f(pSrc,pDst,(IppsDFTSpec_R_32f*)pDFTSpec,pBuffer);
        checkStatus(status);
    }
    template<>
    void DFTInv_PackToR<Ipp64f>(const Ipp64f * pSrc, Ipp64f * pDst, void * pDFTSpec, Ipp8u * pBuffer)
    {
        IppStatus status = ippsDFTInv_PackToR_64f(pSrc,pDst,(IppsDFTSpec_R_64f*)pDFTSpec,pBuffer);
        checkStatus(status);
    }

    
    template<typename T>
    void DFTInv_PermToR(const T * pSrc, T * pDst, void * pDFTSpec, Ipp8u * pBuffer)
    {
        assert(1==0);
    }
    template<>
    void DFTInv_PermToR<Ipp32f>(const Ipp32f * pSrc, Ipp32f * pDst, void * pDFTSpec, Ipp8u * pBuffer)
    {
        IppStatus status = ippsDFTInv_PermToR_32f(pSrc,pDst,(IppsDFTSpec_R_32f*)pDFTSpec,pBuffer);
        checkStatus(status);
    }
    template<>
    void DFTInv_PermToR<Ipp64f>(const Ipp64f * pSrc, Ipp64f * pDst, void * pDFTSpec, Ipp8u * pBuffer)
    {
        IppStatus status = ippsDFTInv_PermToR_64f(pSrc,pDst,(IppsDFTSpec_R_64f*)pDFTSpec,pBuffer);
        checkStatus(status);
    }

    template<typename T>
    void DFTInv_CCSToR(const T * pSrc, T * pDst, void * pDFTSpec, Ipp8u * pBuffer)
    {
        assert(1==0);
    }
    template<>
    void DFTInv_CCSToR<Ipp32f>(const Ipp32f * pSrc, Ipp32f * pDst, void * pDFTSpec, Ipp8u * pBuffer)
    {
        IppStatus status = ippsDFTInv_CCSToR_32f(pSrc,pDst,(IppsDFTSpec_R_32f*)pDFTSpec,pBuffer);
        checkStatus(status);
    }
    template<>
    void DFTInv_CCSToR<Ipp64f>(const Ipp64f * pSrc, Ipp64f * pDst, void * pDFTSpec, Ipp8u * pBuffer)
    {
        IppStatus status = ippsDFTInv_CCSToR_64f(pSrc,pDst,(IppsDFTSpec_R_64f*)pDFTSpec,pBuffer);
        checkStatus(status);
    }

    
    template<typename T>
    struct CDFT
    {
        Ipp8u * pBuffer;        
        Ipp8u * pSpecBuffer;        
        void  * fft;
                        
        CDFT(size_t n) {
            int size,specbuffer,spec;            
            int flag = IPP_FFT_DIV_FWD_BY_N;
            DFTGetSizeC<T>(n,flag,ippAlgHintNone,&spec,&specbuffer,&size);            
            fft = Malloc<Ipp8u>(spec);
            pSpecBuffer = specbuffer > 0? Malloc<Ipp8u>(specbuffer) : NULL;
            pBuffer = Malloc<Ipp8u>(size);
            int order = n;
            DFTInitC<T>(order,flag,ippAlgHintNone,fft,pSpecBuffer);                        
        }
        ~CDFT() {
            if(fft) Free(fft);
            if(pSpecBuffer) Free(pSpecBuffer);
            if(pBuffer) Free(pBuffer);
        }
        
        void Forward(const T* pSrc, T* pDst)
        {                   
            DFTFwd_C2C<T>(pSrc,pDst,fft,pBuffer);
        }
        void Inverse(const T* pSrc, T* pDst)
        {                   
            DFTInv_C2C<T>(pSrc,pDst,fft,pBuffer);
        }
    };

    
    template<typename T>
    struct RDFT
    {
        Ipp8u * pBuffer;        
        Ipp8u * pSpecBuffer;
        void  * fft;
        int     blocks;

        RDFT(size_t n) {
            int size,specbuffer,spec;            
            int flag = IPP_FFT_DIV_FWD_BY_N;
            DFTGetSizeR<T>(n,flag,ippAlgHintNone,&spec,&specbuffer,&size);            
            fft = Malloc<Ipp8u>(spec);
            pSpecBuffer = specbuffer > 0? Malloc<Ipp8u>(specbuffer) : NULL;
            pBuffer = Malloc<Ipp8u>(size);
            int order = n;
            blocks = n;
            DFTInitR<T>(order,flag,ippAlgHintNone,fft,pSpecBuffer);                        
        }
        ~RDFT() {
            if(fft) Free(fft);
            if(pSpecBuffer) Free(pSpecBuffer);
            if(pBuffer) Free(pBuffer);            
        }
        void Forward(const T* pSrc, T* pDst)
        {                   
            DFTFwd_RToPack<T>(pSrc,pDst,fft,pBuffer);            
        }
        void Inverse(const T* pSrc, T* pDst)
        {                   
            DFTInv_PackToR<T>(pSrc,pDst,fft,pBuffer);            
        }
        void Unpack(int len, Ipp32f * pSrc, Ipp32fc * pDst) {
            ConjPack<Ipp32f,Ipp32fc>(pSrc,pDst,len);
        }
        void Unpack(int len, Ipp64f * pSrc, Ipp64fc * pDst) {
            ConjPack<Ipp64f,Ipp64fc>(pSrc,pDst,len);
        }
    };

    void dft(size_t n, const Ipp32fc * pSrc, Ipp32fc * pDst)
    {
        CDFT<Ipp32fc> d(n);
        d.Forward(pSrc,pDst);
    }
    void dft(CDFT<Ipp32fc> &d, const Ipp32fc * pSrc, Ipp32fc * pDst)
    {
        d.Forward(pSrc,pDst);
    }
    void idft(size_t n, const Ipp32fc * pSrc, Ipp32fc * pDst)
    {
        CDFT<Ipp32fc> d(n);
        d.Inverse(pSrc,pDst);
    }
    void idft(CDFT<Ipp32fc> &d,const Ipp32fc * pSrc, Ipp32fc * pDst)
    {        
        d.Inverse(pSrc,pDst);
    }

    void dft(size_t n, const Ipp32f * pSrc, Ipp32f * pDst, Ipp32fc * unpacked)
    {
        RDFT<Ipp32f> d(n);
        d.Forward(pSrc,pDst);
        d.Unpack(n,pDst,unpacked);
    }
    void dft(RDFT<Ipp32f> &d, const Ipp32f * pSrc, Ipp32f * pDst, Ipp32fc * unpacked)
    {        
        d.Forward(pSrc,pDst);
        d.Unpack(d.blocks,pDst,unpacked);
    }
    void idft(size_t n, const Ipp32f * pSrc, Ipp32f * pDst)
    {
        RDFT<Ipp32f> d(n);
        d.Inverse(pSrc,pDst);
    }
    void idft(RDFT<Ipp32f> &d, const Ipp32f * pSrc, Ipp32f * pDst)
    {        
        d.Inverse(pSrc,pDst);    
    }

    void dft(size_t n, const Ipp64fc * pSrc, Ipp64fc * pDst)
    {
        CDFT<Ipp64fc> d(n);
        d.Forward(pSrc,pDst);
    }
    void dft(CDFT<Ipp64fc> &d, const Ipp64fc * pSrc, Ipp64fc * pDst)
    {
        d.Forward(pSrc,pDst);
    }
    void idft(size_t n, const Ipp64fc * pSrc, Ipp64fc * pDst)
    {
        CDFT<Ipp64fc> d(n);
        d.Inverse(pSrc,pDst);
    }
    void idft(CDFT<Ipp64fc> &d,const Ipp64fc * pSrc, Ipp64fc * pDst)
    {        
        d.Inverse(pSrc,pDst);
    }

    void dft(size_t n, const Ipp64f * pSrc, Ipp64f * pDst, Ipp64fc * unpacked)
    {
        RDFT<Ipp64f> d(n);
        d.Forward(pSrc,pDst);
        d.Unpack(n,pDst,unpacked);
    }
    void dft(RDFT<Ipp64f> &d, const Ipp64f * pSrc, Ipp64f * pDst, Ipp64fc * unpacked)
    {        
        d.Forward(pSrc,pDst);
        d.Unpack(d.blocks,pDst,unpacked);
    }
    void idft(size_t n, const Ipp64f * pSrc, Ipp64f * pDst)
    {
        RDFT<Ipp64f> d(n);
        d.Inverse(pSrc,pDst);
    }
    void idft(RDFT<Ipp64f> &d, const Ipp64f * pSrc, Ipp64f * pDst)
    {        
        d.Inverse(pSrc,pDst);    
    }
}