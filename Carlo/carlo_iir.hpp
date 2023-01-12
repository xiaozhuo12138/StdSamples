#pragma once

namespace Casino::IPP
{
    template<typename T>
    void IIRInit(void ** ppState, const T* pTaps, int order, const T* pDlyLine, Ipp8u * pBuf)
    {
        assert(1==0);
    }
    template<>
    void IIRInit<Ipp32f>(void ** ppState, const Ipp32f* pTaps, int order, const Ipp32f* pDlyLine, Ipp8u * pBuf)
    {
        IppStatus status = ippsIIRInit_32f((IppsIIRState_32f**)ppState,pTaps,order,pDlyLine,pBuf);
        checkStatus(status);
    }
    template<>
    void IIRInit<Ipp64f>(void ** ppState, const Ipp64f* pTaps, int order, const Ipp64f* pDlyLine, Ipp8u * pBuf)
    {
        IppStatus status = ippsIIRInit_64f((IppsIIRState_64f**)ppState,pTaps,order,pDlyLine,pBuf);
        checkStatus(status);
    }
    template<typename T>
    void IIRInit_BiQuad(void ** ppState, const T* pTaps, int numBq, const T* pDlyLine, Ipp8u * pBuf)
    {
        assert(1==0);
    }
    template<>
    void IIRInit_BiQuad<Ipp32f>(void ** ppState, const Ipp32f* pTaps, int numBq, const Ipp32f* pDlyLine, Ipp8u * pBuf)
    {
        IppStatus status = ippsIIRInit_BiQuad_32f((IppsIIRState_32f**)ppState,pTaps,numBq,pDlyLine,pBuf);
        checkStatus(status);
    }
    template<>
    void IIRInit_BiQuad<Ipp64f>(void ** ppState, const Ipp64f* pTaps, int numBq, const Ipp64f* pDlyLine, Ipp8u * pBuf)
    {
        IppStatus status = ippsIIRInit_BiQuad_64f((IppsIIRState_64f**)ppState,pTaps,numBq,pDlyLine,pBuf);
        checkStatus(status);
    }
    template<typename T>
    void IIRGetStateSize(int order, int * pBufferSize)
    {
        assert(1==0);
    }
    template<>
    void IIRGetStateSize<Ipp32f>(int order, int * pBufferSize)
    {
        IppStatus status = ippsIIRGetStateSize_32f(order,pBufferSize);
        checkStatus(status);
    }
    template<>
    void IIRGetStateSize<Ipp64f>(int order, int * pBufferSize)
    {
        IppStatus status = ippsIIRGetStateSize_64f(order,pBufferSize);
        checkStatus(status);
    }
    template<typename T>
    void IIRGetStateSize_BiQuad(int order, int * pBufferSize)
    {
        assert(1==0);
    }
    template<>
    void IIRGetStateSize_BiQuad<Ipp32f>(int order, int * pBufferSize)
    {
        IppStatus status = ippsIIRGetStateSize_BiQuad_32f(order,pBufferSize);
        checkStatus(status);
    }
    template<>
    void IIRGetStateSize_BiQuad<Ipp64f>(int order, int * pBufferSize)
    {
        IppStatus status = ippsIIRGetStateSize_BiQuad_64f(order,pBufferSize);
        checkStatus(status);
    }
    template<typename T>
    void IIRGetDlyLine(void * pState, T * pDlyLine)
    {
        assert(1==0);
    }
    template<>
    void IIRGetDlyLine<Ipp32f>(void * pState, Ipp32f * pDlyLine)
    {
        IppStatus status = ippsIIRGetDlyLine_32f((IppsIIRState_32f*)pState,pDlyLine);
        checkStatus(status);
    }
    template<>
    void IIRGetDlyLine<Ipp64f>(void * pState, Ipp64f * pDlyLine)
    {
        IppStatus status = ippsIIRGetDlyLine_64f((IppsIIRState_64f*)pState,pDlyLine);
        checkStatus(status);
    }
    template<typename T>
    void IIRSetDlyLine(void * pState, T * pDlyLine)
    {
        assert(1==0);
    }
    template<>
    void IIRSetDlyLine<Ipp32f>(void * pState, Ipp32f * pDlyLine)
    {
        IppStatus status = ippsIIRSetDlyLine_32f((IppsIIRState_32f*)pState,pDlyLine);
        checkStatus(status);
    }
    template<>
    void IIRSetDlyLine<Ipp64f>(void * pState, Ipp64f * pDlyLine)
    {
        IppStatus status = ippsIIRSetDlyLine_64f((IppsIIRState_64f*)pState,pDlyLine);
        checkStatus(status);
    }
    template<typename T>
    void IIR_(const T* pSrc, T * pDst, int len, void * pState)
    {
        assert(1==0);
    }
    template<>
    void IIR_<Ipp32f>(const Ipp32f* pSrc, Ipp32f * pDst, int len, void * pState)
    {
        IppStatus status = ippsIIR_32f(pSrc,pDst,len,(IppsIIRState_32f*)pState);
        checkStatus(status);
    }
    template<>
    void IIR_<Ipp64f>(const Ipp64f* pSrc, Ipp64f * pDst, int len, void * pState)
    {
        IppStatus status = ippsIIR_64f(pSrc,pDst,len,(IppsIIRState_64f*)pState);
        checkStatus(status);
    }    
    void IIRGenGetBufferSize(int order, int * pBufferSize)
    {
        IppStatus status = ippsIIRGenGetBufferSize(order,pBufferSize);
        checkStatus(status);
    }
    
    void IIRGenLowpassButterworth(Ipp64f freq, Ipp64f ripple, int order, Ipp64f * pTaps, Ipp8u * pBuffer)
    {
        IppStatus status = ippsIIRGenLowpass_64f(freq,ripple,order,pTaps,ippButterworth,pBuffer);
        checkStatus(status);
    }
    void IIRGenHighpassButterworth(Ipp64f freq, Ipp64f ripple, int order, Ipp64f * pTaps, Ipp8u * pBuffer)
    {
        IppStatus status = ippsIIRGenHighpass_64f(freq,ripple,order,pTaps,ippButterworth,pBuffer);
        checkStatus(status);
    }
    void IIRGenLowpassChebyshev1(Ipp64f freq, Ipp64f ripple, int order, Ipp64f * pTaps, Ipp8u * pBuffer)
    {
        IppStatus status = ippsIIRGenLowpass_64f(freq,ripple,order,pTaps,ippChebyshev1,pBuffer);
        checkStatus(status);
    }
    void IIRGenHighpassChebyshev1(Ipp64f freq, Ipp64f ripple, int order, Ipp64f * pTaps, Ipp8u * pBuffer)
    {
        IppStatus status = ippsIIRGenHighpass_64f(freq,ripple,order,pTaps,ippChebyshev1,pBuffer);
        checkStatus(status);
    }


    
    template<typename T>
    struct IIR
    {
        Ipp8u * pBuffer;
        size_t  len;                
        void  * pState;
        T*      dlyLine;
        
        // B0,B1..Border,A0,A1..Aorder = 2*(order+1)
        IIR(size_t n, int order, const T * taps) {
            int bufferSize;
            IIRGetStateSize<T>(n,&bufferSize);            
            len = n;
            pBuffer = Malloc<Ipp8u>(bufferSize);            
            IIRInit<T>(&pState,taps,order,NULL,pBuffer);            
            dlyLine = Malloc<T>(2*(order+1));
        }
        ~IIR() {
            if(pBuffer) Free(pBuffer);      
            if(dlyLine) Free(dlyLine);  
        }
        void setCoefficients(int n, int order, const T* taps) {
            if(pBuffer) Free(pBuffer);
            int bufferSize;
            len = n;
            // have to save it so it doesn't pop
            IIRGetDlyLine<T>(pState,dlyLine);            
            IIRGetStateSize<T>(n,&bufferSize);            
            pBuffer = Malloc<Ipp8u>(bufferSize);            
            IIRInit<T>(&pState,taps,order,NULL,pBuffer);            
            IIRSetDlyLine<T>(pState,dlyLine);            
        }
        void Execute(const T* pSrc, T* pDst)
        {
            IIR_<T>(pSrc,pDst,len,pState);            
        }
    };

    template<typename T>
    struct IIRBiquad
    {        
        Ipp8u * pBuffer;
        size_t  len;                
        void  * pState;
        T*      dlyLine;
        

        // B0,B1..Border,A0,A1..Aorder = 2*(order+1)
        IIRBiquad(size_t n, int numBiquads, const T* taps) {
            int bufferSize;
            IIRGetStateSize<T>(n,&bufferSize);            
            len = n;
            pBuffer = Malloc<Ipp8u>(bufferSize);            
            IIRInit_BiQuad<T>(&pState,taps,numBiquads,NULL,pBuffer);            
            dlyLine = Malloc<T>(2*numBiquads);
        }
        ~IIRBiquad() {
            if(pBuffer) Free(pBuffer);        
            if(dlyLine) Free(dlyLine);
        }
        void setCoefficients(int order, int numBiquads, const T* taps) {
            if(pBuffer) Free(pBuffer);
            int bufferSize;
            IIRGetDlyLine<T>(pState,dlyLine);            
            IIRGetStateSize<T>(numBiquads,&bufferSize);            
            pBuffer = Malloc<Ipp8u>(bufferSize);            
            IIRInit_BiQuad<T>(&pState,taps,numBiquads,NULL,pBuffer);            
            IIRSetDlyLine<T>(pState,dlyLine);            
        }
        void Execute(const T* pSrc, T* pDst)
        {
            IIR_<T>(pSrc,pDst,len,pState);            
        }
    };


    struct IIRButterworthLowpassFilter
    {
        IIR<Ipp64f> * iir;
        Ipp64f      * taps;
        Ipp8u       * buffer;
        IIRButterworthLowpassFilter(size_t n, Ipp64f freq, int order)
        {
            int bufferSize;
            IIRGetStateSize<T>(n,&bufferSize);   
            buffer = Malloc<Ipp8u>(bufferSize);
            taps   = Malloc<Ipp64f>(n);
            IIRGenLowpass<Ipp64f>(freq,0,order,taps,ippButterworth,buffer);
            iir = new IIR<Ipp64f>(n,order,taps);
            assert(iir != nullptr);
        }
        ~IIRButterworthLowpassFilter() {
            if(iir) delete iir;
            if(taps) Free(taps);
            if(buffer) Free(buffer);
        }
        void Execute(const T* pSrc, T* pDst)
        {
            iir->Execute(pSrc,pDst);
        }
    };

    struct IIRButterworthHighpassFilter
    {
        IIR<Ipp64f> * iir;
        Ipp64f      * taps;
        Ipp8u       * buffer;
        IIRButterworthHighpassFilter(size_t n, Ipp64f freq, int order)
        {
            int bufferSize;
            IIRGetStateSize<T>(n,&bufferSize);   
            buffer = Malloc<Ipp8u>(bufferSize);
            taps   = Malloc<Ipp64f>(n);
            IIRGenHighpass<Ipp64f>(freq,0,order,taps,ippButterworth,buffer);
            iir = new IIR<Ipp64f>(n,order,taps);
            assert(iir != nullptr);
        }
        ~IIRButterworthHighpassFilter() {
            if(iir) delete iir;
            if(taps) Free(taps);
            if(buffer) Free(buffer);
        }
        void Execute(const T* pSrc, T* pDst)
        {
            iir->Execute(pSrc,pDst);
        }
    };

    struct IIRChebyshevLowpassFilter
    {
        IIR<Ipp64f> * iir;
        Ipp64f      * taps;
        Ipp8u       * buffer;


        IIRChebyshevLowpassFilter(int order,size_t n, Ipp64f freq, Ipp64f ripple)
        {
            int bufferSize;
            IIRGetStateSize<T>(n,&bufferSize);   
            buffer = Malloc<Ipp8u>(bufferSize);
            taps   = Malloc<Ipp64f>(n);
            IIRGenLowpass<Ipp64f>(freq,ripple,order,taps,ippChebyshev1,buffer);
            iir = new IIR<Ipp64f>(n,order,taps);
            assert(iir != nullptr);
        }
        ~IIRChebyshevLowpassFilter() {
            if(iir) delete iir;
            if(taps) Free(taps);
            if(buffer) Free(buffer);
        }
        void Execute(const T* pSrc, T* pDst)
        {
            iir->Execute(pSrc,pDst);
        }
    };

    struct IIRChebyshevHighpassFilter
    {
        IIR<Ipp64f> * iir;
        Ipp64f      * taps;
        Ipp8u       * buffer;


        IIRChebyshevHighpassFilter(int order,size_t n, Ipp64f freq, Ipp64f ripple)
        {
            int bufferSize;
            IIRGetStateSize<T>(n,&bufferSize);   
            buffer = Malloc<Ipp8u>(bufferSize);
            taps   = Malloc<Ipp64f>(n);
            IIRGenHighpass<Ipp64f>(freq,ripple,order,taps,ippChebyshev1,buffer);
            iir = new IIR<Ipp64f>(n,order,taps);
            assert(iir != nullptr);
        }
        ~IIRChebyshevHighpassFilter() {
            if(iir) delete iir;
            if(taps) Free(taps);
            if(buffer) Free(buffer);
        }
        void Execute(const T* pSrc, T* pDst)
        {
            iir->Execute(pSrc,pDst);
        }
    };

    template<typename T>
    void filter(IIR<T> & filter, const T* pSrc, T* pDst)
    {
        filter.Execute(pSrc,pDst);
    }
    template<typename T>
    void filter(IIRBiquad<T> & filter, const T* pSrc, T* pDst)
    {
        filter.Execute(pSrc,pDst);
    }
    void filter(IIRButterworthLowpassFilter & filter, const T* pSrc, T* pDst)
    {
        filter.Execute(pSrc,pDst);
    }
    void filter(IIRButterworthHighpassFilter & filter, const T* pSrc, T* pDst)
    {
        filter.Execute(pSrc,pDst);
    }
    void filter(IIRChebyshevLowpassFilter & filter, const T* pSrc, T* pDst)
    {
        filter.Execute(pSrc,pDst);
    }
    void filter(IIRChebyshevHighpassFilter & filter, const T* pSrc, T* pDst)
    {
        filter.Execute(pSrc,pDst);
    }

}