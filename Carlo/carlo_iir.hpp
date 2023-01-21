#pragma once
#define PRINT(x) std::cout << (x) << std::endl

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
        
        IIR() {
            pBuffer = nullptr;
            len = 0;
            pState = nullptr;
            dlyLine = nullptr;
        }
        // B0,B1..Border,A0,A1..Aorder = 2*(order+1)
        IIR(size_t n, int order, const T * taps) {
            initCoefficients(n,order,taps);
        }
        ~IIR() {
            if(pBuffer) Free(pBuffer);      
            if(dlyLine) Free(dlyLine);          
        }        
        void initCoefficients(int n, int order, const T* taps) {
            if(pBuffer) Free(pBuffer);
            if(dlyLine) Free(dlyLine);
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
        size_t  len,numb;
        void  * pState;
        T*      dlyLine;
        
        IIRBiquad() {
            pBuffer = nullptr;
            len = 0;
            numb=0;
            pState = nullptr;
            dlyLine = nullptr;
        }
        // B0,B1..Border,A0,A1..Aorder = 2*(order+1)
        IIRBiquad(size_t n, int numBiquads, const T* taps) {                                    
            initCoefficients(n,numBiquads,taps);
        }
        ~IIRBiquad() {
            if(pBuffer) Free(pBuffer);        
            if(dlyLine) Free(dlyLine);
        }        
        void initCoefficients(size_t n, int numBiquads, const T* taps) {
            
            int bufferSize;
            
            len = n;            
            if(dlyLine) IIRGetDlyLine<T>(pState,dlyLine);            
            else dlyLine = Malloc<T>(2*numBiquads);            
            if( numb != numBiquads)
            {                
                if(pBuffer) Free(pBuffer);                
                IIRGetStateSize<T>(numBiquads,&bufferSize);                                                    
                pBuffer = Malloc<Ipp8u>(bufferSize);                            
                numb = numBiquads;
            }         
            IIRInit_BiQuad<T>(&pState,taps,numBiquads,dlyLine,pBuffer);                                    
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
            IIRGetStateSize<Ipp64f>(n,&bufferSize);   
            buffer = Malloc<Ipp8u>(bufferSize);
            taps   = Malloc<Ipp64f>(n);
            IIRGenLowpassButterworth(freq,0,order,taps,buffer);
            iir = new IIR<Ipp64f>(n,order,taps);
            assert(iir != nullptr);
        }
        ~IIRButterworthLowpassFilter() {
            if(iir) delete iir;
            if(taps) Free(taps);
            if(buffer) Free(buffer);
        }
        void Execute(const Ipp64f* pSrc, Ipp64f* pDst)
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
            IIRGetStateSize<Ipp64f>(n,&bufferSize);   
            buffer = Malloc<Ipp8u>(bufferSize);
            taps   = Malloc<Ipp64f>(n);
            IIRGenHighpassButterworth(freq,0,order,taps,buffer);
            iir = new IIR<Ipp64f>(n,order,taps);
            assert(iir != nullptr);
        }
        ~IIRButterworthHighpassFilter() {
            if(iir) delete iir;
            if(taps) Free(taps);
            if(buffer) Free(buffer);
        }
        void Execute(const Ipp64f* pSrc, Ipp64f* pDst)
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
            IIRGetStateSize<Ipp64f>(n,&bufferSize);   
            buffer = Malloc<Ipp8u>(bufferSize);
            taps   = Malloc<Ipp64f>(n);
            IIRGenLowpassChebyshev1(freq,ripple,order,taps,buffer);
            iir = new IIR<Ipp64f>(n,order,taps);
            assert(iir != nullptr);
        }
        ~IIRChebyshevLowpassFilter() {
            if(iir) delete iir;
            if(taps) Free(taps);
            if(buffer) Free(buffer);
        }
        void Execute(const Ipp64f* pSrc, Ipp64f* pDst)
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
            IIRGetStateSize<Ipp64f>(n,&bufferSize);   
            buffer = Malloc<Ipp8u>(bufferSize);
            taps   = Malloc<Ipp64f>(n);
            IIRGenHighpassChebyshev1(freq,ripple,order,taps,buffer);
            iir = new IIR<Ipp64f>(n,order,taps);
            assert(iir != nullptr);
        }
        ~IIRChebyshevHighpassFilter() {
            if(iir) delete iir;
            if(taps) Free(taps);
            if(buffer) Free(buffer);
        }
        void Execute(const Ipp64f* pSrc, Ipp64f* pDst)
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
    void filter(IIRButterworthLowpassFilter & filter, const Ipp64f* pSrc, Ipp64f* pDst)
    {
        filter.Execute(pSrc,pDst);
    }
    void filter(IIRButterworthHighpassFilter & filter, const Ipp64f* pSrc, Ipp64f* pDst)
    {
        filter.Execute(pSrc,pDst);
    }
    void filter(IIRChebyshevLowpassFilter & filter, const Ipp64f* pSrc, Ipp64f* pDst)
    {
        filter.Execute(pSrc,pDst);
    }
    void filter(IIRChebyshevHighpassFilter & filter, const Ipp64f* pSrc, Ipp64f* pDst)
    {
        filter.Execute(pSrc,pDst);
    }

}