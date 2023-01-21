#pragma once

namespace Loudia
{
    class AOK {
    protected:
        // Internal parameters
        int _windowSize;
        int _hopSize;
        int _fftSize;
        Real _normVolume;

        // Internal variables
        int itemp;
        int xlen, tlen, nraf,tlag,nlag,mfft,nrad, nphi, nits;
        int slen,fftlen;
        int tstep,fstep,outct;
        
        int maxrad[100000];	/* max radius for each phi s.t. theta < pi */
        
        
        char name[10];
        
        
        Real pi, rtemp, rtemp1, rtemp2, mu, forget;
        Real vol;			/* total kernel volume	*/
        Real outdelay;		/* delay in samples of output time from current sample time	*/
        Real xr[100000],xi[100000];	/* recent data samples	*/
        Real rectafr[70000];		/* real part of rect running AF	*/
        Real rectafi[70000];		/* imag part of rect running AF	*/
        Real rectafm2[70000];	/* rect |AF|^2	*/
        Real polafm2[70000];		/* polar |AF|^2	*/
        Real rectrotr[70000];	/* real part of rect AF phase shift	*/
        Real rectroti[70000];	/* imag part of rect AF phase shift	*/
        Real req[70000];		/* rad corresp. to rect coord	*/
        Real pheq[70000];		/* phi corresp. to rect coord	*/
        Real plag[70000];		/* lag index at polar AF sample points	*/
        Real ptheta[70000];		/* theta index at polar AF sample points	*/
        Real sigma[100000];		/* optimal kernel spreads	*/
        Real rar[100000],rai[100000];	/* poles for running FFTs for rect AF	*/
        Real rarN[70000];		/* poles for running FFTs for rect AF	*/
        Real raiN[70000];
        Real tfslicer[100000];		/* freq slice at current time	*/
        Real tfslicei[100000];

    public:
        AOK(int windowSize, int hopSize, int fftSize, Real normVolume = 3.0);

        ~AOK();

        void setup();

        void process(const MatrixXC& frames, MatrixXR* timeFreqRep);

        void reset();

        int frameSize() const;

        int fftSize() const;

        protected:
        void fft(int n, int m, Real x[], Real y[]);

        int po2(int n);

        int power(int x, int n);

        void kfill(int len, Real k, Real x[]);

        void cshift(int len, Real x[]);

        Real cmr(Real xr, Real xi, Real yr, Real yi);

        Real cmi(Real xr, Real xi, Real yr, Real yi);

        Real ccmr(Real xr, Real xi, Real yr, Real yi);

        Real ccmi(Real xr, Real xi, Real yr, Real yi);

        void rectamake(int nlag, int n, Real forget, Real ar[], Real ai[], Real arN[], Real aiN[]);

        void pthetamake(int nrad, int nphi, int ntheta, Real ptheta[], int maxrad[]);

        void plagmake(int nrad, int nphi, int nlag, Real plag[]);

        void rectopol(int nraf, int nlag, int nrad, int nphi, Real req[], Real pheq[]);

        void rectrotmake(int nraf, int nlag, Real outdelay, Real rectrotr[], Real rectroti[]);

        void rectaf(Real xr[], Real xi[] , int laglen, int freqlen,  Real alphar[], Real alphai[], Real alpharN[], Real alphaiN[], Real afr[], Real afi[]);

        void polafint(int nrad, int nphi, int ntheta, int maxrad[], int nlag, Real plag[], Real ptheta[], Real rectafm2[], Real polafm2[]);

        Real mklag(int nrad, int nphi, int nlag, int iphi, int jrad);

        Real rectkern(int itau, int itheta, int ntheta, int nphi, Real req[], Real pheq[], Real sigma[]);

        void sigupdate(int nrad, int nphi, int nits, Real vol, Real mu0, int maxrad[], Real polafm2[], Real sigma[]);

        void mkmag2(int tlen, Real xr[], Real xi[], Real xm2[]);
    };


/****************************************************************/
/*                                                              */
/*              aok4.c           (Version 4.0)                  */
/*                                                              */
/*              Adaptive Optimal-Kernel (AOK)                   */
/*              Time-Frequency Representation                   */
/*                                                              */
/*              Douglas L. Jones (author)                       */
/*                University of Illinois                        */
/*                E-mail: jones@uicsl.csl.uiuc.edu              */
/*                                                              */
/*              Richard G. Baraniuk (facilitator)               */
/*                Rice University                               */
/*                E-mail: richb@rice.edu                        */
/*                                                              */
/*      Written:        December 26, 1991  (version 1.0)        */
/*      Modified:       June 10, 1992      (version 2.0)        */
/*                      November 20, 1992  (version 3.0)        */
/*                      February 8, 1992   (version 4.0)        */
/*                      January 28, 1996   (renamed aok)        */
/*                                                              */
/****************************************************************/
/*                                                              */
/*      This version interpolates the polar STAF from           */
/*      the rectangular STAF.  It implicitly applies a          */
/*      rectangular window to the data to create the            */
/*      short-time ambiguity function, and includes             */
/*      all non-zero time lags in the STAF.                     */
/*                                                                
*-----------------------------------------------------------------
* Copyright (C) 1992, 1993, 1994, 1995, 1996 the Board of Trustees of
* the University of Illinois.  All Rights Reserved.  Permission is   
* hereby given to use, copy, modify, and distribute this software    
* provided that (1) the headers, copyright and proprietary notices are
* retained in each copy and (2) any files that are modified are       
* identified as such (see below).  The University of Illinois makes no
* representations or warranties of any kind concerning this software or
* its use.                                                             
*                                                                      
* Any modifications made to this file must be commented and dated      
* in the following style:                                              
*                                                                      
*  Source file:         aok4.c                                         
*  Modifications:       Richard Baraniuk, November 25, 1992            
*                         Inserted this sample edit history            
*                       Douglas Jones, February 8, 1993                
*                         Implemented gradient-project algorithm       
*                         in terms of sigma rather than sigma^2        
*                       Richard Baraniuk, January 29, 1996             
*                         Renamed runrgk --> aok                       
*                       Ricard Marxer, March 30, 2009
*                         Adapt to work in Loudia
*                         
*       Please log any further modifications made to this file:        
*                                                                      
*---------------------------------------------------------------*/
AOK::AOK(int windowSize, int hopSize, int fftSize, Real normVolume) : 
  _windowSize( windowSize ), 
  _hopSize( hopSize ), 
  _fftSize( fftSize ), 
  _normVolume( normVolume )
{
  setup();
}

AOK::~AOK() {
  // TODO: Here we should free the buffers
  // but I don't know how to do that with MatrixXR and MatrixXR
  // I'm sure Nico will...
}


void AOK::setup(){
  // Prepare the buffers
  LOUDIA_DEBUG("AOK: Setting up...");

  tstep = _hopSize;
  tlag = _windowSize;
  fftlen = _fftSize;
  
  if ( fftlen < (2*tlag) )
    {
      fstep = 2*tlag/fftlen;
      fftlen = 2*tlag;
    }
  else
    {
      fstep = 1;
    }
  
  vol = _normVolume;

  /*	tlag = 64; */		/* total number of rectangular AF lags	*/
  nits = (int) log2((Real) tstep+2);	/* number of gradient steps to take each time	*/
  /*	nits = 2; */
  /*	vol = 2.0; */		/* kernel volume (1.0=Heisenberg limit)	*/
  mu = 0.5;		/* gradient descent factor	*/
  
  forget = 0.001;		/* set no. samples to 0.5 weight on running AF	*/
  nraf = tlag;		/* theta size of rectangular AF	*/
  nrad = tlag;		/* number of radial samples in polar AF	*/
  nphi = tlag;		/* number of angular samples in polar AF */
  outdelay = tlag/2;	/* delay in effective output time in samples	*/
  /* nlag-1 < outdelay < nraf to prevent "echo" effect */

  nlag = tlag + 1;	/* one-sided number of AF lags	*/
  mfft = po2(fftlen);
  slen = (int)(1.42*(nlag-1) + nraf + 3);	/* number of delayed samples to maintain	*/
  
  pi = 3.141592654;
  vol = (2.0*vol*nphi*nrad*nrad)/(pi*tlag);	/* normalize volume	*/
  
  LOUDIA_DEBUG("AOK: tlen: " << tlen);
  LOUDIA_DEBUG("AOK: nlag: " << nlag);
  LOUDIA_DEBUG("AOK: slen: " << slen);
  LOUDIA_DEBUG("AOK: xlen: " << xlen);
  LOUDIA_DEBUG("AOK: fftlen: " << fftlen);
  LOUDIA_DEBUG("AOK: nrad: " << nrad);
  LOUDIA_DEBUG("AOK: nraf: " << nraf);
  LOUDIA_DEBUG("AOK: nlag: " << nlag);

  kfill((nrad * nphi), 0.0, polafm2);
  kfill((nraf * nlag), 0.0, rectafr);
  kfill((nraf * nlag), 0.0, rectafi);
  kfill(slen, 0.0, xr);
  kfill(slen, 0.0, xi);
  kfill(nphi, 1.0, sigma);
  
  //tlen = xlen + nraf + 2;

  rectamake(nlag, nraf, forget, rar, rai, rarN, raiN);/* make running rect AF parms	*/
  plagmake(nrad, nphi, nlag, plag);
  pthetamake(nrad, nphi, nraf, ptheta, maxrad);	/* make running polar AF parms	*/
  rectrotmake(nraf, nlag, outdelay, rectrotr, rectroti);
  rectopol(nraf, nlag, nrad, nphi, req, pheq);
  
  reset();

  LOUDIA_DEBUG("AOK: Finished set up...");
}


void AOK::process(const MatrixXC& frames, MatrixXR* timeFreqRep){
  outct = 0;

  timeFreqRep->resize(frames.rows() / tstep, _fftSize);
  
  MatrixXC framesFlipped = frames;

  // fliplr the framesFlipped
  for(int i = 0; i < slen / 2; i++){
    framesFlipped.col(i).swap(framesFlipped.col(slen - 1 - i));
  }

  for ( int row = 0; row < framesFlipped.rows(); row++) {  /*  for each temporal frame of samples  */
    //DEBUG("AOK: Processing, row="<<row);
    // Fill in the input vectors
    //DEBUG("AOK: Processing, setting the xr and xi C arrays, ii="<<ii);
    Eigen::Map<MatrixXR>(xr, 1, framesFlipped.cols()) = framesFlipped.row(row).real();
    Eigen::Map<MatrixXR>(xi, 1, framesFlipped.cols()) = framesFlipped.row(row).imag();
    //DEBUG("AOK: Processing, finished setting the xr and xi C arrays");
    
    rectaf(xr, xi, nlag, nraf, rar, rai, rarN, raiN, rectafr, rectafi);

    if ( (row % tstep) == 0 )	/* output t-f slice	*/
      {
        mkmag2((nlag*nraf), rectafr, rectafi, rectafm2);
        polafint(nrad, nphi, nraf, maxrad, nlag, plag, ptheta, rectafm2, polafm2);
        sigupdate(nrad, nphi, nits, vol, mu, maxrad, polafm2, sigma);
        
        for (int i=0; i < nlag-1; i++)	/* for each tau	*/
          {
            tfslicer[i] = 0.0;
            tfslicei[i] = 0.0;
            
            
            for (int j = 0; j < nraf; j++)	/* integrate over theta	*/
              {
                  rtemp = ccmr(rectafr[i*nraf+j], rectafi[i*nraf+j], rectrotr[i*nraf+j], rectroti[i*nraf+j]);
                  rtemp1 = ccmi(rectafr[i*nraf+j], rectafi[i*nraf+j], rectrotr[i*nraf+j], rectroti[i*nraf+j]);
                  
                  rtemp2 = rectkern(i, j, nraf, nphi, req, pheq, sigma);
                  tfslicer[i] = tfslicer[i] + rtemp*rtemp2;
                  tfslicei[i] = tfslicei[i] + rtemp1*rtemp2;
                  /*	fprintf(ofp," %d , %d , %g, %g, %g , %g , %g \n", i,j,rectafr[i*nraf+j],rectafi[i*nraf+j],rtemp,rtemp1,rtemp2); */
              }
          }
        for (int i=nlag-1; i < (fftlen-nlag+2); i++)	/* zero pad for FFT	*/
          {
            tfslicer[i] = 0.0;
            tfslicei[i] = 0.0;
          }
        
        for (int i=(fftlen-nlag+2); i < fftlen; i++)	/* fill in c.c. symmetric half of array	*/
          {
              tfslicer[i] = tfslicer[fftlen - i];
              tfslicei[i] = - tfslicei[fftlen - i];
          }
        
        
        
        fft(fftlen, mfft, tfslicer, tfslicei);
        /*
          LOUDIA_DEBUG("AOK: Processing, tfslicer: ");
          for(int b=0; b<fftlen; b++ ){
          LOUDIA_DEBUG("AOK: Processing, tfslicer["<<b<<"]="<<tfslicer[b]);
          }
        */
        itemp = fftlen/2 + fstep;
        int col = 0;				/* print output slice	*/
        //DEBUG("AOK: Processing, timeFreqRep->shape: " << timeFreqRep->rows() << ", " << timeFreqRep->cols());
        for (int i=itemp; i < fftlen; i=i+fstep)
          {
            //DEBUG("AOK: Processing, row: " << row << ", col: " << col << ", i: " << i);
            (*timeFreqRep)(outct, col) = tfslicer[i];
            col++;
          }
        for (int i=0; i < itemp; i=i+fstep)
          {
            //DEBUG("AOK: Processing, row: " << row << ", col: " << col << ", i: " << i);
            (*timeFreqRep)(outct, col) = tfslicer[i];
            col++;
          }

        outct = outct + 1;        
      }
  }
}

void AOK::reset(){
  // Initial values
}

int AOK::frameSize() const{
  return (int)(2.42 * _windowSize + 3);
}

int AOK::fftSize() const{
  return _fftSize;
}


/****************************************************************/
/*		fft.c						*/
/*		Douglas L. Jones				*/
/*		University of Illinois at Urbana-Champaign	*/
/*		January 19, 1992				*/
/*								*/
/*   fft: in-place radix-2 DIT DFT of a complex input		*/
/*								*/
/*   input:							*/
/*	n:	length of FFT: must be a power of two		*/
/*	m:	n = 2**m					*/
/*   input/output						*/
/*	x:	Real array of length n with real part of data	*/
/*	y:	Real array of length n with imag part of data	*/
/*								*/
/*   Permission to copy and use this program is granted		*/
/*   as long as this header is included.			*/
/****************************************************************/
void AOK::fft(int n, int m, Real x[], Real y[])
{
	int	i,j,k,n1,n2;
	Real	c,s,e,a,t1,t2;


	j = 0;				/* bit-reverse	*/
	n2 = n/2;
	for (i=1; i < n - 1; i++)
	 {
	  n1 = n2;
	  while ( j >= n1 )
	   {
	    j = j - n1;
	    n1 = n1/2;
	   }
	  j = j + n1;

	  if (i < j)
	   {
	    t1 = x[i];
	    x[i] = x[j];
	    x[j] = t1;
	    t1 = y[i];
	    y[i] = y[j];
	    y[j] = t1;
	   }
	 }


	n1 = 0;				/* FFT	*/
	n2 = 1;

	for (i=0; i < m; i++)
	 {
	  n1 = n2;
	  n2 = n2 + n2;
	  e = -6.283185307179586/n2;
	  a = 0.0;

	  for (j=0; j < n1; j++)
	   {
	    c = cos(a);
	    s = sin(a);
	    a = a + e;

	    for (k=j; k < n; k=k+n2)
	     {
	      t1 = c*x[k+n1] - s*y[k+n1];
	      t2 = s*x[k+n1] + c*y[k+n1];
	      x[k+n1] = x[k] - t1;
	      y[k+n1] = y[k] - t2;
	      x[k] = x[k] + t1;
	      y[k] = y[k] + t2;
	     }
	   }
	 }
	
	    
	return;
}
/*								*/
/*   po2: find the smallest power of two >= input value		*/
/*								*/
int	AOK::po2(int n)
{
	int	m, mm;

	mm = 1;
	m = 0;
	while (mm < n) {
	   ++m;
	   mm = 2*mm;
	}

	return(m);
}
/*								*/
/*   power: compute x^n, x and n positve integers		*/
/*								*/
int AOK::power(int x, int n)
{
	int	i,p;

	p = 1;
	for (i=1; i<=n; ++i)
	     p = p*x;
	return(p);
}
/*								*/
/*   zerofill: set array elements to constant			*/
/*								*/
void AOK::kfill(int len,Real k,Real x[])
{
	int	i;

	for (i=0; i < len; i++)
	  x[i] = k;

	return;
}
/*								*/
/*   cshift: circularly shift an array				*/
/*								*/
void AOK::cshift(int len, Real x[])
{
	int	i;
	Real	rtemp;


	rtemp = x[len-1];

	for (i=len-1; i > 0; i--)
	  x[i] = x[i-1];

	x[0] = rtemp;


	return;
}
/*								*/
/*   cmr: computes real part of x times y			*/
/*								*/
Real	AOK::cmr(Real xr, Real xi, Real yr, Real yi)
{
	Real	rtemp;

	rtemp = xr*yr - xi*yi;

	return(rtemp);
}
/*								*/
/*   cmi: computes imaginary part of x times y			*/
/*								*/
Real	AOK::cmi(Real xr, Real xi, Real yr, Real yi)
{
	Real	rtemp;

	rtemp = xi*yr + xr*yi;

	return(rtemp);
}
/*								*/
/*   ccmr: computes real part of x times y*			*/
/*								*/
Real	AOK::ccmr(Real xr, Real xi, Real yr, Real yi)
{
	Real	rtemp;

	rtemp = xr*yr + xi*yi;

	return(rtemp);
}
/*								*/
/*   ccmi: computes imaginary part of x times y*		*/
/*								*/
Real	AOK::ccmi(Real xr, Real xi, Real yr, Real yi)
{
	Real	rtemp;

	rtemp = xi*yr - xr*yi;

	return(rtemp);
}
/*								*/
/*   rectamake: make vector of poles for rect running AF	*/
/*								*/
void AOK::rectamake(int nlag, int n, Real forget, Real ar[], Real ai[], Real arN[], Real aiN[])
{
	int	i,j;
	Real	trig,decay;
	Real	trigN,decayN;


	trig = 6.283185307/n;
	decay = exp(-forget);

	for (j=0; j < n; j++)
	 {
	  ar[j] = decay*cos(trig*j);
	  ai[j] = decay*sin(trig*j);
	 }

	for (i=0; i < nlag; i++)
	 {
	  trigN = 6.283185307*(n-i);
	  trigN = trigN/n;
	  decayN = exp(-forget*(n-i));

	  for (j=0; j < n; j++)
	   {
	    arN[i*n+j] = decayN*cos(trigN*j);
	    aiN[i*n+j] = decayN*sin(trigN*j);
	   }
	 }


	return;
}
/*								*/
/*   pthetamake: make matrix of theta indices for polar samples	*/
/*								*/
void AOK::pthetamake(int nrad, int nphi, int ntheta, Real ptheta[], int maxrad[])
{
	int	i,j;
	Real	theta,rtemp,deltheta;


	deltheta = 6.283185307/ntheta;

	for (i=0; i < nphi; i++)	/* for all phi ...	*/
	 {
	  maxrad[i] = nrad;

	  for (j = 0; j < nrad; j++)	/* and all radii	*/
	   {
	    theta = - ((4.442882938/nrad)*j)*cos((3.141592654*i)/nphi);
	    if ( theta >= 0.0 )
	       {
	        rtemp = theta/deltheta;
		if ( rtemp > (ntheta/2 - 1) )
		 {
		  rtemp = -1.0;
	          if (j < maxrad[i])  maxrad[i] = j;
		 }
	       }
	      else
	       {
	        rtemp = (theta + 6.283185307)/deltheta;
		if ( rtemp < (ntheta/2 + 1) )
		 {
		  rtemp = -1.0;
	          if (j < maxrad[i])  maxrad[i] = j;
		 }
	       }
		
	    ptheta[i*nrad+j] = rtemp;
	   }
	 }


	return;
}
/*								*/
/*   plagmake: make matrix of lags for polar running AF		*/
/*								*/
void AOK::plagmake(int nrad, int nphi, int nlag, Real plag[])
{
	int	i,j;


	for (i=0; i < nphi; i++)        /* for all phi ...      */
	 {
	  for (j = 0; j < nrad; j++)    /* and all radii        */
	   {
	    plag[i*nrad+j] = mklag(nrad,nphi,nlag,i,j);
	   }
	 }


	return;
}
/*								*/
/*   rectopol: find polar indices corresponding to rect samples	*/
/*								*/
void AOK::rectopol(int nraf, int nlag, int nrad, int nphi, Real req[], Real pheq[])
{
	int	i,j,jt;
	Real	pi,deltau,deltheta,delrad,delphi;


	pi = 3.141592654;

	deltau = sqrt(pi/(nlag-1));
	deltheta = 2.0*sqrt((nlag-1)*pi)/nraf;
	delrad = sqrt(2.0*pi*(nlag-1))/nrad;
	delphi = pi/nphi;

	for (i=0; i < nlag; i++)
	 {
	  for (j=0; j < nraf/2; j++)
	   {
	    req[i*nraf +j] = sqrt(i*i*deltau*deltau + j*j*deltheta*deltheta)/delrad;
	    if ( i == 0 )
	      pheq[i*nraf +j] = 0.0;
	    else pheq[i*nraf +j] = (atan((j*deltheta)/(i*deltau)) + 1.570796327)/delphi;
	   }

	  for (j=0; j < nraf/2; j++)
	   {
	    jt = j - nraf/2;
	    req[i*nraf + nraf/2 + j]  = sqrt(i*i*deltau*deltau + jt*jt*deltheta*deltheta)/delrad;
	    if ( i == 0 )
	      pheq[i*nraf + nraf/2 + j] = 0.0;
	    else pheq[i*nraf + nraf/2 + j] = (atan((jt*deltheta)/(i*deltau)) + 1.570796327)/delphi;
	   }
	 }


	return;
}
/*								*/
/*   rectrotmake: make array of rect AF phase shifts		*/
/*								*/
void AOK::rectrotmake(int nraf, int nlag, Real outdelay, Real rectrotr[], Real rectroti[])
{
	int	i,j;
	Real	twopin;

	twopin = 6.283185307/nraf;


	for (i=0; i < nlag; i++)
	 {
	  for (j=0; j < nraf/2; j++)
	   {
	    rectrotr[i*nraf+j] = cos( (twopin*j)*(outdelay - ((Real) i)/2.0 ) );
	    rectroti[i*nraf+j] = sin( (twopin*j)*(outdelay - ((Real) i)/2.0 ) );
	   }
	  for (j=nraf/2; j < nraf; j++)
	   {
	    rectrotr[i*nraf+j] = cos( (twopin*(j-nraf))*(outdelay - ((Real) i)/2.0 ) );
	    rectroti[i*nraf+j] = sin( (twopin*(j-nraf))*(outdelay - ((Real) i)/2.0 ) );
	   }
	 }


	return;
}
/*								*/
/*   rectaf: generate running AF on rectangular grid;		*/
/*	     negative lags, all DFT frequencies			*/
/*								*/
void AOK::rectaf(Real xr[], Real xi[] , int laglen, int freqlen,  Real alphar[], Real alphai[], Real alpharN[], Real alphaiN[], Real afr[], Real afi[])
{
	int	i,j;
	Real	rtemp,rr,ri,rrN,riN;

	for (i=0; i < laglen; i++)
	 {
	  rr = ccmr(xr[0],xi[0],xr[i],xi[i]);
	  ri = ccmi(xr[0],xi[0],xr[i],xi[i]);

	  rrN = ccmr(xr[freqlen-i],xi[freqlen-i],xr[freqlen],xi[freqlen]);
	  riN = ccmi(xr[freqlen-i],xi[freqlen-i],xr[freqlen],xi[freqlen]);

	  for (j = 0; j < freqlen; j++)
	   {
	    rtemp = cmr(afr[i*freqlen+j],afi[i*freqlen+j],alphar[j],alphai[j]) - cmr(rrN,riN,alpharN[i*freqlen+j],alphaiN[i*freqlen+j]) + rr;
	    afi[i*freqlen + j] = cmi(afr[i*freqlen+j],afi[i*freqlen+j],alphar[j],alphai[j]) - cmi(rrN,riN,alpharN[i*freqlen+j],alphaiN[i*freqlen+j]) + ri;
	    afr[i*freqlen + j] = rtemp;
	   }
	 }


	return;
}
/*								*/
/*   polafint: interpolate AF on polar grid;			*/
/*								*/
void AOK::polafint(int nrad, int nphi, int ntheta, int maxrad[], int nlag, Real plag[], Real ptheta[], Real rectafm2[], Real polafm2[])
{
	int	i,j;
	int	ilag,itheta,itheta1;
	Real	rlag,rtheta,rtemp,rtemp1;


	for (i=0; i < nphi/2; i++)	/* for all phi ...	*/
	 {
	  for (j = 0; j < maxrad[i]; j++)	/* and all radii with |theta| < pi */
	   {
	    ilag = (int) plag[i*nrad+j];
	    rlag = plag[i*nrad+j] - ilag;

	    if ( ilag >= nlag )
	       {
		polafm2[i*nrad+j] = 0.0;
	       }
	      else
	       {
		itheta = (int) ptheta[i*nrad+j];
		rtheta = ptheta[i*nrad+j] - itheta;

		itheta1 = itheta + 1;
		if ( itheta1 >= ntheta )  itheta1 = 0;

		rtemp =  (rectafm2[ilag*ntheta+itheta1] - rectafm2[ilag*ntheta+itheta])*rtheta + rectafm2[ilag*ntheta+itheta];
		rtemp1 =  (rectafm2[(ilag+1)*ntheta+itheta1] - rectafm2[(ilag+1)*ntheta+itheta])*rtheta + rectafm2[(ilag+1)*ntheta+itheta];
		polafm2[i*nrad+j] = (rtemp1-rtemp)*rlag + rtemp;
	       }
	   }
	 }


	for (i=nphi/2; i < nphi; i++)	/* for all phi ...	*/
	 {
	  for (j = 0; j < maxrad[i]; j++)	/* and all radii with |theta| < pi */
	   {
	    ilag = (int) plag[i*nrad+j];
	    rlag = plag[i*nrad+j] - ilag;

	    if ( ilag >= nlag )
	       {
		polafm2[i*nrad+j] = 0.0;
	       }
	      else
	       {
		itheta = (int) ptheta[i*nrad+j];
		rtheta = ptheta[i*nrad+j] - itheta;

		rtemp =  (rectafm2[ilag*ntheta+itheta+1] - rectafm2[ilag*ntheta+itheta])*rtheta + rectafm2[ilag*ntheta+itheta];
		rtemp1 =  (rectafm2[(ilag+1)*ntheta+itheta+1] - rectafm2[(ilag+1)*ntheta+itheta])*rtheta + rectafm2[(ilag+1)*ntheta+itheta];
		polafm2[i*nrad+j] = (rtemp1-rtemp)*rlag + rtemp;
	       }
	   }
	 }


	return;
}
/*								*/
/*   mklag: compute radial sample lag				*/
/*								*/
Real	AOK::mklag(int nrad, int nphi, int nlag, int iphi, int jrad)
{
	Real	delay;

	delay = ((1.414213562*(nlag-1)*jrad)/nrad)*sin((3.141592654*iphi)/nphi);


	return(delay);
}
/*								*/
/*   rectkern: generate kernel samples on rectangular grid	*/
/*								*/
Real	AOK::rectkern(int itau, int itheta, int ntheta, int nphi, Real req[], Real pheq[], Real sigma[])
{
	int	iphi,iphi1;
	Real	kern,tsigma;


	iphi = (int) pheq[itau*ntheta + itheta];
	iphi1 = iphi + 1;
	if (iphi1 > (nphi-1))  iphi1 = 0;
	tsigma = sigma[iphi] + (pheq[itau*ntheta + itheta] - iphi)*(sigma[iphi1]-sigma[iphi]);

/*  Tom Polver says on his machine, exp screws up when the argument of */
/*  the exp function is too large */
	kern = exp(-req[itau*ntheta+itheta]*req[itau*ntheta+itheta]/(tsigma*tsigma));

	return(kern);
}
/*								*/
/*   sigupdate: update RG kernel parameters			*/
/*								*/
void AOK::sigupdate(int nrad, int nphi, int nits, Real vol, Real mu0, int maxrad[], Real polafm2[], Real sigma[])
{
	int	ii,i,j;
	Real	grad[1024],gradsum,gradsum1,tvol,volfac,eec,ee1,ee2,mu;


	for (ii=0; ii < nits; ii++)
	 {
	  gradsum = 0.0;
	  gradsum1 = 0.0;

	  for (i=0; i < nphi; i++)
	   {
	    grad[i] = 0.0;

	    ee1 = exp( - 1.0/(sigma[i]*sigma[i]) );	/* use Kaiser's efficient method */
	    ee2 = 1.0;
	    eec = ee1*ee1;

	    for (j=1; j < maxrad[i]; j++)
	     {
	      ee2 = ee1*ee2;
	      ee1 = eec*ee1;

	      grad[i] = grad[i] + j*j*j*ee2*polafm2[i*nrad+j];
	     }
	    grad[i] = grad[i]/(sigma[i]*sigma[i]*sigma[i]);

	    gradsum = gradsum + grad[i]*grad[i];
	    gradsum1 = gradsum1 + sigma[i]*grad[i];
	   }

	  gradsum1 = 2.0*gradsum1;
	  if ( gradsum < 0.0000001 )  gradsum = 0.0000001;
	  if ( gradsum1 < 0.0000001 )  gradsum1 = 0.0000001;

	  mu = ( sqrt(gradsum1*gradsum1 + 4.0*gradsum*vol*mu0) - gradsum1 ) / ( 2.0*gradsum );


	  tvol = 0.0;

	  for (i=0; i < nphi; i++)
	   {
	    sigma[i] = sigma[i] + mu*grad[i];
	    if (sigma[i] < 0.5)  sigma[i] = 0.5;
/*	    printf("sigma[%d] = %g\n", i,sigma[i]); */
	    tvol = tvol + sigma[i]*sigma[i];
	   }

	  volfac = sqrt(vol/tvol);
	  for (i=0; i < nphi; i++)  sigma[i] = volfac*sigma[i];
	 }


	return;
}
/*								*/
/*   mkmag2: compute squared magnitude of an array		*/
/*								*/
void AOK::mkmag2(int tlen, Real xr[], Real xi[], Real xm2[])
{
	int	i;


	for (i=0; i < tlen; i++)
	 {
	  xm2[i] = xr[i]*xr[i] + xi[i]*xi[i];
	 }

	return;
}

}