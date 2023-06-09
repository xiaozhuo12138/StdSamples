#pragma once

#include "synthlite.h"
#include "CFilter.hpp"
#include "CVAOnePoleFilter.hpp"

#include <cstdint>
#include <cmath>

class CDiodeLadderFilter : public CFilter
{
public:
	CDiodeLadderFilter(void);
	~CDiodeLadderFilter(void);

	CVAOnePoleFilter m_LPF1;
	CVAOnePoleFilter m_LPF2;
	CVAOnePoleFilter m_LPF3;
	CVAOnePoleFilter m_LPF4;

	// variables
	DspFloatType m_dK;			// K, set with Q
	DspFloatType m_dGamma;		// needed for final calc and update
	DspFloatType m_dSG1; 
	DspFloatType m_dSG2; 
	DspFloatType m_dSG3; 
	DspFloatType m_dSG4; 

	// -- CFilter Overrides --
	virtual void reset();
	virtual void setQControl(DspFloatType dQControl);
	virtual void update();

	inline virtual DspFloatType doFilter(DspFloatType xn)
	{
		// --- return xn if filter not supported
		if(m_uFilterType != LPF4)
			return xn;

		m_LPF4.setFeedback(0.0);
		m_LPF3.setFeedback(m_LPF4.getFeedbackOutput());
		m_LPF2.setFeedback(m_LPF3.getFeedbackOutput());
		m_LPF1.setFeedback(m_LPF2.getFeedbackOutput());

		// --- form input
		DspFloatType dSigma = m_dSG1*m_LPF1.getFeedbackOutput() + 
						m_dSG2* m_LPF2.getFeedbackOutput() +
						m_dSG3*m_LPF3.getFeedbackOutput() +
						m_dSG4* m_LPF4.getFeedbackOutput();

		// --- for passband gain compensation!
		xn *= 1.0 + m_dAuxControl*m_dK;

		// --- form input
		DspFloatType dU = (xn - m_dK*dSigma)/(1 + m_dK*m_dGamma);

		// ---NLP
		if(m_uNLP == ON)
			dU = fasttanh(m_dSaturation*dU);

		// --- cascade of four filters
		return m_LPF4.doFilter(m_LPF3.doFilter(m_LPF2.doFilter(m_LPF1.doFilter(dU))));
	}
};

CDiodeLadderFilter::CDiodeLadderFilter(void)
{
	// init
	m_dK = 0;		
	m_dGamma = 0.0;

	// feedback scalars
	m_dSG1 = 0.0; 
	m_dSG2 = 0.0;  
	m_dSG3 = 0.0;  
	m_dSG4 = 0.0;  

	// set filter types
	m_LPF1.m_uFilterType = LPF1;
	m_LPF2.m_uFilterType = LPF1;
	m_LPF3.m_uFilterType = LPF1;
	m_LPF4.m_uFilterType = LPF1;

	// set default filter type
	m_uFilterType = LPF4;

	// flush everything
	reset();
}

CDiodeLadderFilter::~CDiodeLadderFilter(void)
{
}

void CDiodeLadderFilter::reset()
{
	// flush everything
	m_LPF1.reset();
	m_LPF2.reset();
	m_LPF3.reset();
	m_LPF4.reset();
}

// decode the Q value; Q on UI is 1->10
void CDiodeLadderFilter::setQControl(DspFloatType dQControl)
{
	// this maps dQControl = 1->10 to K = 0 -> 17
	m_dK = (17.0)*(dQControl - 1.0)/(10.0 - 1.0);
}

void CDiodeLadderFilter::update()
{
	// base class does modulation
	CFilter::update();

	// calculate alphas
	DspFloatType wd = 2*M_PI*m_dFc;          
	DspFloatType T  = 1/m_dSampleRate;             
	DspFloatType wa = (2/T)*tan(wd*T/2); 
	DspFloatType g = wa*T/2;  

	// Big G's
	DspFloatType G1, G2, G3, G4;

	G4 = 0.5*g/(1.0 + g);
	G3 = 0.5*g/(1.0 + g - 0.5*g*G4);
	G2 = 0.5*g/(1.0 + g - 0.5*g*G3);
	G1 = g/(1.0 + g - g*G2);
	
	m_dGamma = G4*G3*G2*G1;
	
	m_dSG1 =  G4*G3*G2; 
	m_dSG2 =  G4*G3; 
	m_dSG3 =  G4; 
	m_dSG4 =  1.0; 

	// set alphas
	DspFloatType G = g/(1.0 + g);

	m_LPF1.m_dAlpha = G;
	m_LPF2.m_dAlpha = G;
	m_LPF3.m_dAlpha = G;
	m_LPF4.m_dAlpha = G;

	// set betas
	m_LPF1.m_dBeta = 1.0/(1.0 + g - g*G2);
	m_LPF2.m_dBeta = 1.0/(1.0 + g - 0.5*g*G3);
	m_LPF3.m_dBeta = 1.0/(1.0 + g - 0.5*g*G4);
	m_LPF4.m_dBeta = 1.0/(1.0 + g);

	// set deltas
	m_LPF1.m_dDelta = g;
	m_LPF2.m_dDelta = 0.5*g;
	m_LPF3.m_dDelta = 0.5*g;
	m_LPF4.m_dDelta = 0.0;

	m_LPF1.m_dGamma = 1.0 + G1*G2;
	m_LPF2.m_dGamma = 1.0 + G2*G3;
	m_LPF3.m_dGamma = 1.0 + G3*G4;
	m_LPF4.m_dGamma = 1.0;

	// set epsilons
	m_LPF1.m_dEpsilon = G2;
	m_LPF2.m_dEpsilon = G3;
	m_LPF3.m_dEpsilon = G4;
	m_LPF4.m_dEpsilon = 0.0;

	// set a0s
	m_LPF1.m_da0 = 1.0;
	m_LPF2.m_da0 = 0.5;
	m_LPF3.m_da0 = 0.5;
	m_LPF4.m_da0 = 0.5;
}
