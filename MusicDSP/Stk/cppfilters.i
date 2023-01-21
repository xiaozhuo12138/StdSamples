%module cppfilters 
%{
//#include "filter_includes.h"
#include "cppfilters.hpp"
%}

%include "std_vector.i"
%template(float_vector) std::vector<float>;
%template(double_vector) std::vector<double>;
%template(ldouble_vector) std::vector<long double>;

%include "cppfilters.hpp"
/*
%include "biquad.h"
%include "biquad_modified.h"
%include "fo_apf.h"
%include "fo_hpf.h"
%include "fo_lpf.h"
%include "fo_shelving_high.h"
%include "fo_shelving_low.h"
%include "so_apf.h"
%include "so_bpf.h"
%include "so_bsf.h"
%include "so_lpf.h"
%include "so_hpf.h"
%include "so_butterworth_bpf.h"
%include "so_butterworth_bsf.h"
%include "so_butterworth_hpf.h"
%include "so_butterworth_lpf.h"
%include "so_linkwitz_riley_hpf.h"
%include "so_linkwitz_riley_lpf.h"
%include "so_parametric_cq_boost.h"
%include "so_parametric_cq_cut.h"
%include "so_parametric_ncq.h"
*/