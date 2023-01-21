%module Functions
%{
typedef float DspFloatType;
#include "SoundObject.hpp"

#include <cassert>
#include <random>
#include <chrono>
#include <complex>
#include <cstdio>
#include <cstring>
#include <cstdlib>

#include "FX/LFO.hpp"
#include "FX/FourierWave.hpp"
#include "FX/FunctionGenerator.hpp"
#include "FX/FunctionLFO.hpp"
#include "FX/Functions.hpp"
//#include "FX/LowFrequencyOscillator.hpp"
#include "FX/WaveTable.hpp"
//#include "FX/Noise.h"
#include "FX/NoiseGenerator.h"
#include "FX/NoiseGenerators.hpp"
#include "FX/Noisey.hpp"



%}
typedef float DspFloatType;
%include "stdint.i"
%include "std_math.i"
%include "std_vector.i"
%include "std_list.i"
%include "std_map.i"
%include "lua_fnptr.i"

%include "SoundObject.hpp"

%include "FX/LFO.hpp"
%include "FX/FunctionGenerator.hpp"
%include "FX/FunctionLFO.hpp"
%include "FX/Functions.hpp"
//%include "FX/LowFrequencyOscillator.hpp"
%include "FX/WaveTable.hpp"
//%include "FX/Noise.h"
%include "FX/NoiseGenerator.h"
%include "FX/NoiseGenerators.hpp"
%include "FX/Noisey.hpp"

%template(float_vector) std::vector<float>;
%template(double_vector) std::vector<double>;

%template(complex_float_vector) std::vector<std::complex<float>>;
%template(complex_double_vector) std::vector<std::complex<double>>;

%inline %{
    const int BufferSize = 256;
    Std::RandomMersenne noise;
    DspFloatType sampleRate = 44100.0f;
    DspFloatType inverseSampleRate = 1 / 44100.0f;
    DspFloatType invSampleRate = 1 / 44100.0f;
%}
