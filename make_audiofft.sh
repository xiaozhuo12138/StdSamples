swig -lua -c++ -Iinclude/AudioFFT dsp_audiofft.i
gcc -DAUDIOFFT_FFTW3 -Iinclude/AudioFFT -O2 -march=native -fPIC -mavx2 -shared -o audiofft.so dsp_audiofft_wrap.cxx include/AudioFFT/AudioFFT.cpp -lstdc++ -lm -lluajit -lfftw3 -lfftw3f
