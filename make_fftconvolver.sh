swig -lua -c++ -Iinclude/AudioFFT dsp_fftconvolver.i
gcc -DAUDIOFFT_FFTW3 -Iinclude/AudioFFT -O2 -fPIC -march=native -mavx2 -shared -o fftconvolver.so dsp_fftconvolver_wrap.cxx lib/libfftconvolver.a -lstdc++ -lm -lluajit -lfftw3 -lfftw3f
