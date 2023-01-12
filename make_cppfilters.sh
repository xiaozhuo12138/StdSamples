swig -lua -c++ -Iinclude dsp_cppfilters.i
gcc -Iinclude -O2 -fPIC -march=native -mavx2 -shared -o cppfilters.so dsp_cppfilters_wrap.cxx -lstdc++ -lm -lluajit
