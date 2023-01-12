swig -lua -c++ -Iinclude fir1.i
gcc -Iinclude -O2 -march=native -mavx2 -fPIC -shared -o fir1.so fir1_wrap.cxx -lstdc++ -lm -lluajit -Lbin -lfir
