swig -lua -c++ -Iinclude/ fmsynth.i
gcc -fmax-errors=1 -Iinclude -O2 -fPIC -march=native -mavx2 -shared -o fmsynth.so fmsynth_wrap.cxx lib/libfmsynth.a -lstdc++ -lm -lluajit
