swig -lua -I/usr/local/include -Iinclude AudioMidi/audiosystem.i
gcc -fmax-errors=1 -fpermissive -I/usr/local/include -Iinclude  -O2 -march=native -mavx2 -fPIC -shared \
-o audiosystem.so AudioMidi/audiosystem_wrap.c -pthread -lrt -lm -lluajit -L/usr/local/lib -lportaudio -lportmidi
