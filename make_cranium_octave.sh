gcc -Wfatal-errors -fmax-errors=1 -I/usr/local/include/octave-7.3.0 -IOctave -o cranium_octave \
Cranium/cranium_octave.cpp -lstdc++ -lm -L/usr/local/lib/octave/7.3.0 -loctave -loctinterp
