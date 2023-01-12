require('kfr2')
kaiser = kfr2.window_kaiser_ptr(127,3.0)
fmt = kfr2.audio_format()
wav = kfr2.wav_load('Data/redwheel.wav',fmt)
out = kfr2.fir_lowpass(wav,127,200.0/fmt.samplerate,kaiser,true)
kfr2.wav_save(out,"testfir.wav",fmt.channels, fmt.samplerate)