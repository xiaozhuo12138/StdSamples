## Process this file with automake to produce Makefile.in

SUBDIRS = include
AM_CPPFLAGS = -I$(ladspa_prefix)/include -Iinclude -I. \
              -DLOCALEDIR=\"$(datadir)/locale\"
LINK = $(COMPILE) -nostartfiles -shared -lc -o $@
CLEANFILES = parabola_1649_data.so.c \
             sawtooth_1641_data.so.c \
             square_1643_data.so.c
MAINTAINERCLEANFILES = Makefile.in

## These are really libs, eh? I have yet to try libtool.
pkgbin_PROGRAMS = adsr_1653.so adsr_1680.so amp_1654.so branch_1673.so \
                  dahdsr_2021.so difference_2030.so fmod_1656.so \
                  interpolator_1660.so lp4pole_1671.so \
                  product_1668.so pulse_1645.so quantiser20_2027.so \
                  quantiser50_2028.so quantiser100_2029.so random_1661.so \
                  ratio_2034.so sawtooth_1641.so sequencer64_1675.so \
                  sequencer32_1676.so sequencer16_1677.so square_1643.so \
                  sum_1665.so sync_pulse_2023.so sync_square_1678.so \
                  tracker_2025.so triangle_1649.so

wavedata_PROGRAMS = sawtooth_1641_data.so \
                    square_1643_data.so \
                    parabola_1649_data.so
nodist_sawtooth_1641_data_so_SOURCES = sawtooth_1641_data.so.c
nodist_square_1643_data_so_SOURCES = square_1643_data.so.c
nodist_parabola_1649_data_so_SOURCES = parabola_1649_data.so.c

pkgbindir = $(ladspa_plugin_dir)
wavedatadir = $(ladspa_plugin_dir)/$(wavedata_subdir)

noinst_PROGRAMS = wavegen
wavegen_SOURCES = wavegen.c wdatutil.c
wavegen_DEPENDENCIES = include/wdatutil.h include/common.h
wavegen_LINK = $(COMPILE) -lc -lm -o $@

## Need to specify sources so that make dist will include them
lp4pole_1671_so_SOURCES = lp4pole_1671.so.c lp4pole_filter.c
pulse_1645_so_SOURCES = pulse_1645.so.c wavedata.c
quantiser20_2027_so_SOURCES = quantiser.so.c
quantiser50_2028_so_SOURCES = quantiser.so.c
quantiser100_2029_so_SOURCES = quantiser.so.c
sawtooth_1641_so_SOURCES = sawtooth_1641.so.c wavedata.c
sequencer64_1675_so_SOURCES = sequencer.so.c
sequencer32_1676_so_SOURCES = sequencer.so.c
sequencer16_1677_so_SOURCES = sequencer.so.c
square_1643_so_SOURCES = square_1643.so.c wavedata.c
triangle_1649_so_SOURCES = triangle_1649.so.c wavedata.c

adsr_1653_so_DEPENDENCIES = include/common.h
adsr_1680_so_DEPENDENCIES = include/common.h
amp_1654_so_DEPENDENCIES = include/math_func.h
dahdsr_2021_so_DEPENDENCIES = include/common.h
fmod_1656_so_DEPENDENCIES = include/math_func.h
lp4pole_1671_so_DEPENDENCIES = include/lp4pole_filter.h include/common.h
pulse_1645_so_DEPENDENCIES = include/wavedata.h
quantiser20_2027_so_DEPENDENCIES = include/common.h include/math_func.h
quantiser50_2028_so_DEPENDENCIES = include/common.h include/math_func.h
quantiser100_2029_so_DEPENDENCIES = include/common.h include/math_func.h
random_1661_so_DEPENDENCIES = include/common.h include/math_func.h
ratio_2034_so_DEPENDENCIES = include/common.h include/math_func.h
sawtooth_1641_so_DEPENDENCIES = include/wavedata.h
sequencer16_1677_so_DEPENDENCIES = include/common.h include/math_func.h
sequencer32_1676_so_DEPENDENCIES = include/common.h include/math_func.h
sequencer64_1675_so_DEPENDENCIES = include/common.h include/math_func.h
square_1643_so_DEPENDENCIES = include/wavedata.h
sync_pulse_2023_so_DEPENDENCIES = include/common.h
tracker_2025_so_DEPENDENCIES = include/common.h
triangle_1649_so_DEPENDENCIES = include/wavedata.h

clean:
	rm -f *.so *.so.o

wavedata.o: wavedata.c include/wavedata.h include/interpolate.h \
                       include/math_func.h include/common.h
	$(COMPILE) -DWAVEDATA_SUBDIR=\"$(wavedata_subdir)\" -c wavedata.c

sequencer64_1675.so$(EXEEXT): sequencer.so.c
	$(LINK) -DSEQUENCER_MAX_INPUTS=64 -DSEQUENCER_BASE_ID=1675 sequencer.so.c

sequencer32_1676.so$(EXEEXT): sequencer.so.c
	$(LINK) -DSEQUENCER_MAX_INPUTS=32 -DSEQUENCER_BASE_ID=1676 sequencer.so.c

sequencer16_1677.so$(EXEEXT): sequencer.so.c
	$(LINK) -DSEQUENCER_MAX_INPUTS=16 -DSEQUENCER_BASE_ID=1677 sequencer.so.c

## Remove optimisation, as -O1 or greater will cause quantiser to fail. I Don't Know Why.
quantiser20_2027.so$(EXEEXT): quantiser.so.c
	$(LINK) -O0 -DQUANTISER_MAX_INPUTS=20 -DQUANTISER_BASE_ID=2027 -lm quantiser.so.c

quantiser50_2028.so$(EXEEXT): quantiser.so.c
	$(LINK) -O0 -DQUANTISER_MAX_INPUTS=50 -DQUANTISER_BASE_ID=2028 -lm quantiser.so.c

quantiser100_2029.so$(EXEEXT): quantiser.so.c
	$(LINK) -O0 -DQUANTISER_MAX_INPUTS=100 -DQUANTISER_BASE_ID=2029 -lm quantiser.so.c

parabola_1649_data.so.c: wavegen
	./wavegen -r $(sample_rate) -f $(first_note) -s $(note_step) \
          -m $(min_table_size) -g $(gibbs) \
          -w parabola -p parabola -o parabola_1649_data.so.c

sawtooth_1641_data.so.c: wavegen
	./wavegen -r $(sample_rate) -f $(first_note) -s $(note_step) \
          -m $(min_table_size) -g $(gibbs) \
          -w saw -p sawtooth -o sawtooth_1641_data.so.c

square_1643_data.so.c: wavegen
	./wavegen -r $(sample_rate) -f $(first_note) -s $(note_step) \
          -m $(min_table_size) -g $(gibbs) \
          -w square -p square -o square_1643_data.so.c
