/*
** Odin 2 Synthesizer Plugin
** Copyright (C) 2020 - 2021 TheWaveWarden
**
** Odin 2 is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published by
** the Free Software Foundation, either version 3 of the License, or
** (at your option) any later version.
**
** Odin 2 is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
** GNU General Public License for more details.
*/

#pragma once

#define SUBTABLES_PER_WAVETABLE 33 //there are 40 tables for one "sound"

#define NUMBER_OF_WAVETABLES 160  //number of different 1D WTs
#define NUMBER_OF_CHIPDRAW_TABLES 3 //number of chipdraw tables (one per osc slot)
#define NUMBER_OF_WAVEDRAW_TABLES 3 //number of wavedraw tables (one per osc slot)
#define NUMBER_OF_SPECDRAW_TABLES 3 //number of specdraw tables (one per osc slot)
#define NUMBER_OF_LFOTABLES 13
//#define CHIPDRAW_LENGTH 32 //how many samples we draw in the chipdraw window
//#define WAVEDRAW_LENGTH 256 //how many samples we draw in the wavedraw window
//#define SPECDRAW_LENGTH 100
#define NUMBER_OF_HARMONICS 256 //maximum harmonics used (Wavetable length is 512 -> nyquist -> 256)
#define WAVETABLE_LENGTH 512 //how long the actual wavetable is
#define MAX_EXPECTED_SAMPLE_RATE 192000 //use this for calculation of Delaylength
#define MAX_DELAY_TIME 4//max delay time at max samplerate in seconds

#define SIN_AND_COS 2 //just here to avoid magic Numbers
#define NUMBER_OF_WAVETABLES_2D 40
#define TABLES_PER_2D_WT 4
#define PI 3.1415926535897932384626433832795
