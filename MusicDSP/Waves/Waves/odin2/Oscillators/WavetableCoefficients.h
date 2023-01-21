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


//====================
//======= LFOS =======
//====================

#include "Wavetables/Coefficients/LFOPyramid12.h"
#include "Wavetables/Coefficients/LFOPyramid4.h"
#include "Wavetables/Coefficients/LFOPyramid6.h"
#include "Wavetables/Coefficients/LFOPyramid8.h"
#include "Wavetables/Coefficients/LFOSpike.h"
#include "Wavetables/Coefficients/LFOSquare12.h"
#include "Wavetables/Coefficients/LFOSquare25.h"
#include "Wavetables/Coefficients/LFOSquare50.h"
#include "Wavetables/Coefficients/LFOStair12.h"
#include "Wavetables/Coefficients/LFOStair3.h"
#include "Wavetables/Coefficients/LFOStair4.h"
#include "Wavetables/Coefficients/LFOStair6.h"
#include "Wavetables/Coefficients/LFOStair8.h"

// NOTE: All wavetables must be phasecorrected. Use
// WavetableContainer::eliminatePhaseInWavetableCoefficients() to do so!

#include "Wavetables/Coefficients/ChiptuneSquare12_5.h" //4
#include "Wavetables/Coefficients/ChiptuneSquare25.h"   //3
#include "Wavetables/Coefficients/ChiptuneSquare50.h"   //2
#include "Wavetables/Coefficients/ChiptuneTriangle.h"   //1
#include "Wavetables/Coefficients/Sine.h"               //0

// ! HOLE

#include "Wavetables/Coefficients/FatSaw.h"   //13->007
#include "Wavetables/Coefficients/Saw.h"      //11->005
#include "Wavetables/Coefficients/Triangle.h" //12->006

#include "Wavetables/Coefficients/Additive1.h"  //14->008
#include "Wavetables/Coefficients/Additive2.h"  //15->009
#include "Wavetables/Coefficients/Additive3.h"  //16->010
#include "Wavetables/Coefficients/Additive4.h"  //17->011
#include "Wavetables/Coefficients/Additive5.h"  //18->012
#include "Wavetables/Coefficients/Additive6.h"  //19->013
#include "Wavetables/Coefficients/Additive7.h"  //20->014
#include "Wavetables/Coefficients/Additive8.h"  //21->015
#include "Wavetables/Coefficients/Additive9.h"  //22->016
#include "Wavetables/Coefficients/Additive10.h" //23->017
#include "Wavetables/Coefficients/Additive11.h" //24->018
#include "Wavetables/Coefficients/Additive12.h" //25->019
#include "Wavetables/Coefficients/Additive13.h" //26->020
#include "Wavetables/Coefficients/Additive14.h" //27->021
#include "Wavetables/Coefficients/Additive15.h" //28->022
#include "Wavetables/Coefficients/Additive16.h" //29->023

#include "Wavetables/Coefficients/Harmonics1.h"  //30->024
#include "Wavetables/Coefficients/Harmonics2.h"  //31->025
#include "Wavetables/Coefficients/Harmonics3.h"  //32->026
#include "Wavetables/Coefficients/Harmonics4.h"  //33->027
#include "Wavetables/Coefficients/Harmonics5.h"  //34->028
#include "Wavetables/Coefficients/Harmonics6.h"  //35->029
#include "Wavetables/Coefficients/Harmonics7.h"  //36->030
#include "Wavetables/Coefficients/Harmonics8.h"  //37->031
#include "Wavetables/Coefficients/Harmonics9.h"  //38->032
#include "Wavetables/Coefficients/Harmonics10.h" //39->033
#include "Wavetables/Coefficients/Harmonics11.h" //40->034
#include "Wavetables/Coefficients/Harmonics12.h" //41->035
#include "Wavetables/Coefficients/Harmonics13.h" //42->036
#include "Wavetables/Coefficients/Harmonics14.h" //43->037
#include "Wavetables/Coefficients/Harmonics15.h" //44->038
#include "Wavetables/Coefficients/Harmonics16.h" //45->039

#include "Wavetables/Coefficients/Organ1.h" //46->040
#include "Wavetables/Coefficients/Organ2.h" //47->041
#include "Wavetables/Coefficients/Organ3.h" //48->042
#include "Wavetables/Coefficients/Organ4.h" //49->043

#include "Wavetables/Coefficients/BrokenSine1.h" //50->044
#include "Wavetables/Coefficients/BrokenSine2.h" //51->045
#include "Wavetables/Coefficients/BrokenSine3.h" //52->046
#include "Wavetables/Coefficients/BrokenSine4.h" //53->047

#include "Wavetables/Coefficients/Skyline1.h" //54->048
#include "Wavetables/Coefficients/Skyline2.h" //55->049
#include "Wavetables/Coefficients/Skyline3.h" //56->050
#include "Wavetables/Coefficients/Skyline4.h" //57->051

// ! HOLE

#include "Wavetables/Coefficients/Rectangular1.h" //66->056
#include "Wavetables/Coefficients/Rectangular2.h" //67->057
#include "Wavetables/Coefficients/Rectangular3.h" //68->058
#include "Wavetables/Coefficients/Rectangular4.h" //69->059

#include "Wavetables/Coefficients/Cello1.h"  //72->062
#include "Wavetables/Coefficients/Cello2.h"  //73->063
#include "Wavetables/Coefficients/Violin1.h" //70->060
#include "Wavetables/Coefficients/Violin2.h" //71->061

#include "Wavetables/Coefficients/Piano1.h" //74->064
#include "Wavetables/Coefficients/Piano2.h" //75->065
#include "Wavetables/Coefficients/Piano3.h" //76->066
#include "Wavetables/Coefficients/Piano4.h" //77->067

#include "Wavetables/Coefficients/Oboe1.h" //78->068
#include "Wavetables/Coefficients/Oboe2.h" //79->069
#include "Wavetables/Coefficients/Oboe3.h" //80->070
#include "Wavetables/Coefficients/Oboe4.h" //81->071

#include "Wavetables/Coefficients/LegToyBox.h"     //72
#include "Wavetables/Coefficients/LegRip2.h"       //73
#include "Wavetables/Coefficients/LegMale.h"       //74
#include "Wavetables/Coefficients/LegBarbedWire.h" //75

#include "Wavetables/Coefficients/Trumpet1.h" //86->076
#include "Wavetables/Coefficients/Trumpet2.h" //87->077
#include "Wavetables/Coefficients/Trumpet3.h" //88->078
#include "Wavetables/Coefficients/Trumpet4.h" //89->079

// these are all referenced in chiptune wavetable mapping:

//#include "Wavetables/Coefficients/CTDiverging.h" //94->080
#include "Wavetables/Coefficients/CTHigh.h"      //95->081
#include "Wavetables/Coefficients/CTHigh2.h"     //96->082
//#include "Wavetables/Coefficients/CTHigh3.h"     //97->083
#include "Wavetables/Coefficients/CTRich.h"      //98->084
#include "Wavetables/Coefficients/CTSoftTune.h"  //101->087
#include "Wavetables/Coefficients/ChipSaw.h"     //99->085
#include "Wavetables/Coefficients/ChipSine.h"    //100->086

#include "Wavetables/Coefficients/Glass.h"         //102->088
#include "Wavetables/Coefficients/GlassMutated1.h" //103->089
#include "Wavetables/Coefficients/GlassMutated2.h" //104->090
#include "Wavetables/Coefficients/GlassMutated3.h" //105->091

#include "Wavetables/Coefficients/FatSawMutated1.h" //109->092
#include "Wavetables/Coefficients/FatSawMutated2.h" //110->093
#include "Wavetables/Coefficients/FatSawMutated3.h" //111->094
#include "Wavetables/Coefficients/FatSawMutated4.h" //112->095
#include "Wavetables/Coefficients/FatSawMutated5.h" //113->096
#include "Wavetables/Coefficients/FatSawMutated6.h" //114->097
#include "Wavetables/Coefficients/FatSawMutated7.h" //115->098
#include "Wavetables/Coefficients/FatSawMutated8.h" //116->099

#include "Wavetables/Coefficients/ChiptuneSquare50Mutated1.h" //117->100
#include "Wavetables/Coefficients/ChiptuneSquare50Mutated2.h" //118->101
#include "Wavetables/Coefficients/ChiptuneSquare50Mutated3.h" //119->102
#include "Wavetables/Coefficients/ChiptuneSquare50Mutated4.h" //120->103
#include "Wavetables/Coefficients/ChiptuneSquare50Mutated5.h" //121->104
#include "Wavetables/Coefficients/ChiptuneSquare50Mutated6.h" //122->105
#include "Wavetables/Coefficients/ChiptuneSquare50Mutated7.h" //123->106
#include "Wavetables/Coefficients/ChiptuneSquare50Mutated8.h" //124->107

#include "Wavetables/Coefficients/LegAdd8.h"  //108
#include "Wavetables/Coefficients/LegSharp.h" //109
#include "Wavetables/Coefficients/LegPiano.h" //110
#include "Wavetables/Coefficients/LegAdd1.h"   //111

#include "Wavetables/Coefficients/LegCello.h" //112
#include "Wavetables/Coefficients/LegAah.h" //113
#include "Wavetables/Coefficients/LegHarm2.h"  //114
#include "Wavetables/Coefficients/LegNoBass.h"//115

#include "Wavetables/Coefficients/LegBags.h"  //116
#include "Wavetables/Coefficients/LegOrgan.h" //117
#include "Wavetables/Coefficients/LegTriQuad.h"//118
#include "Wavetables/Coefficients/LegAdd3.h"  //119

#include "Wavetables/Coefficients/AKWF_birds_0010.h"      //165->140
#include "Wavetables/Coefficients/AKWF_birds_0011.h"      //166->141
#include "Wavetables/Coefficients/AKWF_birds_0014.h"      //167->142
#include "Wavetables/Coefficients/AKWF_birds_0004.h"      //birds168->143

#include "Wavetables/Coefficients/AKWF_bitreduced_0002.h" //169->144
#include "Wavetables/Coefficients/AKWF_bitreduced_0003.h" //170->145
#include "Wavetables/Coefficients/AKWF_bitreduced_0006.h" //171->146
#include "Wavetables/Coefficients/AKWF_bitreduced_0011.h" //172->147

#include "Wavetables/Coefficients/AKWF_fmsynth_0011.h"    //161->136
#include "Wavetables/Coefficients/AKWF_fmsynth_0032.h"    //162->137
#include "Wavetables/Coefficients/AKWF_fmsynth_0034.h"    //163->138
#include "Wavetables/Coefficients/AKWF_fmsynth_0081.h"    //164->139

#include "Wavetables/Coefficients/AKWF_hvoice_0002.h"     //145->120
#include "Wavetables/Coefficients/AKWF_hvoice_0010.h"     //146->121
#include "Wavetables/Coefficients/AKWF_hvoice_0014.h"     //147->122
#include "Wavetables/Coefficients/AKWF_hvoice_0019.h"     //148->123

#include "Wavetables/Coefficients/AKWF_hvoice_0020.h"     //149->124
#include "Wavetables/Coefficients/AKWF_hvoice_0021.h"     //150->125
#include "Wavetables/Coefficients/AKWF_hvoice_0029.h"     //151->126
#include "Wavetables/Coefficients/AKWF_hvoice_0032.h"     //152->127

#include "Wavetables/Coefficients/AKWF_hvoice_0037.h"     //153->128
#include "Wavetables/Coefficients/AKWF_hvoice_0041.h"     //154->129
#include "Wavetables/Coefficients/AKWF_hvoice_0047.h"     //155->130
#include "Wavetables/Coefficients/AKWF_hvoice_0049.h"     //156->131

#include "Wavetables/Coefficients/AKWF_hvoice_0056.h"     //157->132
#include "Wavetables/Coefficients/AKWF_hvoice_0064.h"     //158->133
#include "Wavetables/Coefficients/AKWF_hvoice_0071.h"     //159->134
#include "Wavetables/Coefficients/AKWF_hvoice_0093.h"     //160->135

#include "Wavetables/Coefficients/PerlinReplace1.h" //177->148
#include "Wavetables/Coefficients/PerlinReplace2.h" //178->149
#include "Wavetables/Coefficients/PerlinReplace3.h" //179->150
#include "Wavetables/Coefficients/PerlinReplace4.h" //180->15

#include "Wavetables/Coefficients/BagPipe.h"         //152
#include "Wavetables/Coefficients/BagPipeMutated1.h" //153
#include "Wavetables/Coefficients/BagPipeMutated5.h" //154
#include "Wavetables/Coefficients/BagPipeMutated4.h" //155

//#include "Wavetables/Coefficients/ChiptuneReplace1.h" //156
//#include "Wavetables/Coefficients/ChiptuneReplace2.h" //157
//#include "Wavetables/Coefficients/ChiptuneReplace3.h" //158
//#include "Wavetables/Coefficients/ChiptuneReplace4.h" //159
