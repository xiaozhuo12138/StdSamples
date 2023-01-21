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

#define WT_NR 6
	
m_wavetable_names_1D[WT_NR] = "Triangle";
//m_fourier_coeffs[WT_NR][0][0] = 1.f; //signals sine only
//m_fourier_coeffs[WT_NR][1][0] = 0.802871; //Triangle scalar

//m_fourier_coeffs[WT_NR][0][1] = 1.0f;
//m_fourier_coeffs[WT_NR][0][3] = -0.111111;
//m_fourier_coeffs[WT_NR][0][5] = 0.04f;
//m_fourier_coeffs[WT_NR][0][7] = -0.0204082f;
//m_fourier_coeffs[WT_NR][0][9] = 0.0123457f;
//m_fourier_coeffs[WT_NR][0][11] = -0.00826446f;
//m_fourier_coeffs[WT_NR][0][13] = 0.00591716f;
//m_fourier_coeffs[WT_NR][0][15] = -0.00444444f;
//m_fourier_coeffs[WT_NR][0][17] = 0.00346021;
//m_fourier_coeffs[WT_NR][0][19] = -0.00277008f;
//m_fourier_coeffs[WT_NR][0][21] = 0.00226757f;
//m_fourier_coeffs[WT_NR][0][23] = -0.00189036f;
//m_fourier_coeffs[WT_NR][0][25] = 0.0016f;
//m_fourier_coeffs[WT_NR][0][27] = -0.00137174f;
//m_fourier_coeffs[WT_NR][0][29] = 0.00118906f;
//m_fourier_coeffs[WT_NR][0][31] = -0.00104058f;
//m_fourier_coeffs[WT_NR][0][33] = 9.18274e-04f;
//m_fourier_coeffs[WT_NR][0][35] = -8.16327e-04f;
//m_fourier_coeffs[WT_NR][0][37] = 7.3046e-04f;
//m_fourier_coeffs[WT_NR][0][39] = -6.57462e-04f;
//m_fourier_coeffs[WT_NR][0][41] = 5.94884e-04f;
//m_fourier_coeffs[WT_NR][0][43] = -5.40833e-04f;
//m_fourier_coeffs[WT_NR][0][45] = 4.93827e-04f;
//m_fourier_coeffs[WT_NR][0][47] = -4.52694e-04f;
//m_fourier_coeffs[WT_NR][0][49] = 4.16493e-04f;
//m_fourier_coeffs[WT_NR][0][51] = -3.84468e-04f;
//m_fourier_coeffs[WT_NR][0][53] = 3.55999e-04f;
//m_fourier_coeffs[WT_NR][0][55] = -3.30579e-04f;
//m_fourier_coeffs[WT_NR][0][57] = 3.07787e-04f;
//m_fourier_coeffs[WT_NR][0][59] = -2.87274e-04f;
//m_fourier_coeffs[WT_NR][0][61] = 2.68745e-04f;
//m_fourier_coeffs[WT_NR][0][63] = -2.51953e-04f;
//m_fourier_coeffs[WT_NR][0][65] = 2.36686e-04f;
//m_fourier_coeffs[WT_NR][0][67] = -2.22767e-04f;
//m_fourier_coeffs[WT_NR][0][69] = 2.1004e-04f;
//m_fourier_coeffs[WT_NR][0][71] = -1.98373e-04f;
//m_fourier_coeffs[WT_NR][0][73] = 1.87652e-04f;
//m_fourier_coeffs[WT_NR][0][75] = -1.77778e-04f;
//m_fourier_coeffs[WT_NR][0][77] = 1.68663e-04f;
//m_fourier_coeffs[WT_NR][0][79] = -1.60231e-04f;
//m_fourier_coeffs[WT_NR][0][81] = 1.52416e-04f;
//m_fourier_coeffs[WT_NR][0][83] = -1.45159e-04f;
//m_fourier_coeffs[WT_NR][0][85] = 1.38408e-04f;
//m_fourier_coeffs[WT_NR][0][87] = -1.32118e-04f;
//m_fourier_coeffs[WT_NR][0][89] = 1.26247e-04f;
//m_fourier_coeffs[WT_NR][0][91] = -1.20758e-04f;
//m_fourier_coeffs[WT_NR][0][93] = 1.1562e-04f;
//m_fourier_coeffs[WT_NR][0][95] = -1.10803e-04f;
//m_fourier_coeffs[WT_NR][0][97] = 1.06281e-04f;
//m_fourier_coeffs[WT_NR][0][99] = -1.0203e-04f;
//m_fourier_coeffs[WT_NR][0][101] = 9.80296e-05f;
//m_fourier_coeffs[WT_NR][0][103] = -9.42596e-05f;
//m_fourier_coeffs[WT_NR][0][105] = 9.0703e-05f;
//m_fourier_coeffs[WT_NR][0][107] = -8.73439e-05f;
//m_fourier_coeffs[WT_NR][0][109] = 8.4168e-05f;
//m_fourier_coeffs[WT_NR][0][111] = -8.11622e-05f;
//m_fourier_coeffs[WT_NR][0][113] = 7.83147e-05f;
//m_fourier_coeffs[WT_NR][0][115] = -7.56144e-05f;
//m_fourier_coeffs[WT_NR][0][117] = 7.30514e-05f;
//m_fourier_coeffs[WT_NR][0][119] = -7.06165e-05f;
//m_fourier_coeffs[WT_NR][0][121] = 6.83013e-05f;
//m_fourier_coeffs[WT_NR][0][123] = -6.60982e-05f;
//m_fourier_coeffs[WT_NR][0][125] = 6.4e-05f;
//m_fourier_coeffs[WT_NR][0][127] = -6.20001e-05f;
//m_fourier_coeffs[WT_NR][0][129] = 6.00925e-05f;
//m_fourier_coeffs[WT_NR][0][131] = -5.82717e-05f;
//m_fourier_coeffs[WT_NR][0][133] = 5.65323e-05f;
//m_fourier_coeffs[WT_NR][0][135] = -5.48697e-05f;
//m_fourier_coeffs[WT_NR][0][137] = 5.32793e-05f;
//m_fourier_coeffs[WT_NR][0][139] = -5.17572e-05f;
//m_fourier_coeffs[WT_NR][0][141] = 5.02993e-05f;
//m_fourier_coeffs[WT_NR][0][143] = -4.89021e-05f;
//m_fourier_coeffs[WT_NR][0][145] = 4.75624e-05f;
//m_fourier_coeffs[WT_NR][0][147] = -4.6277e-05f;
//m_fourier_coeffs[WT_NR][0][149] = 4.5043e-05f;
//m_fourier_coeffs[WT_NR][0][151] = -4.38577e-05f;
//m_fourier_coeffs[WT_NR][0][153] = 4.27186e-05f;
//m_fourier_coeffs[WT_NR][0][155] = -4.16233e-05f;
//m_fourier_coeffs[WT_NR][0][157] = 4.05696e-05f;
//m_fourier_coeffs[WT_NR][0][159] = -3.95554e-05f;
//m_fourier_coeffs[WT_NR][0][161] = 3.85788e-05f;
//m_fourier_coeffs[WT_NR][0][163] = -3.76378e-05f;
//m_fourier_coeffs[WT_NR][0][165] = 3.67309e-05f;
//m_fourier_coeffs[WT_NR][0][167] = -3.58564e-05f;
//m_fourier_coeffs[WT_NR][0][169] = 3.50128e-05f;
//m_fourier_coeffs[WT_NR][0][171] = -3.41986e-05f;
//m_fourier_coeffs[WT_NR][0][173] = 3.34124e-05f;
//m_fourier_coeffs[WT_NR][0][175] = -3.26531e-05f;
//m_fourier_coeffs[WT_NR][0][177] = 3.19193e-05f;
//m_fourier_coeffs[WT_NR][0][179] = -3.121e-05f;
//m_fourier_coeffs[WT_NR][0][181] = 3.05241e-05f;
//m_fourier_coeffs[WT_NR][0][183] = -2.98606e-05f;
//m_fourier_coeffs[WT_NR][0][185] = 2.92184e-05f;
//m_fourier_coeffs[WT_NR][0][187] = -2.85968e-05f;
//m_fourier_coeffs[WT_NR][0][189] = 2.79947e-05f;
//m_fourier_coeffs[WT_NR][0][191] = -2.74115e-05f;
//m_fourier_coeffs[WT_NR][0][193] = 2.68464e-05f;
//m_fourier_coeffs[WT_NR][0][195] = -2.62985e-05f;
//m_fourier_coeffs[WT_NR][0][197] = 2.57672e-05f;
//m_fourier_coeffs[WT_NR][0][199] = -2.52519e-05f;
//m_fourier_coeffs[WT_NR][0][201] = 2.47519e-05f;
//m_fourier_coeffs[WT_NR][0][203] = -2.42665e-05f;
//m_fourier_coeffs[WT_NR][0][205] = 2.37954e-05f;
//m_fourier_coeffs[WT_NR][0][207] = -2.33378e-05f;
//m_fourier_coeffs[WT_NR][0][209] = 2.28932e-05f;
//m_fourier_coeffs[WT_NR][0][211] = -2.24613e-05f;
//m_fourier_coeffs[WT_NR][0][213] = 2.20415e-05f;
//m_fourier_coeffs[WT_NR][0][215] = -2.16333e-05f;
//m_fourier_coeffs[WT_NR][0][217] = 2.12364e-05f;
//m_fourier_coeffs[WT_NR][0][219] = -2.08503e-05f;
//m_fourier_coeffs[WT_NR][0][221] = 2.04746e-05f;
//m_fourier_coeffs[WT_NR][0][223] = -2.0109e-05f;
//m_fourier_coeffs[WT_NR][0][225] = 1.97531e-05f;
//m_fourier_coeffs[WT_NR][0][227] = -1.94065e-05f;
//m_fourier_coeffs[WT_NR][0][229] = 1.9069e-05f;
//m_fourier_coeffs[WT_NR][0][231] = -1.87403e-05f;
//m_fourier_coeffs[WT_NR][0][233] = 1.84199e-05f;
//m_fourier_coeffs[WT_NR][0][235] = -1.81077e-05f;
//m_fourier_coeffs[WT_NR][0][237] = 1.78034e-05f;
//m_fourier_coeffs[WT_NR][0][239] = -1.75067e-05f;
//m_fourier_coeffs[WT_NR][0][241] = 1.72173e-05f;
//m_fourier_coeffs[WT_NR][0][243] = -1.69351e-05f;
//m_fourier_coeffs[WT_NR][0][245] = 1.66597e-05f;
//m_fourier_coeffs[WT_NR][0][247] = -1.6391e-05f;
//m_fourier_coeffs[WT_NR][0][249] = 1.61288e-05f;
//m_fourier_coeffs[WT_NR][0][251] = -1.58728e-05f;
//m_fourier_coeffs[WT_NR][0][253] = 1.56228e-05f;
//m_fourier_coeffs[WT_NR][0][255] = -1.53787e-05f;
#undef WT_NR
