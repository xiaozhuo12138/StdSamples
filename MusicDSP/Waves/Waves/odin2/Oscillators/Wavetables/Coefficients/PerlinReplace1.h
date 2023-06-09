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

#define WT_NR 148


m_wavetable_names_1D[WT_NR] = "PerlinReplace1";




//m_fourier_coeffs[WT_NR][1][0] = 0.249254f; // scalar

//m_fourier_coeffs[WT_NR][0][1] = 0.251153f;
//m_fourier_coeffs[WT_NR][0][2] = 0.639726f;
//m_fourier_coeffs[WT_NR][0][3] = 0.401132f;
//m_fourier_coeffs[WT_NR][0][4] = 0.486307f;
//m_fourier_coeffs[WT_NR][0][5] = 0.494481f;
//m_fourier_coeffs[WT_NR][0][6] = 0.226136f;
//m_fourier_coeffs[WT_NR][0][7] = 0.276042f;
//m_fourier_coeffs[WT_NR][0][8] = 0.290607f;
//m_fourier_coeffs[WT_NR][0][9] = 0.557499f;
//m_fourier_coeffs[WT_NR][0][10] = 0.374731f;
//m_fourier_coeffs[WT_NR][0][11] = 0.152842f;
//m_fourier_coeffs[WT_NR][0][12] = 0.262969f;
//m_fourier_coeffs[WT_NR][0][13] = 0.431098f;
//m_fourier_coeffs[WT_NR][0][14] = 0.451235f;
//m_fourier_coeffs[WT_NR][0][15] = 0.298411f;
//m_fourier_coeffs[WT_NR][0][16] = 0.091236f;
//m_fourier_coeffs[WT_NR][0][17] = 0.049385f;
//m_fourier_coeffs[WT_NR][0][18] = 0.332036f;
//m_fourier_coeffs[WT_NR][0][19] = 0.026647f;
//m_fourier_coeffs[WT_NR][0][20] = 0.223085f;
//m_fourier_coeffs[WT_NR][0][21] = 0.308046f;
//m_fourier_coeffs[WT_NR][0][22] = 0.235807f;
//m_fourier_coeffs[WT_NR][0][23] = 0.296816f;
//m_fourier_coeffs[WT_NR][0][24] = 0.221308f;
//m_fourier_coeffs[WT_NR][0][25] = 0.293199f;
//m_fourier_coeffs[WT_NR][0][26] = 0.084721f;
//m_fourier_coeffs[WT_NR][0][27] = 0.332909f;
//m_fourier_coeffs[WT_NR][0][28] = 0.386767f;
//m_fourier_coeffs[WT_NR][0][29] = 0.287995f;
//m_fourier_coeffs[WT_NR][0][30] = 0.403243f;
//m_fourier_coeffs[WT_NR][0][31] = 0.133027f;
//m_fourier_coeffs[WT_NR][0][32] = 0.214857f;
//m_fourier_coeffs[WT_NR][0][33] = 0.455683f;
//m_fourier_coeffs[WT_NR][0][34] = 0.322053f;
//m_fourier_coeffs[WT_NR][0][35] = 0.486049f;
//m_fourier_coeffs[WT_NR][0][36] = 0.105633f;
//m_fourier_coeffs[WT_NR][0][37] = 0.155178f;
//m_fourier_coeffs[WT_NR][0][38] = 0.057154f;
//m_fourier_coeffs[WT_NR][0][39] = 0.164393f;
//m_fourier_coeffs[WT_NR][0][40] = 0.236335f;
//m_fourier_coeffs[WT_NR][0][41] = 0.318897f;
//m_fourier_coeffs[WT_NR][0][42] = 0.228229f;
//m_fourier_coeffs[WT_NR][0][43] = 0.201817f;
//m_fourier_coeffs[WT_NR][0][44] = 0.102945f;
//m_fourier_coeffs[WT_NR][0][45] = 0.127653f;
//m_fourier_coeffs[WT_NR][0][46] = 0.309365f;
//m_fourier_coeffs[WT_NR][0][47] = 0.121833f;
//m_fourier_coeffs[WT_NR][0][48] = 0.219240f;
//m_fourier_coeffs[WT_NR][0][49] = 0.034422f;
//m_fourier_coeffs[WT_NR][0][50] = 0.080501f;
//m_fourier_coeffs[WT_NR][0][51] = 0.196545f;
//m_fourier_coeffs[WT_NR][0][52] = 0.257382f;
//m_fourier_coeffs[WT_NR][0][53] = 0.231871f;
//m_fourier_coeffs[WT_NR][0][54] = 0.155581f;
//m_fourier_coeffs[WT_NR][0][55] = 0.174397f;
//m_fourier_coeffs[WT_NR][0][56] = 0.051779f;
//m_fourier_coeffs[WT_NR][0][57] = 0.172710f;
//m_fourier_coeffs[WT_NR][0][58] = 0.137171f;
//m_fourier_coeffs[WT_NR][0][59] = 0.126526f;
//m_fourier_coeffs[WT_NR][0][60] = 0.094902f;
//m_fourier_coeffs[WT_NR][0][61] = 0.027743f;
//m_fourier_coeffs[WT_NR][0][62] = 0.151781f;
//m_fourier_coeffs[WT_NR][0][63] = 0.033264f;
//m_fourier_coeffs[WT_NR][0][64] = 0.084671f;
//m_fourier_coeffs[WT_NR][0][65] = 0.040647f;
//m_fourier_coeffs[WT_NR][0][66] = 0.016994f;
//m_fourier_coeffs[WT_NR][0][67] = 0.094648f;
//m_fourier_coeffs[WT_NR][0][68] = 0.116025f;
//m_fourier_coeffs[WT_NR][0][69] = 0.075957f;
//m_fourier_coeffs[WT_NR][0][70] = 0.034118f;
//m_fourier_coeffs[WT_NR][0][71] = 0.075243f;
//m_fourier_coeffs[WT_NR][0][72] = 0.039457f;
//m_fourier_coeffs[WT_NR][0][73] = 0.094461f;
//m_fourier_coeffs[WT_NR][0][74] = 0.055880f;
//m_fourier_coeffs[WT_NR][0][75] = 0.026618f;
//m_fourier_coeffs[WT_NR][0][76] = 0.041292f;
//m_fourier_coeffs[WT_NR][0][77] = 0.048073f;
//m_fourier_coeffs[WT_NR][0][78] = 0.119205f;
//m_fourier_coeffs[WT_NR][0][79] = 0.044695f;
//m_fourier_coeffs[WT_NR][0][80] = 0.046205f;
//m_fourier_coeffs[WT_NR][0][81] = 0.025522f;
//m_fourier_coeffs[WT_NR][0][82] = 0.071565f;
//m_fourier_coeffs[WT_NR][0][83] = 0.073923f;
//m_fourier_coeffs[WT_NR][0][84] = 0.073594f;
//m_fourier_coeffs[WT_NR][0][85] = 0.021129f;
//m_fourier_coeffs[WT_NR][0][86] = 0.031871f;
//m_fourier_coeffs[WT_NR][0][87] = 0.026472f;
//m_fourier_coeffs[WT_NR][0][88] = 0.096537f;
//m_fourier_coeffs[WT_NR][0][89] = 0.008915f;
//m_fourier_coeffs[WT_NR][0][90] = 0.016748f;
//m_fourier_coeffs[WT_NR][0][91] = 0.028880f;
//m_fourier_coeffs[WT_NR][0][92] = 0.024447f;
//m_fourier_coeffs[WT_NR][0][93] = 0.040229f;
//m_fourier_coeffs[WT_NR][0][94] = 0.014656f;
//m_fourier_coeffs[WT_NR][0][95] = 0.031334f;
//m_fourier_coeffs[WT_NR][0][96] = 0.028708f;
//m_fourier_coeffs[WT_NR][0][97] = 0.029862f;
//m_fourier_coeffs[WT_NR][0][98] = 0.018890f;
//m_fourier_coeffs[WT_NR][0][99] = 0.028078f;
//m_fourier_coeffs[WT_NR][0][100] = 0.021918f;
//m_fourier_coeffs[WT_NR][0][101] = 0.030622f;
//m_fourier_coeffs[WT_NR][0][102] = 0.029063f;
//m_fourier_coeffs[WT_NR][0][103] = 0.025760f;
//m_fourier_coeffs[WT_NR][0][104] = 0.028771f;
//m_fourier_coeffs[WT_NR][0][105] = 0.021163f;
//m_fourier_coeffs[WT_NR][0][106] = 0.025503f;
//m_fourier_coeffs[WT_NR][0][107] = 0.009395f;
//m_fourier_coeffs[WT_NR][0][108] = 0.011708f;
//m_fourier_coeffs[WT_NR][0][109] = 0.023259f;
//m_fourier_coeffs[WT_NR][0][110] = 0.011920f;
//m_fourier_coeffs[WT_NR][0][111] = 0.009968f;
//m_fourier_coeffs[WT_NR][0][112] = 0.026867f;
//m_fourier_coeffs[WT_NR][0][113] = 0.017700f;
//m_fourier_coeffs[WT_NR][0][114] = 0.017688f;
//m_fourier_coeffs[WT_NR][0][115] = 0.016847f;
//m_fourier_coeffs[WT_NR][0][116] = 0.036975f;
//m_fourier_coeffs[WT_NR][0][117] = 0.017106f;
//m_fourier_coeffs[WT_NR][0][118] = 0.005862f;
//m_fourier_coeffs[WT_NR][0][119] = 0.007703f;
//m_fourier_coeffs[WT_NR][0][120] = 0.025704f;
//m_fourier_coeffs[WT_NR][0][121] = 0.019716f;
//m_fourier_coeffs[WT_NR][0][122] = 0.014888f;
//m_fourier_coeffs[WT_NR][0][123] = 0.006830f;
//m_fourier_coeffs[WT_NR][0][124] = 0.003124f;
//m_fourier_coeffs[WT_NR][0][125] = 0.014501f;
//m_fourier_coeffs[WT_NR][0][126] = 0.025089f;
//m_fourier_coeffs[WT_NR][0][127] = 0.018214f;
//m_fourier_coeffs[WT_NR][0][128] = 0.029885f;
//m_fourier_coeffs[WT_NR][0][129] = 0.004240f;
//m_fourier_coeffs[WT_NR][0][130] = 0.013884f;
//m_fourier_coeffs[WT_NR][0][131] = 0.010997f;
//m_fourier_coeffs[WT_NR][0][132] = 0.016035f;
//m_fourier_coeffs[WT_NR][0][133] = 0.006044f;
//m_fourier_coeffs[WT_NR][0][134] = 0.008913f;
//m_fourier_coeffs[WT_NR][0][135] = 0.013746f;
//m_fourier_coeffs[WT_NR][0][136] = 0.014568f;
//m_fourier_coeffs[WT_NR][0][137] = 0.008325f;
//m_fourier_coeffs[WT_NR][0][138] = 0.026989f;
//m_fourier_coeffs[WT_NR][0][139] = 0.019316f;
//m_fourier_coeffs[WT_NR][0][140] = 0.016946f;
//m_fourier_coeffs[WT_NR][0][141] = 0.008912f;
//m_fourier_coeffs[WT_NR][0][142] = 0.005746f;
//m_fourier_coeffs[WT_NR][0][143] = 0.012759f;
//m_fourier_coeffs[WT_NR][0][144] = 0.013198f;
//m_fourier_coeffs[WT_NR][0][145] = 0.008717f;
//m_fourier_coeffs[WT_NR][0][146] = 0.006650f;
//m_fourier_coeffs[WT_NR][0][147] = 0.013527f;
//m_fourier_coeffs[WT_NR][0][148] = 0.004993f;
//m_fourier_coeffs[WT_NR][0][149] = 0.018129f;
//m_fourier_coeffs[WT_NR][0][150] = 0.009522f;
//m_fourier_coeffs[WT_NR][0][151] = 0.001999f;
//m_fourier_coeffs[WT_NR][0][152] = 0.011521f;
//m_fourier_coeffs[WT_NR][0][153] = 0.005173f;
//m_fourier_coeffs[WT_NR][0][154] = 0.008963f;
//m_fourier_coeffs[WT_NR][0][155] = 0.008767f;
//m_fourier_coeffs[WT_NR][0][156] = 0.004010f;
//m_fourier_coeffs[WT_NR][0][157] = 0.016571f;
//m_fourier_coeffs[WT_NR][0][158] = 0.009148f;
//m_fourier_coeffs[WT_NR][0][159] = 0.007778f;
//m_fourier_coeffs[WT_NR][0][160] = 0.004063f;
//m_fourier_coeffs[WT_NR][0][161] = 0.005346f;
//m_fourier_coeffs[WT_NR][0][162] = 0.006570f;
//m_fourier_coeffs[WT_NR][0][163] = 0.001138f;
//m_fourier_coeffs[WT_NR][0][164] = 0.003458f;
//m_fourier_coeffs[WT_NR][0][165] = 0.010155f;
//m_fourier_coeffs[WT_NR][0][166] = 0.004279f;
//m_fourier_coeffs[WT_NR][0][167] = 0.009436f;
//m_fourier_coeffs[WT_NR][0][168] = 0.008427f;
//m_fourier_coeffs[WT_NR][0][169] = 0.001565f;
//m_fourier_coeffs[WT_NR][0][170] = 0.003280f;
//m_fourier_coeffs[WT_NR][0][171] = 0.008281f;
//m_fourier_coeffs[WT_NR][0][172] = 0.010444f;
//m_fourier_coeffs[WT_NR][0][173] = 0.002240f;
//m_fourier_coeffs[WT_NR][0][174] = 0.010445f;
//m_fourier_coeffs[WT_NR][0][175] = 0.004668f;
//m_fourier_coeffs[WT_NR][0][176] = 0.003896f;
//m_fourier_coeffs[WT_NR][0][177] = 0.002990f;
//m_fourier_coeffs[WT_NR][0][178] = 0.002823f;
//m_fourier_coeffs[WT_NR][0][179] = 0.004982f;
//m_fourier_coeffs[WT_NR][0][180] = 0.003594f;
//m_fourier_coeffs[WT_NR][0][181] = 0.008810f;
//m_fourier_coeffs[WT_NR][0][182] = 0.005046f;
//m_fourier_coeffs[WT_NR][0][183] = 0.000216f;
//m_fourier_coeffs[WT_NR][0][184] = 0.006800f;
//m_fourier_coeffs[WT_NR][0][185] = 0.006959f;
//m_fourier_coeffs[WT_NR][0][186] = 0.007369f;
//m_fourier_coeffs[WT_NR][0][187] = 0.007185f;
//m_fourier_coeffs[WT_NR][0][188] = 0.007323f;
//m_fourier_coeffs[WT_NR][0][189] = 0.005100f;
//m_fourier_coeffs[WT_NR][0][190] = 0.002649f;
//m_fourier_coeffs[WT_NR][0][191] = 0.003099f;
//m_fourier_coeffs[WT_NR][0][192] = 0.005880f;
//m_fourier_coeffs[WT_NR][0][193] = 0.003366f;
//m_fourier_coeffs[WT_NR][0][194] = 0.004894f;
//m_fourier_coeffs[WT_NR][0][195] = 0.001380f;
//m_fourier_coeffs[WT_NR][0][196] = 0.004467f;
//m_fourier_coeffs[WT_NR][0][197] = 0.003721f;
//m_fourier_coeffs[WT_NR][0][198] = 0.009369f;
//m_fourier_coeffs[WT_NR][0][199] = 0.007252f;
//m_fourier_coeffs[WT_NR][0][200] = 0.005928f;
//m_fourier_coeffs[WT_NR][0][201] = 0.003245f;
//m_fourier_coeffs[WT_NR][0][202] = 0.001517f;
//m_fourier_coeffs[WT_NR][0][203] = 0.003236f;
//m_fourier_coeffs[WT_NR][0][204] = 0.004665f;
//m_fourier_coeffs[WT_NR][0][205] = 0.005038f;
//m_fourier_coeffs[WT_NR][0][206] = 0.001425f;
//m_fourier_coeffs[WT_NR][0][207] = 0.002223f;
//m_fourier_coeffs[WT_NR][0][208] = 0.003553f;
//m_fourier_coeffs[WT_NR][0][209] = 0.003766f;
//m_fourier_coeffs[WT_NR][0][210] = 0.004319f;
//m_fourier_coeffs[WT_NR][0][211] = 0.005404f;
//m_fourier_coeffs[WT_NR][0][212] = 0.003257f;
//m_fourier_coeffs[WT_NR][0][213] = 0.001847f;
//m_fourier_coeffs[WT_NR][0][214] = 0.003367f;
//m_fourier_coeffs[WT_NR][0][215] = 0.004157f;
//m_fourier_coeffs[WT_NR][0][216] = 0.003012f;
//m_fourier_coeffs[WT_NR][0][217] = 0.003218f;
//m_fourier_coeffs[WT_NR][0][218] = 0.004279f;
//m_fourier_coeffs[WT_NR][0][219] = 0.000958f;
//m_fourier_coeffs[WT_NR][0][220] = 0.003650f;
//m_fourier_coeffs[WT_NR][0][221] = 0.001809f;
//m_fourier_coeffs[WT_NR][0][222] = 0.003361f;
//m_fourier_coeffs[WT_NR][0][223] = 0.002710f;
//m_fourier_coeffs[WT_NR][0][224] = 0.002729f;
//m_fourier_coeffs[WT_NR][0][225] = 0.003968f;
//m_fourier_coeffs[WT_NR][0][226] = 0.002604f;
//m_fourier_coeffs[WT_NR][0][227] = 0.002088f;
//m_fourier_coeffs[WT_NR][0][228] = 0.004705f;
//m_fourier_coeffs[WT_NR][0][229] = 0.002762f;
//m_fourier_coeffs[WT_NR][0][230] = 0.001340f;
//m_fourier_coeffs[WT_NR][0][231] = 0.002372f;
//m_fourier_coeffs[WT_NR][0][232] = 0.003417f;
//m_fourier_coeffs[WT_NR][0][233] = 0.003199f;
//m_fourier_coeffs[WT_NR][0][234] = 0.002599f;
//m_fourier_coeffs[WT_NR][0][235] = 0.003174f;
//m_fourier_coeffs[WT_NR][0][236] = 0.000650f;
//m_fourier_coeffs[WT_NR][0][237] = 0.001249f;
//m_fourier_coeffs[WT_NR][0][238] = 0.002731f;
//m_fourier_coeffs[WT_NR][0][239] = 0.003636f;
//m_fourier_coeffs[WT_NR][0][240] = 0.001554f;
//m_fourier_coeffs[WT_NR][0][241] = 0.002659f;
//m_fourier_coeffs[WT_NR][0][242] = 0.002011f;
//m_fourier_coeffs[WT_NR][0][243] = 0.001953f;
//m_fourier_coeffs[WT_NR][0][244] = 0.003323f;
//m_fourier_coeffs[WT_NR][0][245] = 0.003383f;
//m_fourier_coeffs[WT_NR][0][246] = 0.002637f;
//m_fourier_coeffs[WT_NR][0][247] = 0.001693f;
//m_fourier_coeffs[WT_NR][0][248] = 0.002105f;
//m_fourier_coeffs[WT_NR][0][249] = 0.002086f;
//m_fourier_coeffs[WT_NR][0][250] = 0.002231f;
//m_fourier_coeffs[WT_NR][0][251] = 0.001673f;
//m_fourier_coeffs[WT_NR][0][252] = 0.002302f;
//m_fourier_coeffs[WT_NR][0][253] = 0.002837f;
//m_fourier_coeffs[WT_NR][0][254] = 0.002525f;
//m_fourier_coeffs[WT_NR][0][255] = 0.001887f;


#undef WT_NR
