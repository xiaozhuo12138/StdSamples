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

#include "../../GlobalIncludes.h"
#include "../OdinConstants.h"
#include "Wavetables/Tables/LFOTableData.h"
#include "Wavetables/Tables/WavetableData.h"
#include <map>
#include <string>

class WavetableContainer {
public:


  //this class isn't used as a singleton anymore, since it messes with havning mutliple plugin instances
  //static WavetableContainer &getInstance() {
  //  static WavetableContainer instance;
  //  return instance;
  //}
  //WavetableContainer(WavetableContainer const &) = delete;
  //void operator=(WavetableContainer const &) = delete;

  WavetableContainer();
  virtual ~WavetableContainer();

  void loadWavetablesFromConstData(); // assign pointers to wavetables from
                                      // files directly
  void loadWavetablesAfterFourierCreation(); // assign pointers to wavetables
                                             // from files directly
  void createWavetables(float p_samplerate); // create and allocate memory from
                                             // coefficients and assign pointers
  void createLFOtables(float p_samplerate);

  void createChipdrawTable(int p_table_nr, float p_chipdraw_values[32],
                           float p_samplerate);
  void createWavedrawTable(int p_table_nr,
                           float p_wavedraw_values[WAVEDRAW_STEPS_X],
                           float p_samplerate, bool p_const_sections = false);
  void createSpecdrawTable(int p_table_nr,
                           float p_fourier_values[SPECDRAW_STEPS_X],
                           float p_samplerate);

  const float **getWavetablePointers(int p_wavetable);
  const float **getWavetablePointers(const std::string &p_name);
  float **getChipdrawPointer(int p_chipdraw_index);
  float **getWavedrawPointer(int p_wavedraw_index);
  float **getSpecdrawPointer(int p_specdraw_index);
  const float **getLFOPointers(const std::string &p_name);

  int getWavetableIndexFromName(const std::string &p_name);

private:
  //WavetableContainer();

protected:
// Fourrier Coefficients

  //float m_fourier_coeffs[NUMBER_OF_WAVETABLES][SIN_AND_COS]
  //                      [NUMBER_OF_HARMONICS] = {
  //                          0}; // index [x][1][0] will store scalar, since it
                                // is usually constant offset

  //the following functions generato the contribution of a const / lin segment in a function to the fourier transform for a given overtone.
  //do this for all const / lin segments in a funciton to get the fourier transform for any piecewise const or lin function
  float const_segment_one_overtone_sine(float p_start, float p_end,
                                        float p_height, int p_harmonic);
  float const_segment_one_overtone_cosine(float p_start, float p_end,
                                          float p_height, int p_harmonic);
  float lin_segment_one_overtone_sine(float p_a, float p_b, float p_fa,
                                      float p_fb, int p_ot);
  float lin_segment_one_overtone_cosine(float p_a, float p_b, float p_fa,
                                        float p_fb, int p_ot);

  std::map<std::string, int> m_name_index_map;
  std::map<std::string, int> m_LFO_name_index_map;

  //float m_LFO_fourier_coeffs[NUMBER_OF_LFOTABLES][SIN_AND_COS]
  //                          [NUMBER_OF_HARMONICS] = {
  //                              0}; // index [x][1][0] will store scalar, since
                                    // it is usually constant offset

  const float *m_const_wavetable_pointers[NUMBER_OF_WAVETABLES]
                                         [SUBTABLES_PER_WAVETABLE];
  const float *m_const_LFO_pointers[NUMBER_OF_WAVETABLES][1];

  // Wavetable pointers
  float *m_wavetable_pointers[NUMBER_OF_WAVETABLES][SUBTABLES_PER_WAVETABLE];
  float
      *m_chipdraw_pointers[NUMBER_OF_CHIPDRAW_TABLES][SUBTABLES_PER_WAVETABLE];
  float
      *m_wavedraw_pointers[NUMBER_OF_WAVEDRAW_TABLES][SUBTABLES_PER_WAVETABLE];
  float
      *m_specdraw_pointers[NUMBER_OF_SPECDRAW_TABLES][SUBTABLES_PER_WAVETABLE];
  float *m_lfotable_pointers[NUMBER_OF_LFOTABLES][1];

  // drawn tables
  float m_chipdraw_tables[NUMBER_OF_CHIPDRAW_TABLES][SUBTABLES_PER_WAVETABLE]
                         [WAVETABLE_LENGTH];
  float m_wavedraw_tables[NUMBER_OF_WAVEDRAW_TABLES][SUBTABLES_PER_WAVETABLE]
                         [WAVETABLE_LENGTH] = {0};
  float m_specdraw_tables[NUMBER_OF_SPECDRAW_TABLES][SUBTABLES_PER_WAVETABLE]
                         [WAVETABLE_LENGTH];

  // specdraw scalar (1/sqrt(harmonic))
  float m_specdraw_scalar[SPECDRAW_STEPS_X];

  std::string m_wavetable_names_1D[NUMBER_OF_WAVETABLES];
  std::string m_LFO_names[NUMBER_OF_WAVETABLES];

  bool m_wavetables_created = false;

  float ***m_wavetables; // dynamic allocation
};
