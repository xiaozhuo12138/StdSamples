#include "TX81Z_extra.h"

namespace TX81Z {
static const float frequencyRatios[64] = { 0.50, 0.71, 0.78, 0.87, 1.00, 1.41, 1.57, 1.73, 2.00, 2.82, 3.00, 3.14, 3.46, 4.00, 4.24, 4.71, 5.00, 5.19, 5.65, 6.00, 6.28, 6.92, 7.00, 7.07, 7.85, 8.00, 8.48, 8.65, 9.00, 9.42, 9.89, 10.00, 10.38, 10.99, 11.00, 11.30, 12.00, 12.11, 12.56, 12.72, 13.00, 13.84, 14.00, 14.10, 14.13, 15.00, 15.55, 15.57, 15.70, 16.96, 17.27, 17.30, 18.37, 18.84, 19.03, 19.78, 20.41, 20.76, 21.20, 21.98, 22.49, 23.55, 24.22, 25.95 };
static const float frequencyRatiosMax[64] = { 0.93, 1.32, 1.37, 1.62, 1.93, 2.73, 3.04, 3.35, 2.93, 4.14, 3.93, 4.61, 5.08, 4.93, 5.55, 6.18, 5.93, 6.81, 6.96, 6.93, 7.75, 8.54, 7.93, 8.37, 9.32, 8.93, 9.78, 10.27, 9.93, 10.89, 11.19, 10.93, 12.00, 12.46, 11.93, 12.60, 12.93, 13.73, 14.03, 14.01, 13.93, 15.46, 14.93, 15.42, 15.60, 15.93, 16.83, 17.19, 17.17, 18.24, 18.74, 18.92, 19.65, 20.31, 20.65, 21.06, 21.88, 22.38, 22.47, 23.45, 24.11, 25.02, 25.84, 27.57 };

float computeRatio(unsigned coarse, unsigned fine)
{
    float min = frequencyRatios[coarse];
    float max = frequencyRatiosMax[coarse];
    if (min < 1.0f)
        return std::min(max, min + ((max - min) / 7.0f) * fine);
    else
        return min + ((max - min) / 15.0) * fine;
}

float computeFrequency(unsigned range, unsigned coarse, unsigned fine)
{
    unsigned base = 1 << range;

    if (coarse < 4)
        fine = std::min(fine, 7u);

    if (coarse < 4)
        return (8 + fine) * base;
    else
        return (16 * (coarse / 4) + fine) * base;
}

const float dataLfoSpeedToFrequency[100] = { 0.070, 0.063, 0.064, 0.074, 0.092, 0.118, 0.151, 0.192, 0.241, 0.296, 0.358, 0.428, 0.505, 0.589, 0.681, 0.780, 0.887, 1.002, 1.126, 1.258, 1.399, 1.550, 1.710, 1.881, 2.062, 2.254, 2.458, 2.674, 2.902, 3.143, 3.397, 3.666, 3.949, 4.246, 4.559, 4.888, 5.233, 5.595, 5.974, 6.370, 6.785, 7.218, 7.669, 8.140, 8.629, 9.139, 9.668, 10.217, 10.787, 11.376, 11.987, 12.617, 13.268, 13.939, 14.630, 15.342, 16.073, 16.824, 17.594, 18.382, 19.189, 20.014, 20.856, 21.714, 22.588, 23.476, 24.379, 25.295, 26.222, 27.160, 28.107, 29.063, 30.025, 30.992, 31.963, 32.935, 33.907, 34.877, 35.842, 36.802, 37.752, 38.692, 39.618, 40.528, 41.419, 42.288, 43.133, 43.949, 44.734, 45.485, 46.198, 46.868, 47.494, 48.069, 48.591, 49.055, 49.457, 49.792, 50.055, 50.242 };

float lfoSpeedToFrequency(unsigned speed)
{
    return dataLfoSpeedToFrequency[speed];
}
}