// [src] https://github.com/ocornut/imgui/issues/123
// [src] https://github.com/ocornut/imgui/issues/55

//v2 - added data, chaged curve to gimp code, some bug fixes
// v1.22 - flip button; cosmetic fixes
// v1.21 - oops :)
// v1.20 - add iq's interpolation code
// v1.10 - easing and colors        
// v1.00 - jari komppa's original

//#pragma once

#include "imgui.h"

#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui_internal.h"
#include <algorithm> 
//#include <cmath>

/* To use, add this prototype somewhere.. 
namespace ImGui
{
    int Curve(const char *label, const ImVec2& size, ImVec2 *points, int maxpoints, float *data, int datalen);
	void CurveGetData(ImVec2 *points, const int maxpoints, float *data, int datalen);
};
*/
/*
    Example of use:
    ImVec2 points[10];
	float data[256];
    ...
    points[0].x = -1; // init points data so editor knows to take it from here
	data[0].x = -1; //force data calculation without user modify
    ...
    if (ImGui::Curve("Das editor", ImVec2(600, 200), points, 10, data,256))
    {
        // curve changed
		//data can be used
    }

*/

namespace tween {
    enum TYPE
    {
        LINEAR,

        QUADIN,          // t^2
        QUADOUT,
        QUADINOUT,
        CUBICIN,         // t^3
        CUBICOUT,
        CUBICINOUT,
        QUARTIN,         // t^4
        QUARTOUT,
        QUARTINOUT,
        QUINTIN,         // t^5
        QUINTOUT,
        QUINTINOUT,
        SINEIN,          // sin(t)
        SINEOUT,
        SINEINOUT,
        EXPOIN,          // 2^t
        EXPOOUT,
        EXPOINOUT,
        CIRCIN,          // sqrt(1-t^2)
        CIRCOUT,
        CIRCINOUT,
        ELASTICIN,       // exponentially decaying sine wave
        ELASTICOUT,
        ELASTICINOUT,
        BACKIN,          // overshooting cubic easing: (s+1)*t^3 - s*t^2
        BACKOUT,
        BACKINOUT,
        BOUNCEIN,        // exponentially decaying parabolic bounce
        BOUNCEOUT,
        BOUNCEINOUT,

        SINESQUARE,      // gapjumper's
        EXPONENTIAL,     // gapjumper's
        SCHUBRING1,      // terry schubring's formula 1
        SCHUBRING2,      // terry schubring's formula 2
        SCHUBRING3,      // terry schubring's formula 3

        SINPI2,          // tomas cepeda's
        SWING,           // tomas cepeda's & lquery's
    };

    // }

    // implementation

    static inline
    float ease( int easetype, float t )
    {
        using namespace std;

        const float d = 1.f;
        const float pi = 3.1415926535897932384626433832795;
        const float pi2 = 3.1415926535897932384626433832795 / 2;

        float p = t/d;

        switch( easetype )
        {
            // Modeled after the line y = x
            default:
            case LINEAR: {
                return p;
            }

            // Modeled after the parabola y = x^2
            case QUADIN: {
                return p * p;
            }

            // Modeled after the parabola y = -x^2 + 2x
            case QUADOUT: {
                return -(p * (p - 2));
            }

            // Modeled after the piecewise quadratic
            // y = (1/2)((2x)^2)             ; [0, 0.5)
            // y = -(1/2)((2x-1)*(2x-3) - 1) ; [0.5, 1]
            case QUADINOUT: {
                if(p < 0.5) {
                    return 2 * p * p;
                }
                else {
                    return (-2 * p * p) + (4 * p) - 1;
                }
            }

            // Modeled after the cubic y = x^3
            case CUBICIN: {
                return p * p * p;
            }

            // Modeled after the cubic y = (x - 1)^3 + 1
            case CUBICOUT: {
                double f = (p - 1);
                return f * f * f + 1;
            }

            // Modeled after the piecewise cubic
            // y = (1/2)((2x)^3)       ; [0, 0.5)
            // y = (1/2)((2x-2)^3 + 2) ; [0.5, 1]
            case CUBICINOUT: {
                if(p < 0.5) {
                    return 4 * p * p * p;
                }
                else {
                    double f = ((2 * p) - 2);
                    return 0.5 * f * f * f + 1;
                }
            }

            // Modeled after the quartic x^4
            case QUARTIN: {
                return p * p * p * p;
            }

            // Modeled after the quartic y = 1 - (x - 1)^4
            case QUARTOUT: {
                double f = (p - 1);
                return f * f * f * (1 - p) + 1;
            }

            // Modeled after the piecewise quartic
            // y = (1/2)((2x)^4)        ; [0, 0.5)
            // y = -(1/2)((2x-2)^4 - 2) ; [0.5, 1]
            case QUARTINOUT:  {
                if(p < 0.5) {
                    return 8 * p * p * p * p;
                }
                else {
                    double f = (p - 1);
                    return -8 * f * f * f * f + 1;
                }
            }

            // Modeled after the quintic y = x^5
            case QUINTIN:  {
                return p * p * p * p * p;
            }

            // Modeled after the quintic y = (x - 1)^5 + 1
            case QUINTOUT:  {
                double f = (p - 1);
                return f * f * f * f * f + 1;
            }

            // Modeled after the piecewise quintic
            // y = (1/2)((2x)^5)       ; [0, 0.5)
            // y = (1/2)((2x-2)^5 + 2) ; [0.5, 1]
            case QUINTINOUT:  {
                if(p < 0.5) {
                    return 16 * p * p * p * p * p;
                }
                else {
                    double f = ((2 * p) - 2);
                    return  0.5 * f * f * f * f * f + 1;
                }
            }

            // Modeled after quarter-cycle of sine wave
            case SINEIN: {
                return sin((p - 1) * pi2) + 1;
            }

            // Modeled after quarter-cycle of sine wave (different phase)
            case SINEOUT: {
                return sin(p * pi2);
            }

            // Modeled after half sine wave
            case SINEINOUT: {
                return 0.5 * (1 - cos(p * pi));
            }

            // Modeled after shifted quadrant IV of unit circle
            case CIRCIN: {
                return 1 - sqrt(1 - (p * p));
            }

            // Modeled after shifted quadrant II of unit circle
            case CIRCOUT: {
                return sqrt((2 - p) * p);
            }

            // Modeled after the piecewise circular function
            // y = (1/2)(1 - sqrt(1 - 4x^2))           ; [0, 0.5)
            // y = (1/2)(sqrt(-(2x - 3)*(2x - 1)) + 1) ; [0.5, 1]
            case CIRCINOUT: {
                if(p < 0.5) {
                    return 0.5 * (1 - sqrt(1 - 4 * (p * p)));
                }
                else {
                    return 0.5 * (sqrt(-((2 * p) - 3) * ((2 * p) - 1)) + 1);
                }
            }

            // Modeled after the exponential function y = 2^(10(x - 1))
            case EXPOIN: {
                return (p == 0.0) ? p : pow(2, 10 * (p - 1));
            }

            // Modeled after the exponential function y = -2^(-10x) + 1
            case EXPOOUT: {
                return (p == 1.0) ? p : 1 - pow(2, -10 * p);
            }

            // Modeled after the piecewise exponential
            // y = (1/2)2^(10(2x - 1))         ; [0,0.5)
            // y = -(1/2)*2^(-10(2x - 1))) + 1 ; [0.5,1]
            case EXPOINOUT: {
                if(p == 0.0 || p == 1.0) return p;
                
                if(p < 0.5) {
                    return 0.5 * pow(2, (20 * p) - 10);
                }
                else {
                    return -0.5 * pow(2, (-20 * p) + 10) + 1;
                }
            }

            // Modeled after the damped sine wave y = sin(13pi/2*x)*pow(2, 10 * (x - 1))
            case ELASTICIN: {
                return sin(13 * pi2 * p) * pow(2, 10 * (p - 1));
            }

            // Modeled after the damped sine wave y = sin(-13pi/2*(x + 1))*pow(2, -10x) + 1
            case ELASTICOUT: {
                return sin(-13 * pi2 * (p + 1)) * pow(2, -10 * p) + 1;
            }

            // Modeled after the piecewise exponentially-damped sine wave:
            // y = (1/2)*sin(13pi/2*(2*x))*pow(2, 10 * ((2*x) - 1))      ; [0,0.5)
            // y = (1/2)*(sin(-13pi/2*((2x-1)+1))*pow(2,-10(2*x-1)) + 2) ; [0.5, 1]
            case ELASTICINOUT: {
                if(p < 0.5) {
                    return 0.5 * sin(13 * pi2 * (2 * p)) * pow(2, 10 * ((2 * p) - 1));
                }
                else {
                    return 0.5 * (sin(-13 * pi2 * ((2 * p - 1) + 1)) * pow(2, -10 * (2 * p - 1)) + 2);
                }
            }

            // Modeled (originally) after the overshooting cubic y = x^3-x*sin(x*pi)
            case BACKIN: { /*
                return p * p * p - p * sin(p * pi); */
                double s = 1.70158f;
                return p * p * ((s + 1) * p - s);
            }

            // Modeled (originally) after overshooting cubic y = 1-((1-x)^3-(1-x)*sin((1-x)*pi))
            case BACKOUT: { /*
                double f = (1 - p);
                return 1 - (f * f * f - f * sin(f * pi)); */
                double s = 1.70158f;
                return --p, 1.f * (p*p*((s+1)*p + s) + 1);
            }

            // Modeled (originally) after the piecewise overshooting cubic function:
            // y = (1/2)*((2x)^3-(2x)*sin(2*x*pi))           ; [0, 0.5)
            // y = (1/2)*(1-((1-x)^3-(1-x)*sin((1-x)*pi))+1) ; [0.5, 1]
            case BACKINOUT: { /*
                if(p < 0.5) {
                    double f = 2 * p;
                    return 0.5 * (f * f * f - f * sin(f * pi));
                }
                else {
                    double f = (1 - (2*p - 1));
                    return 0.5 * (1 - (f * f * f - f * sin(f * pi))) + 0.5;
                } */
                double s = 1.70158f * 1.525f;
                if (p < 0.5) {
                    return p *= 2, 0.5 * p * p * (p*s+p-s);
                }
                else {
                    return p = p * 2 - 2, 0.5 * (2 + p*p*(p*s+p+s));
                }
            }

#           define tween$bounceout(p) ( \
                (p) < 4/11.0 ? (121 * (p) * (p))/16.0 : \
                (p) < 8/11.0 ? (363/40.0 * (p) * (p)) - (99/10.0 * (p)) + 17/5.0 : \
                (p) < 9/10.0 ? (4356/361.0 * (p) * (p)) - (35442/1805.0 * (p)) + 16061/1805.0 \
                           : (54/5.0 * (p) * (p)) - (513/25.0 * (p)) + 268/25.0 )

            case BOUNCEIN: {
                return 1 - tween$bounceout(1 - p);
            }

            case BOUNCEOUT: {
                return tween$bounceout(p);
            }

            case BOUNCEINOUT: {
                if(p < 0.5) {
                    return 0.5 * (1 - tween$bounceout(1 - p * 2));
                }
                else {
                    return 0.5 * tween$bounceout((p * 2 - 1)) + 0.5;
                }
            }

#           undef tween$bounceout

            case SINESQUARE: {
                double A = sin((p)*pi2);
                return A*A;
            }

            case EXPONENTIAL: {
                return 1/(1+exp(6-12*(p)));                
            }

            case SCHUBRING1: {
                return 2.0f*(p+(0.5f-p)*fabs(0.5f-p))-0.5f;                
            }

            case SCHUBRING2: {
                double p1pass= 2*(p+(0.5f-p)*fabs(0.5f-p))-0.5f;
                double p2pass= 2*(p1pass+(0.5f-p1pass)*fabs(0.5f-p1pass))-0.5f;
                double pAvg=(p1pass+p2pass)/2;
                return pAvg;
            }

            case SCHUBRING3: {
                double p1pass= 2*(p+(0.5f-p)*fabs(0.5f-p))-0.5f;
                double p2pass= 2*(p1pass+(0.5f-p1pass)*fabs(0.5f-p1pass))-0.5f;
                return p2pass;
            }

            case SWING: {
                return ((-cos(pi * p) * 0.5) + 0.5);
            }

            case SINPI2: {
                return sin(p * pi2);
            }
        }
    }   
}

namespace ImGui
{
    // [src] http://iquilezles.org/www/articles/minispline/minispline.htm
    // key format (for dim == 1) is (t0,x0,t1,x1 ...)
    // key format (for dim == 2) is (t0,x0,y0,t1,x1,y1 ...)
    // key format (for dim == 3) is (t0,x0,y0,z0,t1,x1,y1,z1 ...)
    void spline( const float *key, int num, int dim, float t, float *v )
    {
        static signed char coefs[16] = {
            -1, 2,-1, 0,
             3,-5, 0, 2,
            -3, 4, 1, 0,
             1,-1, 0, 0 };

        const int size = dim + 1;

        // find key
        int k = 0; while( key[k*size] < t ) k++;

        // interpolant
        const float h = (t-key[(k-1)*size])/(key[k*size]-key[(k-1)*size]);

        // init result
        for( int i=0; i < dim; i++ ) v[i] = 0.0f;

        // add basis functions
        for( int i=0; i<4; i++ )
        {
            int kn = k+i-2; if( kn<0 ) kn=0; else if( kn>(num-1) ) kn=num-1;

            const signed char *co = coefs + 4*i;

            const float b  = 0.5f*(((co[0]*h + co[1])*h + co[2])*h + co[3]);

            for( int j=0; j < dim; j++ ) v[j] += b * key[kn*size+j+1];
        }
    }
	void BarycentricWeights(int max, const ImVec2 *points, float * w)
	{
		for(int i = 0; i<max; i++){
			w[i] = 1.0;
			for(int j=0;j<max; j++){
				if(i!=j)
					w[i] *= points[j].x - points[i].x;
			}
			w[i] = 1.0/w[i];	
		}
	}
	//polinomial interpolation. oscilates wildly
	float BarycentricSecondForm(float p,int max, const ImVec2 *points, float* w)
	{
		
		if (p <= points[0].x) return points[0].y;
		if (p >= points[max -1].x) return points[max - 1].y;
		
		float num = 0.0;
		float den = 0.0;
		for(int i=0; i<max; i++){
			float fac = w[i]/(p - points[i].x);
			num += fac * points[i].y;
			den += fac;
		}
		return num/den;
	}
	//catmull rom based on iñigo quilez
    float CurveValueSmooth(float p, int maxpoints, const ImVec2 *points)
    {
        if (points == 0)
            return 0;
        //if (p < 0) return points[0].y;
		if (p <= points[0].x) return points[0].y;
		if (p >= points[maxpoints -1].x) return points[maxpoints - 1].y;

        float *input = new float [ maxpoints * 2 ];
        float output[1];

        for( int i = 0; i < maxpoints; ++i ) {
            input[ i * 2 + 0 ] = points[i].x;
            input[ i * 2 + 1 ] = points[i].y;
        }

        spline( input, maxpoints, 1, p, output );
        
        delete [] input;
        return output[0];
    }
	
	//linear interpolation
    float CurveValue(float p, int maxpoints, const ImVec2 *points, int max)
    {
        if (maxpoints < 2 || points == 0)
            return 0;
       // if (p < 0) return points[0].y;
		if (p <= points[0].x) return points[0].y;
		if (p >= points[max -1].x) return points[max - 1].y;
		
        int left = 0;
        while (left < maxpoints && points[left].x < p && points[left].x != -1) left++;
        if (left) left--;

        if (left == maxpoints-1)
            return points[maxpoints - 1].y;

        float d = (p - points[left].x) / (points[left + 1].x - points[left].x);

        return points[left].y + (points[left + 1].y - points[left].y) * d;
    }
	
	
	void CalcCurves(ImVec2 *points, int max, float *data,int datalen)
	{
		//float weights[max];
		//BarycentricWeights(max, points, weights);

        for(int i = 0; i < datalen; ++i ) {
			float px = float(i)/float(datalen -1);
			data[i] = CurveValue(px, max, points, max);
			//data[i] = CurveValueSmooth(px, max, points);
			//data[i] = BarycentricSecondForm(px, max, points, weights);
        } 
	}
	/*
	* This function calculates the curve values between the control points
	* p2 and p3, taking the potentially existing neighbors p1 and p4 into
	* account.
	*
	* This function uses a cubic bezier curve for the individual segments and
	* calculates the necessary intermediate control points depending on the
	* neighbor curve control points.
	*/

	void gimp_curve_plot (ImVec2 *points,
					int       p1,
					int       p2,
					int       p3,
					int       p4,
					float *data, int datalen)
	{
	int    i;
	float x0, x3;
	float y0, y1, y2, y3;
	float dx, dy;
	float slope;
	
	/* the outer control points for the bezier curve. */
	x0 = points[p2].x;
	y0 = points[p2].y;
	x3 = points[p3].x;
	y3 = points[p3].y;
	
	/*
	* the x values of the inner control points are fixed at
	* x1 = 2/3*x0 + 1/3*x3   and  x2 = 1/3*x0 + 2/3*x3
	* this ensures that the x values increase linearily with the
	* parameter t and enables us to skip the calculation of the x
	* values altogehter - just calculate y(t) evenly spaced.
	*/
	
	dx = x3 - x0;
	dy = y3 - y0;
	
	assert(dx > 0);
	
	if (p1 == p2 && p3 == p4)
		{
		/* No information about the neighbors,
		* calculate y1 and y2 to get a straight line
		*/
		y1 = y0 + dy / 3.0;
		y2 = y0 + dy * 2.0 / 3.0;
		}
	else if (p1 == p2 && p3 != p4)
		{
		/* only the right neighbor is available. Make the tangent at the
		* right endpoint parallel to the line between the left endpoint
		* and the right neighbor. Then point the tangent at the left towards
		* the control handle of the right tangent, to ensure that the curve
		* does not have an inflection point.
		*/
		slope = (points[p4].y - y0) / (points[p4].x - x0);
	
		y2 = y3 - slope * dx / 3.0;
		y1 = y0 + (y2 - y0) / 2.0;
		}
	else if (p1 != p2 && p3 == p4)
		{
		/* see previous case */
		slope = (y3 - points[p1].y) / (x3 - points[p1].x);
	
		y1 = y0 + slope * dx / 3.0;
		y2 = y3 + (y1 - y3) / 2.0;
		}
	else /* (p1 != p2 && p3 != p4) */
		{
		/* Both neighbors are available. Make the tangents at the endpoints
		* parallel to the line between the opposite endpoint and the adjacent
		* neighbor.
		*/
		slope = (y3 - points[p1].y) / (x3 - points[p1].x);
	
		y1 = y0 + slope * dx / 3.0;
	
		slope = (points[p4].y - y0) / (points[p4].x - x0);
	
		y2 = y3 - slope * dx / 3.0;
		}
	
		/*
		* finally calculate the y(t) values for the given bezier values. We can
		* use homogenously distributed values for t, since x(t) increases linearily.
		*/
		for (i = 0; i <= int (dx * (float) (datalen - 1) + 0.5); i++)
		{
			float y, t;
			int    index;
		
			t = i / dx / (float) (datalen - 1);
			y =     y0 * (1-t) * (1-t) * (1-t) +
				3 * y1 * (1-t) * (1-t) * t     +
				3 * y2 * (1-t) * t     * t     +
					y3 * t     * t     * t;
		
			index = i + int (x0 * (float) (datalen - 1) + 0.5);
		
			if (index < datalen)
				data[index] = std::max(0.0f, std::min(y, 1.0f));;
		}
	}
	
	void CalcCurvesGimp(ImVec2 *points, int max, float *data,int datalen)
	{
		//before x0
        int boundary = int (points[0].x * (float) (datalen - 1) + 0.5);
		for(int i=0; i<boundary; i++)
			data[i] = points[0].y;
		//after xn
		int boundary2 = int (points[max - 1].x * (float) (datalen - 1) + 0.5);
		for(int i=boundary2; i < datalen ; i++)
			data[i] = points[max - 1].y;
		
		int  p1, p2, p3, p4;
		for (int i = 0; i < max - 1; i++)
        {
          p1 = std::max (i - 1, 0);
          p2 = i;
          p3 = i + 1;
          p4 = std::min (i + 2, max - 1);

          gimp_curve_plot (points, p1, p2, p3, p4, data, datalen);
        }
		
	}
	
	
	int CurveGetMaxPoints(ImVec2 *points, const int maxpoints,bool &modified)
	{
		int max = 0;
        while (max < maxpoints && points[max].x >= 0) max++;
		
		ImVec2 p;
        int kill = 0;
        do
        {
            if (kill)
            {
				modified = true;
                for (int i = kill + 1; i < max; i++)
                {
                    points[i - 1] = points[i];
                }
                max--;
                points[max].x = -1;
                kill = 0;
            }

            for (int i = 1; i < max ; i++)
            {
				p = points[i] - points[i - 1];
				float dis = sqrt(p.x*p.x+p.y*p.y);
				if (dis < 1.0/128.0)
				// if (fabs(points[i].x - points[i - 1].x) < 1.0 / 128.0)
                {
                    kill = i;
                }
            }
        }
        while (kill);
		return max;
	}
	
	void CurveGetData(ImVec2 *points, const int maxpoints, float *data, int datalen)
	{
		if (points[0].x < 0)
        {
			points[0].x = 0;
            points[0].y = 0;
            points[1].x = 1;
            points[1].y = 1;
            points[2].x = -1;
        }
		bool dummy;
		int max = CurveGetMaxPoints(points, maxpoints,dummy);
		CalcCurvesGimp(points, max, data, datalen);
	}
	
    bool Curve(const char *label, const ImVec2& size, ImVec2 *points, const int maxpoints, float *data, int datalen,bool pressed_on_modified)
    {
        bool modified = false;
        int i;
        if (maxpoints < 2 || points == 0)
            return false;

        if (data[0] < 0)
			modified = true;
		if (points[0].x < 0)
        {
            modified = true;
			points[0].x = 0;
            points[0].y = 0;
            points[1].x = 1;
            points[1].y = 1;
            points[2].x = -1;
        }

        ImGuiWindow* window = GetCurrentWindow();
        ImGuiContext& g = *GImGui;
        const ImGuiStyle& style = g.Style;
        const ImGuiID id = window->GetID(label);
        if (window->SkipItems)
            return false;

        ImRect bb(window->DC.CursorPos, window->DC.CursorPos + size);
        ItemSize(bb);
        if (!ItemAdd(bb, 0))
            return false;

        //const bool hovered = ItemHoverable(bb, id);
		bool hovered, held;
		bool pressed = ButtonBehavior(bb, id, &hovered, &held, ImGuiButtonFlags_PressedOnClickReleaseAnywhere );

		int max = CurveGetMaxPoints(points,maxpoints,modified);


        RenderFrame(bb.Min, bb.Max, GetColorU32(ImGuiCol_FrameBg, 1), true, style.FrameRounding);

        float ht = bb.Max.y - bb.Min.y;
        float wd = bb.Max.x - bb.Min.x;

        if (hovered)
        {
            SetHoveredID(id);
            if (g.IO.MouseDown[0])
            {
				modified = true;
                ImVec2 pos = (g.IO.MousePos - bb.Min) / (bb.Max - bb.Min);
                pos.y = 1 - pos.y;              

                int left = 0;
                while (left < max && points[left].x < pos.x) left++;
                //if (left) 
				left--;
				
				int sel = -1;
				ImVec2 p;
				if (left > -1){
					p = points[left] - pos;
					float p1d = sqrt(p.x*p.x + p.y*p.y);
					if (p1d < (1 / 16.0)) sel = left;
				}
				p = points[left+1] - pos;
				float p2d = sqrt(p.x*p.x + p.y*p.y);
				if (p2d < (1 / 16.0)) sel = left + 1;
				
                if (sel != -1)
                {
                    points[sel] = pos;
                }
                else
                {   
                    if (g.IO.MouseDownDuration[0] == 0.0){ //just clicked
					if (max < maxpoints)
                    {
                        max++;
                        for (i = max-1; i > left + 1; i--)
                        {
                            points[i] = points[i - 1];
                        }
                        points[left + 1] = pos;

                    }
                    if (max < maxpoints)
                        points[max].x = -1;
					}
                }

            }
        }

        // bg grid

        for (i = 0; i < 9; i++)
        {
            window->DrawList->AddLine(
                ImVec2(bb.Min.x + (wd / 10) * (i + 1), bb.Min.y),
                ImVec2(bb.Min.x + (wd / 10) * (i + 1), bb.Max.y),
                GetColorU32(ImGuiCol_TextDisabled));
        }   
		
		for (i = 0; i < 9; i++)
        {
            float height =  bb.Min.y + (ht / 10) * (i + 1);
			window->DrawList->AddLine(
				ImVec2(bb.Min.x , height),
				ImVec2(bb.Max.x , height),
				GetColorU32(ImGuiCol_TextDisabled));
        }   
		
		if(modified)
			CalcCurvesGimp(points, max, data, datalen);
		//show curves
		for( i = 0; i < datalen - 1; ++i ) {
			float py = 1 - data[i];
			float qy = 1 - data[i+1];
			float px = float(i)/float(datalen -1);
			float qx = float(i + 1)/float(datalen -1);
			ImVec2 p( px * (bb.Max.x - bb.Min.x) + bb.Min.x, py * (bb.Max.y - bb.Min.y) + bb.Min.y);
            ImVec2 q( qx * (bb.Max.x - bb.Min.x) + bb.Min.x, qy * (bb.Max.y - bb.Min.y) + bb.Min.y);
            window->DrawList->AddLine(p, q, GetColorU32(ImGuiCol_PlotLines));
		}

        // lines
        for (i = 1; i < max; i++)
        {
            ImVec2 a = points[i - 1];
            ImVec2 b = points[i];
            a.y = 1 - a.y;
            b.y = 1 - b.y;
            a = a * (bb.Max - bb.Min) + bb.Min;
            b = b * (bb.Max - bb.Min) + bb.Min;
            window->DrawList->AddLine(a, b, GetColorU32(ImGuiCol_PlotLinesHovered));
        }

        //if (hovered)
        //{
            // control points
            for (i = 0; i < max; i++)
            {
                ImVec2 p = points[i];
                p.y = 1 - p.y;
                p = p * (bb.Max - bb.Min) + bb.Min;
                ImVec2 a = p - ImVec2(2, 2);
                ImVec2 b = p + ImVec2(2, 2);
                window->DrawList->AddRect(a, b, GetColorU32(ImGuiCol_PlotLinesHovered));
            }
       // }

        // buttons; 
		bool combomodified = false;
		ImGui::PushID(label);
        if( ImGui::Button("Flip") ) {
            for( i = 0; i < max; ++i) { 
                points[i].y = 1 - points[i].y;
            }
			combomodified = true;
        }
		ImGui::PopID();
        ImGui::SameLine();

        // curve selector
        const char* items[] = { 
            "Custom", 

            "Linear",
            "Quad in",
            "Quad out",
            "Quad in  out",
            "Cubic in",
            "Cubic out",
            "Cubic in  out",
            "Quart in",
            "Quart out",
            "Quart in  out",
            "Quint in",
            "Quint out",
            "Quint in  out",
            "Sine in",
            "Sine out",
            "Sine in  out",
            "Expo in",
            "Expo out",
            "Expo in  out",
            "Circ in",
            "Circ out",
            "Circ in  out",
            "Elastic in",
            "Elastic out",
            "Elastic in  out",
            "Back in",
            "Back out",
            "Back in  out",
            "Bounce in",
            "Bounce out",
            "Bounce in out",

            "Sine square",
            "Exponential",

            "Schubring1",
            "Schubring2",
            "Schubring3",

            "SinPi2",
            "Swing"
        };
        int item = 0;
        if( modified ) {
            item = 0;
        }
		
		ImGui::PushID(label);
        if( ImGui::Combo("E", &item, items, IM_ARRAYSIZE(items)) ) {
            max = maxpoints;
			if (item == 1){ //linear
				max = 2;
			}
            if( item > 0 ) {
                for( i = 0; i < max; ++i) { 
                    points[i].x = i / float(max-1); 
                    points[i].y = float( tween::ease( item - 1, points[i].x ) );
                }
				if (max < maxpoints)
                    points[max].x = -1;
				combomodified = 1;			
            }
        }
		ImGui::PopID();
        char buf[128];
        const char *str = label;

        if( hovered ) {
            ImVec2 pos = (g.IO.MousePos - bb.Min) / (bb.Max - bb.Min);
            pos.y = 1 - pos.y;              

            sprintf(buf, "%s (%f,%f)", label, pos.x, pos.y );
            str = buf;
        }

        RenderTextClipped(ImVec2(bb.Min.x, bb.Min.y + style.FramePadding.y), bb.Max, str, NULL, NULL);//, ImGuiAlign_Center);
		if(combomodified){
			CalcCurvesGimp(points, max, data, datalen);
			pressed = true;
			return true;
		}
		if(pressed_on_modified)
			pressed = modified;
        return pressed;
    }

};

IMGUI_IMPL_API bool Curve(const char *label, const ImVec2& size, ImVec2 *points, const int maxpoints, float *data, int datalen,bool pressed_on_modified)
{
	return ImGui::Curve(label, size, points, maxpoints, data, datalen,pressed_on_modified);
}

IMGUI_IMPL_API void CurveGetData(ImVec2 *points, const int maxpoints, float *data, int datalen)
{
	return ImGui::CurveGetData(points, maxpoints, data, datalen);
}