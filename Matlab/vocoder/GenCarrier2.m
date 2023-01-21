function [array_final,time_final]=GenCarrier2(f0, hmin, hmax, fs, duration, phases, scalenew, k)
%[array,time]=GenCarrier2(f0, hmin, hmax, fs, duration, phases, scalenew, k)
%
%Generates a harmonic complex with consecutive harmonic number hmin to hmax
%and also choice of phases. Shaped by digital 8th order Butterworth.
%
%input:
%       f0 = fundamental frequency (Hz)
%     hmin = minimum harmonic number
%     hmax = maximum harmonic number
%       fs = sampling frequency (Hz)
% duration = duration in seconds
%   phases =
%       0, 'sin', 'sine'                                => sin phase
%       1, 'cos', 'cosine'                              => cos phase
%       2, 'alt'                                        => alternating, odd: cos, even: sin
%       3, 'rand', 'random'                             => random
%       4, 'pshc'                                       => pulse-spreading harmonic complex
%       5, 'sch', 'sch+', 'schroeder'                   => positive Schroeder
%       6, 'sch-', 'schroeder-'                         => negative Schroeder
% scalenew = scaling
%   k = order of PSHC. only used for PSHC generation. Gives an effective rate
%            equal to F0 * k^2
%
%output:
%    array (values)
%    time (in seconds)


%---------
% This file is part of
% vocoder: a versatile Matlab vocoder for research purposes
% Copyright (C) 2013 LMA CNRS, Olivier Macherey / CRNL CNRS, UMCG, Etienne Gaudrain
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
%---------
% See README.md for rules on citations.
%---------

% modified from HARMCOMP to input (software) filter cutoffs & slopes
% based on Bob Carlyon's Fortran harmcomp3.for
% dragged into 21st century by CJL (Fri, May 17 2002, 14:20:28 ; 137/366)
% then modified by Bob to output real array with values between -1.0 and +1.0
% th eresulting HRMCOMPREAL modified here to HRMCOMPDIG to do all filtering 
% digitally

% modified from hrmcompdig_filttimes.m to start the stimulus in between 2 periods (so that every pulses
% are identical).

% Modified on 23/11/2011 by Olivier Macherey to allow PSHC
% generation.
% Any publication with results obtained using the PSHC option should cite the following:
%
% Hilkhuysen, G., Mesnildrey, Q., and Macherey, O. (2013) "Pulse-spreading
% harmonic complexes: an advantageous carrier for simulating cochlear
% implants." Conference on implantable auditory prostheses, Granlibakken
% Conference Center, CA, USA.
% 
% AND (when/if accepted for publication)
%
% Hilkhuysen, G., and Macherey, O. "Optimizing pulse-spreading harmonic
% complexes to minimize intrinsic modulations after cochlear filtering",
% submitted to JASA.

% Modified on 09/09/2013 by Etienne Gaudrain to change argument handling,
% phase generation (outside the loop), ...

%rand('state',sum(100*clock)) %set random number state in case we use random phases
% tic
global clip
clip=0;

halfpi=pi*0.5;

halfperiod_samples=round(fs/f0/4);
% add half a pulse train period at the beginning and at the end of the pulse train
nopts=round(duration*fs)+2*halfperiod_samples; %*[ ] round equiv to fortran's nint?

temparray=zeros(nopts,1);

n=1:nopts;
time=(n./fs)';

% = 0 => sine
% = 1 => cos
% = 2 => ALT: odd=cos even=sin
% = 3 => random
% = 4 => PSHC
% = 5 => schroeder

hn = hmin:hmax;
freq = f0*hn;

switch phases
    case {0, 'sine', 'sin'}
        phase = zeros(size(freq));

    case {1, 'cos', 'cosine'}
        phase = ones(size(freq))*halfpi;

    case {2, 'alt'}
        phase = ones(size(freq))*halfpi;
        phase(mod((hmin:hmax),2)==0) = 0;

    case {3, 'rand', 'random'}
        phase=rand(size(freq))*2*pi;

    case {4, 'pshc'}
        
        if nargin<8
            error('PSHC order (i.e. k) must be provided for rate-expanding phase');
        end
        
        r=randperm(k)-1;
        u=rand(1,k);
        
        phase = zeros(size(freq));
        for j = 0:1:k-1
            s = mod((hn+j)/k,1)==0;
            phase(s) = 2*pi*(hn(s)*r(j+1)/k^2+u(j+1));
        end

    case {5, 'schroeder', 'sch+', 'sch'}
        phase=pi*hn.*(hn+1)/round(((hmax-hmin)/f0)+1);
    
    case {5, 'schroeder-', 'sch-'}
        phase=-pi*hn.*(hn+1)/round(((hmax-hmin)/f0)+1);

    otherwise
        error('Invalid value for PHASES');
end

for i=1:length(hn)
    temparray = temparray + sin((2.*pi*time*freq(i))+phase(i));
end

numcomps=hmax-hmin+1;
scale=scalenew./numcomps;
array=temparray*scale;

if(any(abs(array)>1.0))
    warning('Clipping!');
    clip=1;
    return;  %*[ ] equiv performance in matlab and fortran?
end

%now apply "filtorder"  order Butterworth filter. 

%Now do it in 2 stages - lowpass then highpass
% Wn=[cutlo]/(0.5*fs);
% [b,a]=butter(8,Wn,'high');
% array=filter(b,a,array);
% Wn=[cuthi]/(0.5*fs);
% [b,a]=butter(8,Wn);
% array=filter(b,a,array);

% Extract the desired duration of the pulse by removing half a period at
% the beginning and half a period at the end
nopts_final=1:round(duration*fs);
time_final=nopts_final./fs;

array_final = array(halfperiod_samples+1:1:end-halfperiod_samples);

% toc