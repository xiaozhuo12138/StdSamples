% EQUALFADES
% This script analyzes an exponential crossfade
% for "equal amplitude" or "equal power"

clear;clc;close all;
Fs = 44100;

% Square-root Fades
x = 2; % x can be any number >= 2
numOfSamples = 1*Fs; % 1 second fade in/out
aIn = linspace(0,1,numOfSamples); aIn = aIn(:);
fadeIn = (aIn).^(1/x);

aOut = linspace(1,0,numOfSamples); aOut = aOut(:);
fadeOut = (aOut).^(1/x);

% Compare Amplitude vs. Power of Cross-fade
plot(aIn,fadeIn,aIn,fadeOut,aIn,fadeIn+fadeOut,...
    aIn,(fadeIn.^2) + (fadeOut.^2));
axis([0 1 0 1.5]);
legend('Fade-in','Fade-out','Crossfade Amplitude','Crossfade Power');

% This source code is provided without any warranties as published in 
% "Hack Audio: An Introduction to Computer Programming and Digital Signal
% Processing in MATLAB" � 2019 Taylor & Francis.
% 
% It may be used for educational purposes, but not for commercial 
% applications, without further permission.
%
% Book available here (uncomment):
% url = ['https://www.routledge.com/Hack-Audio-An-Introduction-to-'...
% 'Computer-Programming-and-Digital-Signal/Tarr/p/book/9781138497559'];
% web(url,'-browser');
% 
% Companion website resources (uncomment):
% url = 'http://www.hackaudio.com'; 
% web(url,'-browser');

