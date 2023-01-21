% EXPONENTIAL
% This function implements exponential soft-clipping
% distortion. An input parameter "G" is used 
% to control the amount of distortion applied 
% to the input signal.
%
% Input variables
%   in : input signal
%   G : drive amount (1-10)
%
% See also CUBICDISTORTION, ARCTANDISTORTION, DISTORTIONEXAMPLE

function [ out ] = exponential(in,G)

N = length(in);
out = zeros(N,1);
for n = 1:N
    
    out(n,1) = sign(in(n,1)) * (1 - exp(-abs(G*in(n,1))));
    
end

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
