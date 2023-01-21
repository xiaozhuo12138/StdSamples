function FIR_highpass = HighpassFilter(As, Fs, fs, fp, Kaiser)
%FIR Highpass Filter design
% ARGUMENTS
%           As: stopband attenuation - dB (scalar)
%           Fs: sampling frequency - Hz (scalar)
%           fp: passband edge - Hz (scalar)
%           fs: stopband edge - Hz (scalar)
%           Kaiser: flag for using Kaiser window - 1 or 0 (scalar)
% RETURNS
%           FIR_highpass: FIR highpass filter coefficients

% Cut-off frequency
fc = (fs + fp)/2;

% Transition band (rad/sec)
Tb = 2*pi*(fp - fs)/Fs;

% Choice of window function based on stopband attenuation
if Kaiser == 0
    if As <= 21 % Rectangular
        N = ceil((1.8*pi/Tb)); % Filter Order
        w = rectwin(N)';  % Window function

    elseif As > 21 && As <= 26 % Bartlett
        N = ceil((6.1*pi/Tb));  % Filter Order
        w = bartlett(N)';  % Window function

    elseif As > 26 && As <= 44 % Hann
        N = ceil((6.2*pi/Tb));  % Filter Order
        w = hann(N)';  % Window function

    elseif As > 44 && As <= 53 % Hamming
        N = ceil((6.6*pi/Tb));  % Filter Order
        w = hamming(N)';  % Window function

    elseif As > 53 && As <= 74 % Blackman
        N = ceil((11*pi/Tb));  % Filter Order
        w = blackman(N)';  % Window function
    end
else
    % Beta estimation
    if As > 50
        beta = 0.1102*(As - 8.7);
    elseif 21 < As && As <= 50
        beta = 0.5842*(As - 21)^0.4 + 0.07886*(As-21);
    else
        beta = 0;
    end

    % Filter Order
    N = ceil((As - 8)/(2.285*Tb));

    % Kaiser window function, w[n]
    w = besseli(0, beta*sqrt(1-(((0:N-1)-N/2)/(N/2)).^2))/besseli(0, beta);
end

% Ideal impulse response of highpass filter
alpha = N/2;
h_id = zeros(1, N);
for n = 0 : N-1
    if n == alpha
        h_id(n+1) = 1 - 2*fc/Fs;
    else
        h_id(n+1) = -sin(2*pi*fc/Fs*(n-alpha))./(pi*(n-alpha));
    end
end

% Multiplying the ideal filter response to window function
FIR_highpass = h_id.*w;

end
