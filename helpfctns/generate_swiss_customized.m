function [ X,labels ] = generate_swiss_broken_customized( n,noise, dimensions, theta_0, theta_1 )
%GENERATE_SWISS_CUSTOMIZED 
%
%   input -----------------------------------------------------------------
%
%       o n : (1 x 1), the number of samples to be generated.
%       o noise : (1 x 1), the level of noise
%       o theta_0 : (1 x 1), start angle of the broken part
%               dom(theta_0) = [0, pi]
%       o theta_1 : (1 x 1), end angle of the broken part
%               dom(theta_1) = [0, pi]
%   output-----------------------------------------------------------------
%
%       o X           : (n x dimensions (=3) ), data
%       o labels      : (n x 1), class of the data (0 or 1)
%

    t = (3 * pi / 2).* [(1 + 2 * rand(ceil(n / 2), 1) * .4); 
       (1 + 2 * (rand(floor(n / 2), 1) * .4 + .4))];  
    
    %t = [ () * (1+2*rand(ceil(n/2),1) * .4)); 
    %        ];
    
    height = 30 * rand(n, 1);
    X = [t .* cos(t) height t .* sin(t)] + noise * randn(n, 3);
    
    labels(1:size((3 * pi / 2) * (1 + 2 * rand(ceil(n / 2), 1) * .4),1),1) = 0;
    
    labels(1+size((3 * pi / 2) * (1 + 2 * rand(ceil(n / 2), 1) * .4),1):size((3 * pi / 2) *...
            (1 + 2 * rand(ceil(n / 2), 1) * .4),1) + ...
            size((3 * pi / 2) * (1 + 2 * (rand(floor(n / 2), 1) * .4 + .6)),1),1) = 1;
        
    t = [t height];

end
