function [ X ] = rotation_3d(X, axis, theta )
%   ROTATION_3D Rotates data around a random axis. =	I+omega^~sintheta+omega^~^2(1-costheta)
%   axis = []
%   X  (N x D)  Matrix wit h data to be rotated
%   cols (3x1)  Which columns to be affected
%   http://mathworld.wolfram.com/RodriguesRotationFormula.html

if nargin < 3
   theta = pi/4; 
end
[N, D] = size(X);
I = eye(D,D);
w = [0 axis(3) axis(2);axis(3) 0 axis(1);axis(2) axis(1) 0];
embedd = eye(3,3) + w*sin(theta) + w*w*(1-cos(theta));
rot = blkdiag(1,embedd,eye(D-4,D-4));

X = X*rot;

end

