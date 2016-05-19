function plot_sne(Y, labels, no_dims)
%PLOT_SNE Summary of this function goes here
%   Detailed explanation goes here
if nargin==2
    no_dims = 2;
end

if ~isempty(labels)
    if no_dims == 1
        scatter(Y, Y, 9, labels, 'filled');
    elseif no_dims == 2
        scatter(Y(:,1), Y(:,2),9, labels, 'filled');
    else
        scatter3(Y(:,1), Y(:,2), Y(:,3), 40, labels, 'filled');
    end
    axis equal tight
    drawnow
end

end


