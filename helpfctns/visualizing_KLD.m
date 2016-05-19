%% Shows effect of plotting Kullback Leibler divergence
n = 100; d = 100;
kb = @(p,q) p .*log(p./q); 
Z = zeros(n,d);
for i = 1:n
    for j = 1:d
        Z(i,j) = kb(i,j);
    end
end

contour(Z,30)
hold on; c = colorbar('Ticks',[0,100,200,300,400],...
         'TickLabels',{'0','1','2','3','4'});
% Add labels and stuff
%surf(Z)
xlabel('p [x100]')
ylabel('q [x100]')
set(gca,'Fontsize',16)
set(gca,'FontName','Helvetica')
c.Label.String = 'Contribution to KLD'