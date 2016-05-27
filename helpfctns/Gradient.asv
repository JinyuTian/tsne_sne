%% Shows gradient of two SNE
dl= [1:100]/10; %low dimension distance
d = [1:100]/10; %high dimension distance
gradient = @(p,q,dl) (p-q).*dl.*(1+dl.^2).^-1 ;
Z = zeros(length(d),length(dl));
for i = 1:length(d)
    for j = 1:(length(dl))
        p=tpdf(dl(j),3);
        q=normpdf(d(i),0,2);
        Z(i,j) = gradient(p,q,dl(j));
    end
end
figure()
[dl_grid,d_grid]=meshgrid(dl,d);

contourf(dl_grid,d_grid,Z,30,'LineStyle', 'none')
colorbar;
xlabel('High dimensional distance >')
ylabel('Low dimensional distane >')
title('Gradient of t-SNE')
set(gca, 'XTick', []);
set(gca, 'YTick', []);


for i = 1:length(d)
    for j = 1:(length(dl))
        q=normpdf(d(i),0,8);
        pg=normpdf(dl(j),0,2);
        Zg(i,j)=gradient(pg,q,dl(j));
    end
end
figure()
contourf(dl_grid,d_grid,Zg,30,'LineStyle', 'none')
colorbar;
xlabel('High dimensional distance >')
ylabel('Low dimensional distane >')
title('Gradient of SNE')
set(gca, 'XTick', []);
set(gca, 'YTick', []);