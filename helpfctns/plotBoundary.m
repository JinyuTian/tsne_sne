function plotBoundary(data,labels,model)

%Extract the SV
SVs = full(model.SVs);
rangeX= linspace(min(data(:,1))-1, max(data(:,1))+1, 200);
rangeY= linspace(min(data(:,2))-1,max(data(:,2))+1,200);
[X,Y]=meshgrid(rangeX,rangeY);
%rectify X and Y
X=X(:);
Y=Y(:);
temp=zeros(length(X),1);


%Predcit the value of the grid
[predicted_label, Accuracy, decision]=svmpredict(temp,[X Y],model);
%Generate color
color=predicted_label;
color=color*255/max(color);
color=reshape(color,length(rangeX),length(rangeY));

%edge processing
H=fspecial('disk',ceil(model.Parameters(4)*100));
color=imfilter(color,H,'replicate');

%plot the decision
contourf(rangeX,rangeY,color,50, 'LineStyle', 'none')
hold on
%plot the data points

colorspec='ymcrgbk';
for i=min(labels):max(labels)
   ind=find(labels==i);
   plot(data(ind,1), data(ind,2), 'ko', 'MarkerFaceColor', colorspec(i-min(labels)+1)); hold on
end
%plot the Support Vector
scatter(SVs(:,1),SVs(:,2),90,'o','MarkerEdgeColor', [1 1 1], 'MarkerEdgeColor', 'w', 'LineWidth', 1.5);




end