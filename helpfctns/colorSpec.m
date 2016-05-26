function k=colorSpec(i)
%generates 100 different  color specification
%the ymcrgbk chart
chart=[1 1 0; 1 0 1;0 1 1;1 0 0; 0 1 0;0 0 1;0 0 0];
k=chart(mod(i,7)+1,:)*(1-0.1*floor(i/7));
end
