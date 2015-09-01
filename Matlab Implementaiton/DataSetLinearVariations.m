function [data] = DataSetLinear(variation)
%DATASETLINEARVARIATIONS returns a linear dataset that is not a perfect fit

x0=ones(500,1);
xs=rand(500,4);
y= .2.*x0 + .4.*xs(:,1) + .6.*xs(:,2) + .8.*xs(:,3) + 1.*xs(:,4);
data = [x0 xs y];

end



