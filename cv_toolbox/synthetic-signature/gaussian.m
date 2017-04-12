x=0:1:8; y=x; %x and y values between 0 and +1 spaced by 0.005
r = 0.9;
[X,Y]=meshgrid(x,y); %generate a 2D grid of xy values
Z = 1.5 * exp (-((X).^2+(Y).^2) / (2*r^2)); % generate the Gaussian
Z

% Z = An * exp( - (x^2 / (2PHIx)
%function on the grid
v=0:.1:1.; % contours will be from 0 to 1 in steps of 0.1
[C,h]=contour(X,Y,Z,v); % generate the contour plot, including values
%to label contours
axis square %make the plot square
clabel(C,h) %label the contours
xlabel('x (ph)'), ylabel('y (ph)');
set(gca,'ydir','reverse');
set(gca,'XAxisLocation','top'); %Xaxis labels on top
% colormap([0 0 0]); % use black only