
%%%%%%MATLAB STUB%%%%%%%%%
% This program is used to convert one given image into cartoon image by adding two filter's properties. (Bilateral and Canny)
% Author : Ashwini Singh-PA 2 Computer Vision
% Input  : castle.jpg(RGB image)
% Output : BilateralOutput.jpg, CannyOutput.jpg, CartoonOutput.jpg

function CartoonImage
img= imread('Castle.jpg');
img = im2double(img);
bilat = bilateralFilter(img, 9, 5, 10);
%note: the matlab function takes kernel size as (floor(kernelWidth/2))
%so this matches the kernel of size 9 passed into your function.
matlabbilat = bfilter2(img, 4, 5, 10);
str = strcat('Bilateral Filter RMSE: ',int2str(RMSerror(bilat, matlabbilat)) );
str
edges = Canny(rgb2gray(img), 100, 200);
matlabedges = edge(rgb2gray(img),'canny');
%left off here
str = strcat('Canny Edge RMSE: ',int2str(RMSerror(edges, matlabedges)) );
str
cartoon = cartoonImage(bilat, edges);
%The lines below can be used for looking at your output images, and saving
%them once your code is implemented. Be sure to include the images in your
%report.

% imshow(bilat);
% imwrite( bilat,'BilateralOutput.jpg');
% imshow(edges);
% imwrite(edges,'CannyOutput.jpg')
% imshow( cartoon);
% imwrite( cartoon,'CartoonOutput.jpg');

end

function I = bilateralFilter(img, kSize, sigmaColor, sigmaSpace)
%     Implementation of Bilateral Filtering for smoothing the image.
%     It is a non-linear, edge preserving filter for digital images based on a Gaussian distribution
%     (Param descriptions from the OpenCV Documentation)
%     :param img: Double format 3 channel color image.
%     :param kSize: Diameter of each pixel neighborhood that is used during filtering.
%         If it is non-positive, it is computed from sigmaSpace.
%     :param sigmaColor: Filter sigma in the color space. A larger value of the parameter
%         means that farther colors within the pixel neighborhood (see sigmaSpace) will
%         be mixed together, resulting in larger areas of semi-equal color.
%     :param sigmaSpace: Filter sigma in the coordinate space. A larger value of the
%         parameter means that farther pixels will influence each other as long as their
%         colors are close enough (see sigmaColor ). When d>0, it specifies the neighborhood
%         size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace.
%     :param borderType: always Reflect_101
%     :return: Filtered image of same size and type as img
%     """
%TODO: Write a bilateral filter

tic;
disp('bilateral')
lab=rgb2lab(img);                                                          % Convert into LAB color space
half_size=floor(kSize/2);                                                  % for padding
up=zeros(half_size,size(lab,2)+2*half_size);                               % up and down padding
left=zeros(size(lab,1),half_size);                                         % left and right padding
a=[up;[left lab(:,:,2) left];up];
b=[up;[left lab(:,:,3) left];up];
la=[up;[left lab(:,:,1) left];up];
[row,column]=size(a);
% CONVERT INTO 1-DIMENSION
c=reshape(a,[],1);
d=reshape(b,[],1);
lb=reshape(la,[],1);
result=lb;
% for each pixel of the original image, calculate weight for given window
% and then normalize it.
% instead of taking euclidian distance, we are taking difference between
% two components of LAB ('a' and 'b')
half_size=floor(kSize/2)*floor(kSize/2)/2;
for i=size(a,1)*floor(kSize/2)+floor(kSize/2)+1:size(result,1)-(size(a,1)*floor(kSize/2)+floor(kSize/2))
    a_window=(c(i-half_size:i+half_size)-c(i)).^2;
    b_window=(d(i-half_size:i+half_size)-d(i)).^2;
    l_window=lb(i-half_size:i+half_size);
    lab_window=(l_window-lb(i)).^2;
    first_comp=(a_window+b_window)/(2*sigmaColor*sigmaColor);
    second_comp=lab_window/(2*sigmaSpace*sigmaSpace);
    window=exp(-(first_comp)).*exp(-(second_comp));
    I=sum(sum(window.*l_window));
    result(i)=I/sum(sum(window));
end

norm=lab;
% CONVERT BACK TO 2D
half_size=floor(kSize/2);
result=reshape(result,row,column);
norm(:,:,1)=result(half_size+1:end-half_size,half_size+1:end-half_size);
I=lab2rgb(norm);                                                            % Convert back to rgb
%imshow(I)
toc;
end
function I = Canny(img, thresh1, thresh2)
%     The Canny edge detector is an edge detection operator that uses a multi-stage algorithm to detect a wide range of edges in images.
%     :param img: 8-bit input image.
%     :param thresh1: hysteresis threshold 1.
%     :param thresh2: hysteresis threshold 2.
%     :return: a single channel 8-bit with the same size as img

tic;
disp('Canny')
h = fspecial('gaussian', [5 5], 1);                                        % Take the guassian filter
I=img;
I=double(imfilter(I,h,'replicate'));                                       % Apply guassian filter to the image
Gx=[-1 0 1; -2 0 2; -1 0 1];                                               % Sobel gradient for pixels along Y-axis
Gy=[-1 -2 -1; 0 0 0; 1 2 1];                                               % Sobel gradient for pixels along X-axis
% Padding of Image
up=zeros(1,size(I,2)+2);
left=zeros(size(I,1),1);
I=[up; [left I left]; up];
resultx=zeros(size(I,1),size(I,2));
resulty=zeros(size(I,1),size(I,2));
% Apply Gradients Gx and Gy to the image and get the resultant image
for i=2:size(resultx,1)-1
    for j=2:size(resultx,2)-1
        resultx(i,j)=I(i-1,j-1).*Gx(1,1) + I(i-1,j).*Gx(1,2) + I(i-1,j+1).*Gx(1,3)+ I(i,j-1).*Gx(2,1)+ I(i,j).*Gx(2,2)+ I(i,j+1).*Gx(2,3)...
            + I(i+1,j-1).*Gx(3,1)+ I(i+1,j).*Gx(3,2)+ I(i+1,j+1).*Gx(3,3);
        resulty(i,j)=I(i-1,j-1).*Gy(1,1) + I(i-1,j).*Gy(1,2) + I(i-1,j+1).*Gy(1,3)+ I(i,j-1).*Gy(2,1)+ I(i,j).*Gy(2,2)+ I(i,j+1).*Gy(2,3)...
            + I(i+1,j-1).*Gy(3,1)+ I(i+1,j).*Gy(3,2)+ I(i+1,j+1).*Gy(3,3);
    end
end
% Gradient's direction calculation:
% Divide the matrix into 4 parts (0,45,90,135)
% where,
% 0: along X-direction                      ( - )
% 45: towards Bottom-Right direction        ( \ )
% 90: along Y-direction                     ( | )
% 135: towards Top-Right direction          ( / )

theta=atan2(resultx,resulty)*(180.0/pi);
theta((find(theta>=0 & theta<45 | theta>=-180 & theta<-135)))=0;
theta((find(theta>=45 & theta<90  | theta>=-135 & theta<-90)))=45;
theta((find(theta>=90 & theta<135 | theta>=-90 & theta<-45)))=90;
theta((find(theta>=135 | theta>=-45 & theta<0)))=135;
% Gradient's magnitude calculation
Gh=sqrt(resultx.^2+resulty.^2);
% Check along Corresponding associated directions, if it has any pixel in neighbours (window) which is
% greater than current pixel then replace current pixel with max pixel.

for i=2:size(theta,1)-1
    for j=2:size(theta,2)-1
        if theta(i,j)==0
            if (Gh(i,j)< max(Gh(i-1,j),Gh(i+1,j)))
                Gh(i,j)=0;
            end
            
        elseif theta(i,j)==90
            if (Gh(i,j)< max(Gh(i,j-1),Gh(i,j+1)))
                Gh(i,j)=0;
            end
            
        elseif theta(i,j)==45
            if (Gh(i,j)< max(Gh(i-1,j-1),Gh(i+1,j+1)))
                Gh(i,j)=0;
            end
            
        elseif theta(i,j)==135
            if (Gh(i,j)< max(Gh(i+1,j-1),Gh(i-1,j+1)))
                Gh(i,j)=0;
            end
        end
    end
end
% Convert to gray scale
Gh=Gh.*255;
% Apply threshold
Gh((find(Gh>thresh1/2 & thresh2/2<=90)))=0.5;
Gh((find(Gh<=thresh1/2)))=0;
Gh((find(Gh>thresh1/2)))=1;
% Check if Gradient has a value of 0.5,
% then check in window(3x3) for the max_value,
% if max_value is less that 0.5 then set current pixel to 0,else to 1.
for i =2:size(Gh,1)-1
    for j =2:size(Gh,2)-1
        if (Gh(i,j)==0.5)
            max_value=max([Gh(i-1,j-1),Gh(i-1,j),Gh(i-1,j+1),Gh(i,j-1),Gh(i,j),Gh(i,j+1),Gh(i+1,j-1),Gh(i+1,j),Gh(i+1,j+1)]);
            if (max_value<=0.5)
                Gh(i,j)=0;
            else
                Gh(i,j)=1;
            end
        end
    end
end
% Clip extra padded zeros
I=Gh(2:end-1,2:end-1);
toc;
end
function I = cartoonImage(filtered, edges)

%     :param filtered: a bilateral filtered image
%     :param edges: a canny edge image
%     :return: a cartoon image
%
%TODO: Create a cartoon image
tic;
disp('Cartoon');
for i=1:size(edges,1)
    for j=1:size(edges,2)
        if(edges(i,j)==1)      
            filtered(i,j,:)=0;
        end
    end
end
I = filtered;
toc;
end
function RMSE = RMSerror(img1, img2)

%     A testing function to see how close your images match expectations
%     Try to make sure your error is under 1. Some floating point error will occur.
%     :param img1: Image 1
%     :param img2: Image 2
%     :return: The error between the two images

diff = img1 - img2;
squaredErr = diff .^2;
meanSE = mean(squaredErr(:));
RMSE = sqrt(meanSE);
end

