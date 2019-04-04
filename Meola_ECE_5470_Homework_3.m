% Charlotte Meola
% ECE 5470, Digital Image Processing
% Homework #3

close all; 
clear all; 
clc;


%1: Loads the input image and creates an image intensity matrix ['f'].

f = imread('input_img.png') % 'f' = f(x, y)

% Display result:
figure(1)
subplot(3, 4, 1)
imshow(f);
title('Original Image', 'FontSize', 8)
[M_orig, N_orig] = size(f);



% 2: Applies 2D zero padding to 'f'. Saves new padded image as 'f_padded'.

f_padded = padarray(f,[M_orig, N_orig], 0, 'post') 
[M_pad, N_pad] = size(f_padded);

% Display result:
subplot(3, 4, 2)
imshow(f_padded );
title('Original Padded Image', 'FontSize', 8)




% 3: Normalizes the 'f_padded' intensity values from 0 to 1, saves as 'f_padded_norm'.
% The returned matrix 'f_padded_norm' contains values in the range 
% 0.0 (black) to 1.0 ( white) via the mat2gray function.

f_padded_norm = mat2gray(f_padded); % = f(x, y)/255




% 4: Creates a centered image ['f_padded_norm_centered'] by multiplying each
%       element of 'f_padded_norm' by (-1)^(x+y).

% Allocate space for image:
f_padded_norm_centered = zeros(size(f_padded_norm));

% Loop over all rows and columns:
for ii=1:M_pad;
    for jj=1:N_pad;
        % Get pixel value:
        pixel=f_padded_norm(ii,jj);
        % Center the image by (-1)^(x+y):
        new_pixel =pixel*((-1)^(ii+jj));
        % Save new pixel value in centered image:
        f_padded_norm_centered(ii,jj)=new_pixel;
     end
end

% Display result:
subplot(3, 4, 3)
imshow((f_padded_norm_centered), []);
title('Centered, Normalized, & Padded Image', 'FontSize', 8)




% 5: Gets DFT matrix, ['F'], of 'f_padded_norm'.

F = fft2((double(f_padded_norm_centered))); % 'F' = F(u,v)

% Display result:
subplot(3, 4, 4)
imshow(log(abs(F)), []);
title({'Fourier of the Image','F(u, v)'}, 'FontSize', 8)




% 6: Generates 'N', the Gaussian of the 'F' matrix. Displays plot of 
%       Gaussian LPF and created LPF function, "Lo." 

% Filter size parameter:
Size = 10; 
X = 1:N_pad;
Y = 1:M_pad;
[X Y] = meshgrid(X,Y);
Cx = 0.5*N_pad;
Cy = 0.5*M_pad;
LPF = exp(-((X-Cx).^2+(Y-Cy).^2)./(2*Size).^2);

% Display result of 3D Gaussian LPF plot:
subplot(3, 4, 5);
   mesh(X,Y,LPF);
   axis([ 0 N_pad 0 M_pad 0 1])
   h=gca; 
   get(h,'FontSize') 
   set(h,'FontSize',8)
   title('Gaussian LPF','fontsize',8)
   
% Allocate space for image:
N = zeros(size(F)); % Gaussian image, N.
D = zeros(size(F)); % Matrix of distance values, D.
   
% Loop over all rows and columns:
for ii=1:M_pad;
    for jj=1:N_pad;
        % Get D(u,v) value, distance:
        D(ii, jj) = sqrt((ii-M_pad/2)^2+(jj-N_pad/2)^2);
        % using D0 value of 100:
        if D(ii,jj) <= 100 
            new_pixel =1; % Passes D values below 100.
        else
            new_pixel =0; % Does NOT pass D values above 100.
        end
        % Save new pixel in Gaussian image matrix, N:
        N(ii,jj)=new_pixel;
     end
end    
 
% Display Gaussian image result:
subplot(3, 4, 6)
imshow((N), []); % 'N' = N(u,v)
title({'Gaussian of the Image, N(u, v)','D_0 = 100'}, 'FontSize', 8)

     


% 7: Generates 'L', the Laplacian of the Gaussian matrix ['N'].

% Allocate space for image:
L = zeros(size(N)); % Laplacian of Gaussian, L.

% Loop over all rows and columns:
for ii=1:M_pad;
    for jj=1:N_pad;
        % Get DL(u,v) value, distance:
        DL = sqrt((ii)^2+(jj)^2);
        % Uses equation L(u, v) = -4*pi^2*(u^2+v^2).
        new_pixel = -4*pi^2*(DL^2);
        % Save new pixel value in centered image:
        L(ii,jj)= new_pixel;
     end
end

% Display result:
subplot(3, 4, 7)
imshow(abs(L), []); % 'L' = L(u,v)
title({'Laplacian of Gaussian Image','L(u, v)'}, 'FontSize', 8)




% 8: Multiplies F(u,v)*N(u, v)*L(u, v) element-by-element to get resulting image ['M'].

% Allocate space for image:
M = zeros(size(N)); % Multiplied image, M.

% Loop over all rows and columns:
for ii=1:M_pad;
    for jj=1:N_pad;
        % Multiplies each element:
        new_pixel =F(ii, jj)*L(ii, jj)*N(ii, jj);
        % Saves in new matrix, M.
        M(ii,jj)=new_pixel;
     end
end

% Display result:
subplot(3, 4, 8)
imshow(abs(M), []);
title({'Multiplied Image','M(u, v) = F(u,v)*N(u, v)*L(u, v)' }, 'FontSize', 8)




% 9: Gets the inverse Fourier ['inv_M'] of the mutliplied image ['M'].

inv_M = ifft2(double(M)); % = F^-1{ F(u,v)*N(u,v)*L(u,v)}

% Display result:
subplot(3, 4, 9)
imshow(abs(inv_M), []);
title({'Inv Fourier Multiplied Image','F^-1{ F(u,v)*N(u,v)*L(u,v)}'}, 'FontSize', 8)




% 10: Uncenters the 'inv_M' image by again mutliplying each element by 
%       (-1)^(x+y) to create the 'uncentered' image.

% Allocate space for centered image:
 uncentered = zeros(size(M)); % Uncentered multiplied image, uncentered.
 
% Loop over all rows and columns:
for ii=1:M_pad
    for jj=1:N_pad
        % Get pixel value:
        pixel=inv_M(ii,jj);
        % Uncenter the image by (-1)^(x+y):
        new_pixel =pixel*((-1)^(ii+jj));
        % Save new pixel value in uncentered image:
        uncentered(ii,jj)=new_pixel;
    end
end

% Display result:
subplot(3, 4, 10)
% Normalize:
uncentered = mat2gray(abs(uncentered));
imshow(uncentered, []);
title('Uncentered Inverse Fourier Image', 'FontSize', 8)




% 11: Unpads 'uncentered' image back to the original size of 'f'.

% Only keep original size from padded image:
unpadded = uncentered(1:M_orig,1:N_orig);

% Display result:
subplot(3, 4, 11)
unpadded = abs(unpadded);
imshow(unpadded);
title('Final Unpadded Image', 'FontSize', 8);




% 12: Saves the final subplot figure of all displayed results from parts #1-11.
saveas(gcf,'Results_Figure.png')




% 13. Saves the final resulting, unpadded image as 'output_img' in file folder.
figure(2);
imshow((unpadded));
saveas(gcf, 'output_img.png');

