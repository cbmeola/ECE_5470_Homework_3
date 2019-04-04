% Charlotte Meola
% ECE 5470, Digital Image Processing
% Homework #3

% The following MATLAB code accomplishes the following, in order:
%-------------------------------------------------------------------------
% 1: Loads the input image and creates image intensity array [f].

% 2: Apply 2D zero padding to [orig_img]. Saves new padded image [f_padded].

% 3: Normalizes the [padded_img] intensity values from 0 to 1 [f_padded_norm].

% 4: Creates a centered image [f_padded_norm_centered] by multiplying each  
%       element of [f_padded_norm] by (-1)^(x+y).

% 5: Gets DFT matrix, [F], of [norm_padded_img].

% 6: Generates N, the Gaussian of the F matrix.  Displays plot of 
%       Gaussian LPF and created LPF function, "Lo."

% 7: Generates [L], the Laplacian of the Gaussian matrix [N].

% 8: Multiplies F*N*L element-by-element to get resulting image [multiplied_img].

% 9: Gets the inverse Fourier [inv_M] of the mutliplied image, [M].

% 10: Uncenter the [inv_M] image by again mutliplying each element by 
%       (-1)^(x+y) to create the [uncentered] image.

% 11: Unpad [uncentered] image back to the original size of [f].

% 12: Save final subplots of all results from parts 1-11. 
%----------------------------------------------------------------

close all; 
clear all; 
clc;

% 1: Loads the image, saves image intensity array, and displays image:
f = imread('input_img.png')

% Display result:
figure(1)
subplot(3, 4, 1)
imshow(f);
title('Original Image', 'FontSize', 8)
[M_orig, N_orig] = size(f);



% 2: Apply 2D zero padding to [orig_img]. Saves new padded image [f_padded].
f_padded = padarray(f,[M_orig, N_orig], 0, 'post')
[M_pad, N_pad] = size(f_padded);

% Display result:
subplot(3, 4, 2)
imshow(f_padded );
title('Original Padded Image', 'FontSize', 8)



% 3: Normalize the image intensity values from [0, 1].
% The returned matrix (padded_img_norm) contains values in the range 
% 0.0 (black) to 1.0 ( white) via the mat2gray function.
f_padded_norm = mat2gray(f_padded);



% 4: Creates a centered image [centered_img] by multiplying each element 
% by (-1)^(x+y).

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
title('Centered Normalized Padded Image', 'FontSize', 8)



% 5: Gets F(u,v), the DFT matrix [F], of centered_img:
F = fft2((double(f_padded_norm_centered)));

% Display result:
subplot(3, 4, 4)
imshow(log(abs(F)), []);
title({'Fourier of the Image','F(u, v)'}, 'FontSize', 8)



% 6: Generates N, the Gaussian of the F matrix.  Displays plot of 
% Gaussian LPF and created LPF function, "Lo."

% Filter size parameter:
R = 10; 
X = 1:N_pad;
Y = 1:M_pad;
[X Y] = meshgrid(X,Y);
Cx = 0.5*N_pad;
Cy = 0.5*M_pad;
Lo = exp(-((X-Cx).^2+(Y-Cy).^2)./(2*R).^2);

% Display result of 3D LPF plot:
subplot(3, 4, 5);
   mesh(X,Y,Lo);
   axis([ 0 N_pad 0 M_pad 0 1])
   h=gca; 
   get(h,'FontSize') 
   set(h,'FontSize',8)
   title('Gaussian LPF','fontsize',8)
   
% Allocate space for image:
N = zeros(size(F));
D = zeros(size(F));
   
% Loop over all rows and columns:
for ii=1:M_pad;
    for jj=1:N_pad;
        % Get D(u,v) value, distance:
        D(ii, jj) = sqrt((ii-M_pad/2)^2+(jj-N_pad/2)^2);
        if D(ii,jj) <= 100
            new_pixel =1;
        else
            new_pixel =0;
        end
        % Save new pixel value image:
        N(ii,jj)=new_pixel;
     end
end    
 
% Display result:
subplot(3, 4, 6)
imshow((N), []);
title({'Gaussian of the Image','N(u, v)'}, 'FontSize', 8)

     

% 7: Generates L, the Laplacian of the Gaussian matrix [N]: ----------------------------------
% Uses equation: L(u, v) = -4*pi^2*(u^2+v^2)

% Allocate space for image:
L = zeros(size(N));

% Loop over all rows and columns:
for ii=1:M_pad;
    for jj=1:N_pad;
        DL = sqrt((ii)^2+(jj)^2);
        new_pixel =-4*pi^2*(DL^2);
        % Save new pixel value in centered image:
        L(ii,jj)=new_pixel;
     end
end

% Display result:
subplot(3, 4, 7)
imshow(abs(L), []);
title({'Laplacian of Gaussian Image','L(u, v)'}, 'FontSize', 8)



% 8: Multiplies F(u,v)*N(u,v)*L(u,v) to get resulting image [multiplied_img].

% Allocate space for image:
M = zeros(size(N));

% Loop over all rows and columns:
for ii=1:M_pad;
    for jj=1:N_pad;
        % Multiplies each element:
        new_pixel =F(ii, jj)*L(ii, jj)*N(ii, jj);
        M(ii,jj)=new_pixel;
     end
end

% Display result:
subplot(3, 4, 8)
imshow(abs(M), []);
title({'Multiplied Image','F(u,v)*N(u, v)*L(u, v)' }, 'FontSize', 8)



% 9: Gets the inverse Fourier [inv_M] of the mutliplied image, [M].
% F^-1{ F(u,v)*N(u,v)*L(u,v) }

inv_M = ifft2(double(M));

% Display result:
subplot(3, 4, 9)
imshow(abs(inv_M), []);
title('Inv Fourier Multiplied Image', 'FontSize', 8)




% 10: Uncenter the [inv_M] image by again mutliplying each element by 
%       (-1)^(x+y) to create the [uncentered] image.

% Allocate space for centered image:
 uncentered = zeros(size(M));
 
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
uncentered = mat2gray(abs(uncentered));
imshow(uncentered, []);
title('Uncentered Inverse Fourier Image', 'FontSize', 8)



% 11: Unpad [uncentered] image back to the original size of [f].

% Only keep original size from padded image:
unpadded = uncentered(1:M_orig,1:N_orig);
%unpadded = mat2gray(unpadded)

% Display result:
subplot(3, 4, 11)
unpadded = abs(unpadded);
imshow(unpadded)
title('Final Unpadded Image', 'FontSize', 8)



% 12: Save final subplots of all results from parts 1-11. 
saveas(gcf,'Results_Figure.png')

% Save output image:
figure(2);
imshow((unpadded));
saveas(gcf, 'output_img.png');

