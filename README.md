# ECE_5470_Homework_3

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
