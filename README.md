Charlotte Meola
ECE 5470, Digital Image Processing
Homework #3


The MATLAB code from file 'Meola_ECE_5470_Homework_3' accomplishes the following, in order:
-------------------------------------------------------------------------

1: Loads the input image and creates an image intensity matrix ['f'].

2: Applies 2D zero padding to 'f'. Saves new padded image as 'f_padded'.

3: Normalizes the 'f_padded' intensity values from 0 to 1, saves as 'f_padded_norm'.

4: Creates a centered image ['f_padded_norm_centered'] by multiplying each  
      element of 'f_padded_norm' by (-1)^(x+y).

5: Gets DFT matrix, ['F'], of 'f_padded_norm'.

6: Generates 'N', the Gaussian of the 'F' matrix.  Displays plot of 
      Gaussian LPF and created LPF function, "Lo."

7: Generates 'L', the Laplacian of the Gaussian matrix ['N'].

8: Multiplies F*N*L element-by-element to get resulting image ['M'].

9: Gets the inverse Fourier ['inv_M'] of the mutliplied image ['M'].

10: Uncenters the 'inv_M' image by again mutliplying each element by 
      (-1)^(x+y) to create the 'uncentered' image.

11: Unpads 'uncentered' image back to the original size of 'f'.

12: Save final subplots of all results from parts #1-11. 

----------------------------------------------------------------
