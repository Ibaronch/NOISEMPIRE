##############################################
Name="NOISEMPIRE"
Version="1.0.2" # Major.Minor.Patch sequence. When a major, minor, or patch update is made, the corresponding number is increased.
Years="2023-2024"
Developer="Ivano Baronchelli"

##############################################
# Simulates a pure noise map using real ALMA images as a reference
# Call this script as follows:
#########################################
# $ python noisempire.py config_file.txt
# or, if you want to specify the image name in the command line:
# $ python noisempire.py config_file.txt --INPUT_IMAGE input_image.fits
#########################################
#
# Given an input ALMA image (or flattened cube), it creates a simulated
# image made of pure noise.
#
# Ivano Baronchelli 2023-2024
###################################################################
# Versioning 
###################################################################
# V1.0.1 --> V1.0.2 May 3 2024
#  - corrected bug (arising flattening cubes problem when extracting radial - elliptical patterns) 
#    
#
# V1.0.0 --> V1.0.1 May 2 2024
#  - removed duplicated import (re)
#  - removed misleading comments 

# Major version changes are related to incompatible API changes.
# Minor version changes are related to adding new functionality in a backward-compatible manner.
# Patch version changes are related to bug fixes which are also backward compatible.

print('\n\n              ###################################')
print('              #         '+ Name +' '+ Version+'        #')
print('              #   '+Developer +' '+Years+'   #')
print('              ###################################\n\n')



from pdb import set_trace as stop # in this way you can just write stop() to stop the program.
import os
import sys
from astropy.io import fits
import numpy as np
from astropy.wcs import WCS
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt # to show plot and images
import math
import scipy.signal # for image convolution or to create a 2d Gaussian
from scipy.fft import fft2, ifft2, fftshift
from scipy.ndimage import binary_dilation # expand a boolean mask of a few pixels
import cv2 #("conda install opencv")
from scipy.ndimage import zoom
from skimage import measure
import re # used to identify variable types
import argparse # used to pass arguments to main()

def get_type_and_convert(value):
    # Check if an input variable is an integer, float, boolean or a string
    # and returns it in the correct format
    #
    # Check if the value is an integer
    if re.match(r'^[-+]?[0-9]+$', value):
        return int(value)
    # Check if the value is a float
    elif re.match(r'^[-+]?[0-9]*\.[0-9]+$', value):
        return float(value)
    # Check if the value is a boolean True
    elif value.lower() == 'true':
        return True
    # Check if the value is a boolean False
    elif value.lower() == 'false':
        return False
    # Check if the value is None
    elif value.lower() == 'none':
        return None
    # Otherwise, it's likely a string   
    else:
        return value


def get_param_value(PARAMS, VALUES, param_name, convert=True):
    # inputs:
    # PARAMS --> list of parameters in a string format
    # VALUES --> values for each parameter in PARAMS
    # param_name --> parameter for which we want to get the value (string)
    # convert --> if True (default), the parameter is automatically recognized
    #             and converted to float, integer, boolean or string 
    idx1=np.where(PARAMS==param_name)
    idx1=idx1[0][0]
    if convert==False:
        PARAMETER=VALUES[idx1]
    if convert==True:
        PARAMETER=get_type_and_convert(VALUES[idx1])
    return PARAMETER

class Readcol(object):
    ##################################################################################
    # Read specified columns in an ascii file and returns them in the specified format
    # Example of use:
    # cat=Readcol(path0+'fin_F160.cat','f,f,x,x,x,a20',i,skipline=17,skipcol=7)
    # X=cat.col(0)# will be a float numpy array
    # Y=cat.col(1) # will be a float numpy array
    # Z=cat.col(2) # will be a string numpy array
    # K=cat.col(3) # will be an integer numpy array
    # # Note that the first 17 lines (skipline) and the first 7 columns (skipcol) 
    # # are not considered. For the columns skipped using "skipcol", the format should
    # # not be specified. For the other columns, the user can specify if the format 
    # # should be float (f), integer (i), or a string (a). The maximum lenght of the  
    # # strings to be read  should also be specified using the string_length keyword. 
    # # By default, string columns are considered 30 characters long.
    # # The indexes "n" of the output columns, used in col(n), start from 0 and do not 
    # # consider skipped columns (skipcol or 'x').
    ##################################################################################
    def __init__(self, filename, form,skipline=0, skipcol=0, sep='default', skip_sym='#', string_length=30 ):
        self.filename=filename # string containing path+filename
        self.form=form         # format of the elements in the columns. Example:
                               # 'f,f,f,x,a,i' --> first three columns (after the 
                               # skipped ones, specified using skipcol) will be
                               # returned as float, third column is jumped, fourth 
                               # column is returned as a character, fifth as an integer.   
        self.skipline=skipline # number of lines to skip (optional, default=0)
        self.skipcol=skipcol   # number of clumns to skip (optional, default=0)
        self.sep=sep           # Separator. Default are spaces. Other options 
                               # are ',' '\t' (tab). It accepts all the options
                               # allowed in string.split()
        self.skip_sym=skip_sym # skip all lines beginning with the character specified.
                               # Default is lines starting with "#". Void lines are always 
                               # skipped

        if os.path.isfile(filename)==0:
            print ('File '+ filename +' not found')
        if os.path.isfile(filename)==True:
            FILE=open(filename,'r')
            ALL_COL_STR=FILE.readlines()
            FILE.close()
            FORMAT=np.array(form.split(',')) # Must be converted otherwise it doesn't work
            ncol=len(FORMAT[FORMAT != 'x'])  # Number of output columns 
            out_format=['' for x in range(ncol)]   # Format of output columns
            out_format=np.array(out_format, str) # numpy array of strings
            nlines=len(ALL_COL_STR)-skipline # Number of output lines 
            space_string = " " * string_length
            all_col=[[space_string for x in range(ncol)] for x in range(nlines)]
            all_col=np.array(all_col, str) # numpy array of strings
            CN=skipcol # Input Column Number (also 'x' considered here)
            RCN=0      # Real (output) Column Number (no 'x')
            while CN < len(FORMAT)+skipcol:
                if FORMAT[CN-skipcol]!='x':
                    LN=skipline # Input line Number
                    RLN=0       # Real (output) line Number (no 'x')
                    while LN < len(ALL_COL_STR):
                        line=ALL_COL_STR[LN]
                        if line != '\n':
                            # Read line if it doesn't start with the skip symbol
                            if line.split()[0][0] not in self.skip_sym:
                                if self.sep=='default':
                                    linesplit = line.split()
                                if self.sep!='default':
                                    linesplit = line.split(self.sep)
                                #--------------------------------
                                if CN<=np.size(linesplit)-1:
                                    all_col[RLN,RCN]=linesplit[CN]
                                if CN>np.size(linesplit)-1:
                                    all_col[RLN,RCN]=' '
                                #--------------------------------
                                RLN=RLN+1
                        LN=LN+1


                    out_format[RCN]=FORMAT[CN-skipcol]
                    RCN=RCN+1
                CN=CN+1
        self.out_format=out_format
        self.all_col=all_col
        self.nline=RLN
    def col(self,coln):
        ###################################################
        # "coln" corresponds to the column number on the ascii
        # file (start from 0) minus "skipcol", minus the number
        # of "x" indicated in the input format string "form"  
        ###################################################
        if self.out_format[coln].lower()=='a':
            OUTCOL=np.array(self.all_col[:self.nline,coln], str)
        if self.out_format[coln].lower()=='i':
            OUTCOL=np.array(self.all_col[:self.nline,coln], int)
        if self.out_format[coln].lower()=='f':
            OUTCOL=np.array(self.all_col[:self.nline,coln], float)
        return OUTCOL


def get_granularity(image, N_sigma=1.0):
    """
    Compute the granularity of a given image as the
    inverse of the number of "blobs" found above a
    given threshold (1 positive sigma, by default)
    """
    input_image=np.copy(image)
    # remove outliers (2% of higher and lower values)
    outliers_low=np.where(input_image < np.percentile(input_image,[2]))
    input_image[outliers_low]=np.percentile(input_image,[2])
    outliers_up=np.where(input_image > np.percentile(input_image,[98]))
    input_image[outliers_up]=np.percentile(input_image,[98])
    
    inverse_image=np.max(input_image)-input_image

    # Normalize to the range [0, 1]
    normalized_input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
    normalized_inverse_image = (inverse_image - inverse_image.min()) / (inverse_image.max() - inverse_image.min())
    # Scale to the range [0, 255]
    scaled_input_image = (normalized_input_image * 255).astype(np.uint8)
    scaled_inverse_image = (normalized_inverse_image * 255).astype(np.uint8)
    # Apply thresholding to create a binary image
    seg_thresh=127+N_sigma*np.round(np.std(scaled_input_image))
    maxval=np.max(scaled_input_image)
    _, binary_input_image = cv2.threshold(scaled_input_image, seg_thresh, maxval, cv2.THRESH_BINARY)
    _, binary_inverse_image = cv2.threshold(scaled_inverse_image, seg_thresh, maxval, cv2.THRESH_BINARY)
    # Perform connected component analysis to label connected regions
    labeled_input_img = measure.label(binary_input_image, connectivity=2)  # Set connectivity as needed
    labeled_inverse_img = measure.label(binary_inverse_image, connectivity=2)  # Set connectivity as needed
    # Count the number of segments
    N_seg=(np.max(labeled_input_img)+np.max(labeled_inverse_img))/2.
    # Min granularity: 0, i.e., one positive pixels
    # Max granularity: 1, i.e., one positive pixel every 4 pixels (corresponding to
    #                      white pixels separated by one single black pixel) 
    granularity =N_seg/(np.size(input_image)/4.)
    return granularity


def convolve_local_ACF(input_image, ref_image, box_size=32, SHOW_PLOT=False, norm_blocks='No', shrink_factor=1.0):

    """
    Convolve an input image ("input_image") with the 
    Auto-Correlation Function (ACF) of a reference image
    ("ref_image"). The ACF is computed on a local scale,
    by dividing the image in boxes whose size (in pixels)
    is set by the user.
    The function Returns the input image convolved as described
    - If norm_blocks is set to 'No', the boxes are not normalized 
      after the convolution process
    - If norm_blocks is set to "Global", the distribution of the 
      pixel values inside the boxes are normalized to the same 
      (percentile-based) stdv measured in the overall ref_image
    - If norm_blocks is set to "Local", the distribution of the 
      pixel values inside the boxes are normalized to the same 
      (percentile-based) stdv measured in the same box of the
      reference image
    
    Setting shrink_factor, the ACF can be shrinked 
    (from scipy.ndimage import zoom) to reach
    the desired granularity of the final image
 
    """

    if norm_blocks == "Global":
        Global_ref_stdv=(np.percentile(ref_image,[84]) - np.percentile(ref_image,[16])) /2.

    # Create temporary working images
    tmp_img=np.copy(input_image)
    tmp_img[:,:]=0.
    tmp_img_shy=np.copy(tmp_img)
    tmp_img_shx=np.copy(tmp_img)
    tmp_img_shxy=np.copy(tmp_img)
    map_weights_y=np.copy(tmp_img)
    map_weights_x=np.copy(tmp_img)
    x_strip=np.copy(tmp_img[0,:]) # o ne-dimensional weights (along x)
    y_strip=np.copy(tmp_img[:,0]) # one-dimensional weights (along y)

    # Get image dimensions
    img_height, img_width = input_image.shape[0:2]

    # Number of sectors along each axis
    N_SECT_axis = round(img_width/box_size)

    # Actual size of the blocks in pixels
    SECT_Npix_x = round(img_width / N_SECT_axis)
    SECT_Npix_y = round(img_height / N_SECT_axis)
    
    # shifts along x and y
    shift_x = round(0.5*img_width / N_SECT_axis)
    shift_y = round(0.5*img_height / N_SECT_axis)
    
    # Define triangular function (0-to-1) for the 
    # amplitude of the weights
    amp=1.0
    # period
    per_x=SECT_Npix_x # Triangular Period along x 
    per_y=SECT_Npix_y # Triangular Period along y 
    indices_x = np.arange(len(x_strip))
    indices_y = np.arange(len(y_strip))
    x_strip=1.0 - np.abs((indices_x % per_x) - per_x/2)
    x_strip=x_strip-np.min(x_strip)
    x_strip=amp*x_strip/np.max(x_strip)
    y_strip=1.0 - np.abs((indices_y % per_y) - per_y/2)
    y_strip=y_strip-np.min(y_strip)
    y_strip=amp*y_strip/np.max(y_strip)
    
    # Compute weighting maps
    map_weights_x[:,:]=x_strip
    # set x extremes to 1 (image borders must be kept into account)
    map_weights_x[:,0:per_x//2]=amp # to keep the borders into account
    map_weights_x[:,(img_width-1)-per_x//2 : img_width] = amp
    
    map_weights_y.T[:,:]=y_strip # WARNING: must be transposed (.T), to fill it with the 1D strip!
    # set y extremes to 1 (image borders must be kept into account)
    map_weights_y[0:per_y//2 , :]=amp # to keep the borders into account
    map_weights_y[(img_height-1)-per_y//2 : img_height , :] = amp

    pix_start_y = 0
    pix_stop_y = SECT_Npix_y
    borrow_y=0
    additive_pix_y=0
    additive_pix_y_tot=0

    for i in range(N_SECT_axis):

        pix_start_x = 0
        pix_stop_x = SECT_Npix_x
        borrow_x=0
        additive_pix_x=0
        additive_pix_x_tot=0
        for j in range(N_SECT_axis):

            # Compute local ACF
            LOCAL_ACF = scipy.signal.correlate2d(
                ref_image[pix_start_y:pix_stop_y, pix_start_x:pix_stop_x],
                ref_image[pix_start_y:pix_stop_y, pix_start_x:pix_stop_x],
                mode="same",
                boundary="wrap",
            )
            LOCAL_ACF = zoom(LOCAL_ACF, shrink_factor)

            # Compute local ACF + shift y
            if (pix_start_y+shift_y < img_height and pix_stop_y+shift_y <= img_height):
                LOCAL_ACF_shy = scipy.signal.correlate2d(
                    ref_image[pix_start_y+shift_y:pix_stop_y+shift_y, pix_start_x:pix_stop_x],
                    ref_image[pix_start_y+shift_y:pix_stop_y+shift_y, pix_start_x:pix_stop_x],
                    mode="same",
                    boundary="wrap",
                )
                LOCAL_ACF_shy = zoom(LOCAL_ACF_shy, shrink_factor)
#            LOCAL_ACF_shy = zoom(LOCAL_ACF_shy, shrink_factor)

            # Compute local ACF + shift x
            if (pix_start_x+shift_x < img_width and pix_stop_x+shift_x <= img_width):
                LOCAL_ACF_shx = scipy.signal.correlate2d(
                    ref_image[pix_start_y:pix_stop_y, pix_start_x+shift_x:pix_stop_x+shift_x],
                    ref_image[pix_start_y:pix_stop_y, pix_start_x+shift_x:pix_stop_x+shift_x],
                    mode="same",
                    boundary="wrap",
                )
                LOCAL_ACF_shx = zoom(LOCAL_ACF_shx, shrink_factor)

            # Compute local ACF + shift x &  + shift y
            if (pix_start_y+shift_y < img_height and pix_stop_y+shift_y <= img_height and pix_start_x+shift_x < img_width and pix_stop_x+shift_x <= img_width):
                LOCAL_ACF_shxy = scipy.signal.correlate2d(
                    ref_image[pix_start_y+shift_y:pix_stop_y+shift_y, pix_start_x+shift_x:pix_stop_x+shift_x],
                    ref_image[pix_start_y+shift_y:pix_stop_y+shift_y, pix_start_x+shift_x:pix_stop_x+shift_x],
                    mode="same",
                    boundary="wrap",
                )
                LOCAL_ACF_shxy = zoom(LOCAL_ACF_shxy, shrink_factor)


        
            if (SHOW_PLOT):
                # Create a plot to visualize the ACF DEBUG PURPOSE ONLY!
                plt.figure(figsize=(8, 8))
                plt.imshow(
                    LOCAL_ACF,
                    cmap="viridis",
                    origin="lower",
                    extent=[
                        -LOCAL_ACF.shape[1] // 2,
                        LOCAL_ACF.shape[1] // 2,
                        -LOCAL_ACF.shape[0] // 2,
                        LOCAL_ACF.shape[0] // 2,
                    ],
                )
                plt.colorbar(label="Local ACF")
                plt.title("Local Noise Auto-Correlation Function (ACF)")
                plt.xlabel("Offset (Pixels)")
                plt.ylabel("Offset (Pixels)")
                plt.show()


            # Convolve img with local ACF (mode=same to obtain the same image size)
            #----------------------------------------------------------------------
            #
            # SAME POSITION ############################################
            tmp_img[pix_start_y:pix_stop_y, pix_start_x:pix_stop_x] = scipy.signal.convolve2d(
                input_image[pix_start_y:pix_stop_y, pix_start_x:pix_stop_x],
                LOCAL_ACF,
                mode="same",
                boundary='wrap'
            )

            # TEST TEST - END
#            if box_size==256:
#                tmp_img[pix_start_y:pix_stop_y, pix_start_x:pix_stop_x] = scipy.signal.convolve2d(
#                    input_image[pix_start_y:pix_stop_y, pix_start_x:pix_stop_x],
#                    LOCAL_ACF[int(256/2 - 20) : int(256/2 + 20) , int(256/2 - 20) : int(256/2 + 20)],
#                    mode="same",boundary='wrap'
#                )

            #plt.imshow(LOCAL_ACF[int(256/2 - 25) : int(256/2 + 25) , int(256/2 - 25) : int(256/2 + 25)], cmap='gray')
            #plt.imshow(LOCAL_ACF, cmap='gray')
            #plt.gca().invert_yaxis()
            #plt.show()
            ## TEST TEST - END

            # -- Normalization
            if norm_blocks!='No':
                block_img=tmp_img[pix_start_y:pix_stop_y, pix_start_x:pix_stop_x]
                Local_img_stdv=(np.percentile(block_img,[84]) - np.percentile(block_img,[16])) /2.
                if norm_blocks=='Local':
                    block_ref=ref_image[pix_start_y:pix_stop_y, pix_start_x:pix_stop_x]
                    Local_ref_stdv=(np.percentile(block_ref,[84]) - np.percentile(block_ref,[16])) /2.
                    tmp_img[pix_start_y:pix_stop_y, pix_start_x:pix_stop_x]=(tmp_img[pix_start_y:pix_stop_y, pix_start_x:pix_stop_x]/Local_img_stdv)*Local_ref_stdv
                if norm_blocks=='Global':
                    tmp_img[pix_start_y:pix_stop_y, pix_start_x:pix_stop_x]=(tmp_img[pix_start_y:pix_stop_y, pix_start_x:pix_stop_x]/Local_img_stdv)*Global_ref_stdv


            # POSITION + shift y ############################################
            if (pix_start_y+shift_y < img_height and pix_stop_y+shift_y <= img_height):
                tmp_img_shy[pix_start_y+shift_y:pix_stop_y+shift_y, pix_start_x:pix_stop_x] = scipy.signal.convolve2d(
                    input_image[pix_start_y+shift_y:pix_stop_y+shift_y, pix_start_x:pix_stop_x],
                    LOCAL_ACF_shy,
                    mode="same",
                    boundary='wrap'
                )
                # -- Normalization
                if norm_blocks!='No':
                    block_img_shy=tmp_img_shy[pix_start_y+shift_y:pix_stop_y+shift_y, pix_start_x:pix_stop_x]
                    Local_img_stdv_shy=(np.percentile(block_img_shy,[84]) - np.percentile(block_img_shy,[16])) /2.
                    if norm_blocks=='Local':
                        block_ref_shy=ref_image[pix_start_y+shift_y:pix_stop_y+shift_y, pix_start_x:pix_stop_x]
                        Local_ref_stdv_shy=(np.percentile(block_ref_shy,[84]) - np.percentile(block_ref_shy,[16])) /2.
                        tmp_img_shy[pix_start_y+shift_y:pix_stop_y+shift_y, pix_start_x:pix_stop_x]=(tmp_img_shy[pix_start_y+shift_y:pix_stop_y+shift_y, pix_start_x:pix_stop_x]/Local_img_stdv_shy)*Local_ref_stdv_shy
                    if norm_blocks=='Global':
                        tmp_img_shy[pix_start_y+shift_y:pix_stop_y+shift_y, pix_start_x:pix_stop_x]=(tmp_img_shy[pix_start_y+shift_y:pix_stop_y+shift_y, pix_start_x:pix_stop_x]/Local_img_stdv_shy)*Global_ref_stdv

            # POSITION + shift x ############################################
            if (pix_start_x+shift_x < img_width and pix_stop_x+shift_x <= img_width):
                tmp_img_shx[pix_start_y:pix_stop_y, pix_start_x+shift_x:pix_stop_x+shift_x] = scipy.signal.convolve2d(
                    input_image[pix_start_y:pix_stop_y, pix_start_x+shift_x:pix_stop_x+shift_x],
                    LOCAL_ACF_shx,
                    mode="same",
                    boundary='wrap'
                )
                # -- Normalization
                if norm_blocks!='No':
                    block_img_shx=tmp_img_shx[pix_start_y:pix_stop_y, pix_start_x+shift_x:pix_stop_x+shift_x]
                    Local_img_stdv_shx=(np.percentile(block_img_shx,[84]) - np.percentile(block_img_shx,[16])) /2.
                    if norm_blocks=='Local':
                        block_ref_shx=ref_image[pix_start_y:pix_stop_y, pix_start_x+shift_x:pix_stop_x+shift_x]
                        Local_ref_stdv_shx=(np.percentile(block_ref_shx,[84]) - np.percentile(block_ref_shx,[16])) /2.
                        tmp_img_shx[pix_start_y:pix_stop_y, pix_start_x+shift_x:pix_stop_x+shift_x]=(tmp_img_shx[pix_start_y:pix_stop_y, pix_start_x+shift_x:pix_stop_x+shift_x]/Local_img_stdv_shx)*Local_ref_stdv_shx
                    if norm_blocks=='Global':
                        tmp_img_shx[pix_start_y:pix_stop_y, pix_start_x+shift_x:pix_stop_x+shift_x]=(tmp_img_shx[pix_start_y:pix_stop_y, pix_start_x+shift_x:pix_stop_x+shift_x]/Local_img_stdv_shx)*Global_ref_stdv

            # POSITION + shift x + shift y ############################################
            if (pix_start_y+shift_y < img_height and pix_stop_y+shift_y <= img_height and pix_start_x+shift_x < img_width and pix_stop_x+shift_x <= img_width):
                tmp_img_shxy[pix_start_y+shift_y:pix_stop_y+shift_y, pix_start_x+shift_x:pix_stop_x+shift_x] = scipy.signal.convolve2d(
                    input_image[pix_start_y+shift_y:pix_stop_y+shift_y, pix_start_x+shift_x:pix_stop_x+shift_x],
                    LOCAL_ACF_shxy,
                    mode="same",
                    boundary='wrap'
                )
                # -- Normalization
                if norm_blocks!='No':
                    block_img_shxy=tmp_img_shxy[pix_start_y+shift_y:pix_stop_y+shift_y, pix_start_x+shift_x:pix_stop_x+shift_x]
                    Local_img_stdv_shxy=(np.percentile(block_img_shxy,[84]) - np.percentile(block_img_shxy,[16])) /2.
                    if norm_blocks=='Local':
                        block_ref_shxy=ref_image[pix_start_y+shift_y:pix_stop_y+shift_y, pix_start_x+shift_x:pix_stop_x+shift_x]
                        Local_ref_stdv_shxy=(np.percentile(block_ref_shxy,[84]) - np.percentile(block_ref_shxy,[16])) /2.
                        tmp_img_shxy[pix_start_y+shift_y:pix_stop_y+shift_y, pix_start_x+shift_x:pix_stop_x+shift_x]=(tmp_img_shxy[pix_start_y+shift_y:pix_stop_y+shift_y, pix_start_x+shift_x:pix_stop_x+shift_x]/Local_img_stdv_shxy)*Local_ref_stdv_shxy
                    if norm_blocks=='Global':
                        tmp_img_shxy[pix_start_y+shift_y:pix_stop_y+shift_y, pix_start_x+shift_x:pix_stop_x+shift_x]=(tmp_img_shxy[pix_start_y+shift_y:pix_stop_y+shift_y, pix_start_x+shift_x:pix_stop_x+shift_x]/Local_img_stdv_shxy)*Global_ref_stdv



            # When the approximate size of the boxes does not perfectly fit the image size...
            # ...along x
            borrow_x=(j+1)*(img_width/N_SECT_axis - round(img_width/N_SECT_axis)) - additive_pix_x_tot
            additive_pix_x=round(borrow_x - int(borrow_x))
            additive_pix_x_tot=additive_pix_x_tot+additive_pix_x
            pix_start_x = pix_stop_x
            pix_stop_x = pix_stop_x + SECT_Npix_x + additive_pix_x
                
        # When the approximate size of the boxes does not perfectly fit the image size...
        # ...along y
        borrow_y=(i+1)*(img_height/N_SECT_axis - round(img_height/N_SECT_axis)) - additive_pix_y_tot
        additive_pix_y=round(borrow_y - int(borrow_y))
        additive_pix_y_tot=additive_pix_y_tot+additive_pix_y
        pix_start_y = pix_stop_y
        pix_stop_y = pix_stop_y + SECT_Npix_y + additive_pix_y


    # COMPUTE MERGED IMAGE  -----------------------------------------------
    tmp_x=tmp_img*map_weights_x + tmp_img_shx*(amp-map_weights_x)
    tmp_y=tmp_img*map_weights_y + tmp_img_shy*(amp-map_weights_y)
    tmp_xy=tmp_img_shxy*(amp-map_weights_x)*(amp-map_weights_y)
    output_image=tmp_x*map_weights_y + tmp_y*map_weights_x + tmp_xy*2

    # FLAT FIELDING -------------------------------------------------------
    # IMPORTANT NOTE: the same weights used above, for computing the merged 
    #                 image, must be used using the same order

    Flat_img_ref=np.copy(map_weights_x) 
    Flat_img_ref[:]=1.
    Flat_img=np.copy(Flat_img_ref)
    Flat_img[tmp_img==0]=0.
    Flat_img_shx=np.copy(Flat_img_ref)
    Flat_img_shy=np.copy(Flat_img_ref)
    Flat_img_shxy=np.copy(Flat_img_ref)
    Flat_img_shx[tmp_img_shx==0]=0.
    Flat_img_shy[tmp_img_shy==0]=0.
    Flat_img_shxy[tmp_img_shxy==0]=0.

    Flat_tmp_x=Flat_img*map_weights_x+Flat_img_shx*(amp-map_weights_x)
    Flat_tmp_y=Flat_img*map_weights_y+Flat_img_shy*(amp-map_weights_y)
    Flat_tmp_xy=Flat_img_shxy*(amp-map_weights_x)*(amp-map_weights_y)

    Flat_main=Flat_tmp_x*map_weights_y+Flat_tmp_y*map_weights_x+Flat_tmp_xy*2
    
    output_image=output_image/Flat_main

    return output_image#*0.5


def create_elliptical_annular_mask(image_shape, center, b_inner, a_inner, b_outer, a_outer, angle):

    """ 
    generates an elliptical annular mask with specified
    inner and outer ellipse parameters. The mask is applied
    to an image with the specified shape, and the ellipses 
    are rotated by a given angle around a specified center.
    The function uses NumPy for array operations.

    image_shape: Tuple representing the shape of the image.
    center: Tuple representing the center coordinates of the ellipses.
    b_inner: Semi-minor axis of the inner ellipse.
    a_inner: Semi-major axis of the inner ellipse.
    b_outer: Semi-minor axis of the outer ellipse.
    a_outer: Semi-major axis of the outer ellipse.
    angle: Rotation angle (in degrees) applied to the ellipses.
    """

    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    print('WARNING: THIS ANGLE MUST BE CHECKED')
    print(' Function')
    print('create_elliptical_annular_mask()')
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

    angle=angle+90

    mask = np.zeros(image_shape, dtype=bool)
    y, x = np.ogrid[:image_shape[0], :image_shape[1]]

    # OPTION 1: Clockwise rotation
    #x_rot = (x - center[0]) * np.sin(np.deg2rad(angle)) + (y - center[1]) * np.cos(np.deg2rad(angle))
    #y_rot = (x - center[0]) * np.cos(np.deg2rad(angle)) + (y - center[1]) * np.sin(np.deg2rad(angle))

    # OPTION 2: Counter-clockwise rotation
    x_rot = (x - center[0]) * np.cos(np.deg2rad(angle)) + (y - center[1]) * np.sin(np.deg2rad(angle))
    y_rot = -(x - center[0]) * np.sin(np.deg2rad(angle)) + (y - center[1]) * np.cos(np.deg2rad(angle))

    distance = (x_rot / a_outer)**2 + (y_rot / b_outer)**2
    inner_distance = (x_rot / a_inner)**2 + (y_rot / b_inner)**2
    mask[(distance <= 1) & (inner_distance >= 1)] = True
    return mask

def compute_elliptical_structure(in_image, sigma_thresh, stripes_width, stripes_length, AXES_RATIO, P_ANGLE):
    #
    # Given an input image, for each pixel it computes the average 
    # computed over a strip whose width and length are defined
    # (in pixels) using the input parameters (stripes_width and 
    # stripes_length). Each strip orientation is concentrical
    # to the image center and defined by the parameters 
    # "AXES_RATIO" (ratio between a and b axes) and by P_ANGLE
    # (defining the position angle of the ellipses used).
    # Only pixels with intensity higher than sigma_thresh times
    # the RMS of the image are taken into account.
    #
    # AXES_RATIO = a/b

    Elliptically_averaged_image=np.copy(in_image)
    Elliptically_averaged_image[:]=0.

    # Calculate the center of the image
    img_center_x = in_image.shape[1] / 2
    img_center_y = in_image.shape[0] / 2
    
    ###########################################
    img_idx_set=np.copy(in_image)
    img_idx_set[:]=0
    ###########################################
    # Compute size (width, height) of the image 
    width=in_image.shape[0]
    height=in_image.shape[1]
    max_size=np.max([width,height])
    # Create a meshgrid to represent the x and y coordinates of each pixel
    xx, yy = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    # Calculate the arctan(y/x)
    pixel_angles = np.arctan2(yy, xx)
    # Angle going from 0 to 2pi
    pixel_angles = pixel_angles-np.min(pixel_angles)
    pixel_angles = pixel_angles/np.max(pixel_angles)
    pixel_angles = pixel_angles*2.0*np.pi
    #  Actual X and Y coordinates [pixels]
    x_coord, y_coord = np.meshgrid(np.linspace(0,width-1 , width), np.linspace(0, height-1, height))
    ###########################################

    # Number of pixels to consider inside a corona:
    N_PIX_COR_W=stripes_width
    N_PIX_COR_L=stripes_length//2

    ####################
    # isotropic RMS
    box_size=np.round(np.sqrt(np.round(stripes_width)*np.round(stripes_length)))
    N_elem_box=np.round(stripes_width)*np.round(stripes_length)
    SIGMA_BOX=stdv_of_array_blocks_means(in_image,box_size)
    ####################

    # Iterate over increasing distances (values of 'b')
    #for b in range(1, max(in_image.shape[1], in_image.shape[0]) // 2, N_PIX_COR):  # Limit the range to the smaller size
    
    for b in range(1, max(in_image.shape[1], in_image.shape[0]), int(np.round(N_PIX_COR_W))):  # Limit the range to the smaller size
        #a = b / (FWHM_PIX_y / FWHM_PIX_x)  # Corresponding semi-major axis for the elliptical annulus
        a = b * AXES_RATIO # Corresponding semi-major axis for the elliptical annulus
        
        # Create an elliptical annular mask with the current 'a' and 'b' values
        elliptical_annular_mask = create_elliptical_annular_mask(in_image.shape, (img_center_x, img_center_y), b - N_PIX_COR_W, a - N_PIX_COR_W, b + N_PIX_COR_W, a + N_PIX_COR_W, P_ANGLE)
    
        if np.size(in_image[elliptical_annular_mask])>0:
            

            ######################################
            img_idx_set[elliptical_annular_mask]=1
            idx_set=np.where(img_idx_set==1)
            IDX_cols,IDX_rows=idx_set
            # iterate through all the pixels in this elliptical corona
            for ii in range(np.size(in_image[elliptical_annular_mask])):
                #this_x=x_coord[elliptical_annular_mask][ii]
                #this_y=y_coord[elliptical_annular_mask][ii]

                # Find closest pixels inside the corona
                idx_average=np.where(np.sqrt(((y_coord[idx_set]-y_coord[elliptical_annular_mask][ii])**2) + ((x_coord[idx_set]-x_coord[elliptical_annular_mask][ii])**2)) < N_PIX_COR_L )

                strip_average=np.mean(in_image[elliptical_annular_mask][idx_average])
                if np.abs(strip_average) > sigma_thresh*SIGMA_BOX:
                    # Update the pixels in the copy of the input image with the average value
                    #Elliptically_averaged_image[IDX_cols[ii],IDX_rows[ii]]= average_value
                    Elliptically_averaged_image[IDX_cols[ii],IDX_rows[ii]]= strip_average # np.mean(in_image[elliptical_annular_mask][idx_average])

            img_idx_set[elliptical_annular_mask]=0 # Reset list of pixels considered for the average
            ######################################

    
    return Elliptically_averaged_image


def stdv_of_array_blocks_means(array_2d, box_size):
    """
    Calculate the standard deviation of the mean values within blocks of specified size in a 2D array.

    Parameters:
    - array_2d (numpy.ndarray): Input 2D array containing data.
    - box_size (int): Size of the square blocks used for calculating means.
    
    Returns:
    - float: Standard deviation of the mean values within the specified blocks.
    """
    # Get the shape of the input array
    rows, cols = array_2d.shape

    # Calculate the number of complete boxes in each dimension
    num_boxes_rows = int(rows // round(box_size))
    num_boxes_cols = int(cols // round(box_size))
    
    # TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST
    # Initialize an array to store the average values of each block
    Block_means = np.zeros((num_boxes_rows,num_boxes_cols))

    # Iterate over blocks
    for y in range(num_boxes_rows):
        for x in range(num_boxes_cols):
            Block_means[y,x]=np.mean(array_2d[y*round(box_size):(y+1)*round(box_size) , x*round(box_size): (x+1)*round(box_size)])
    # TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST


    # Trim the array to a multiple of box_size
    trimmed_array = array_2d[:num_boxes_rows*round(box_size),:num_boxes_cols*round(box_size)]

    # Calculate the standard deviation of the block means
    #std_deviation_of_block_means = np.std(Block_means)
    std_deviation_of_block_means = (np.percentile(Block_means,[84]) - np.percentile(Block_means,[16])) /2.
    std_deviation_of_block_means=std_deviation_of_block_means[0]
    #print("std_deviation_of_block_means:")
    #print(std_deviation_of_block_means)
    #print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    #AA,BB,CC,DD=get_histogram(Block_means,perc_min=2.5,perc_max=97.5,SHOW_PLOT=True,get_perc=-99)

    ## TEST Save resampled image to a file 
    #TEST_name='Boxes_'+original_image_name
    #fits.writeto(directory+'/'+TEST_name, Block_means , overwrite=True)

    return std_deviation_of_block_means
    


def radial_patterns(input_image, sigma_thresh, delta_theta, delta_rad, SHOW_PLOT=False):
    # Given a 2D numpy array (an image), it computes the average
    # of the pixel aligned with the image center along each direction
    # The average is computed inside circular corona sections whose 
    # size is defined by the following parameters:
    # > delta_theta [radians], representing the HALF length of the circular 
    #                          corona section (i.e., the bended strip) 
    #                          inside which the average value is computed
    # > delta_rad [pixels], representing The half width of the  circular 
    #                          corona section (i.e., the bended strip) 
    #                          inside which the average value is computed
    # Important Notes: this functions can be used in successive runs to 
    #              obtain radial patterns and angular patterns. this can 
    #              be achieved by opportunely setting delta_theta and 
    #              delta_rad so that they enclose thin stripes oriented 
    #              circularly (delta_theta>>, delta_rad<<) or radially 
    #              (delta_theta<<, delta_rad>>) with respect to the image 
    #              center.
    ####################################################################

    # Create an output image with the same shape as the input
    output_image = np.copy(input_image)
    output_image[:,:] = 0. 

    # Create an array with position angles as values
    pixel_angles = np.copy(input_image)
    pixel_angles[:,:] = 0.
    
    # Compute size (width, height) of the image 
    width=input_image.shape[0]
    height=input_image.shape[1]
    max_size=np.max([width,height])

    # Create a meshgrid to represent the x and y coordinates of each pixel
    xx, yy = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    
    # Calculate the arctan(y/x)
    pixel_angles = np.arctan2(yy, xx)
    # Angle going from 0 to 2pi
    pixel_angles = pixel_angles-np.min(pixel_angles)
    pixel_angles = pixel_angles/np.max(pixel_angles)
    pixel_angles = pixel_angles*2.0*np.pi

    #  Calculate distance of the pixels from the image center
    new_x=(xx/np.max(xx))*float(width//2) # -128 to 128. min distance=0.5 pixels
    new_y=(yy/np.max(yy))*float(height//2) # -128 to 128. min distance=0.5 pixels
    pixel_distances = np.sqrt(new_x**2 + new_y**2)

    #  Actual X and Y coordinates [pixels]
    x_coord, y_coord = np.meshgrid(np.linspace(0,width-1 , width), np.linspace(0, height-1, height))
    #x_coord=new_x-np.min(new_x)
    #y_coord=new_y-np.min(new_y)

    if SHOW_PLOT==True:
        plt.imshow(pixel_distances, cmap='gray')
        plt.title('pixel distances')
        # invert the y-axis (to make it consistent with visualization in ds9)
        plt.gca().invert_yaxis()
        plt.show()

    if SHOW_PLOT==True:
        plt.imshow(pixel_angles, cmap='gray')
        plt.title('pixel angle')
        # invert the y-axis (to make it consistent with visualization in ds9)
        plt.gca().invert_yaxis()
        plt.show()

    approx_stripes_half_size=max_size*delta_theta/(np.pi/2.)
    #box_size=np.round(np.sqrt(((delta_rad)*2.)*((approx_stripes_half_size)*2.)))
    box_size=np.round(np.sqrt(((np.round(delta_rad))*2.)*((np.round(approx_stripes_half_size))*2.)))
    N_elem_box=box_size*box_size

    SIGMA_BOX=stdv_of_array_blocks_means(input_image,box_size)

    # Iterate through all pixels in the image
    # For each pixel, compute position angle
    for ii in range(input_image.shape[0]):
        for jj in range(input_image.shape[1]):
            angle_i = pixel_angles[ii,jj]
            distance_i = pixel_distances[ii,jj]
            delta_theta_corr=delta_theta*max_size/np.max([distance_i,1])

            # Remember:
            # & = bitwise logical operator "and"
            # | = bitwise logical operator "or"

            # Normal case (0°<theta<360°)
            if (angle_i>=delta_theta_corr) and (angle_i<=2.0*np.pi-delta_theta_corr):

                STRIP=input_image[(pixel_angles > angle_i-delta_theta_corr)
                                  & (pixel_angles < angle_i+delta_theta_corr)
                                  & (pixel_distances > distance_i-delta_rad ) 
                                  & (pixel_distances < distance_i+delta_rad )]

            # CLOSE TO THE EDGE (theta>~0°)
            if (angle_i<delta_theta_corr):

                STRIP =input_image[(  (pixel_angles < angle_i+delta_theta_corr)
                                      | (pixel_angles > 2.0*np.pi - delta_theta_corr + angle_i) )
                                   & (pixel_distances > distance_i-delta_rad ) 
                                   & (pixel_distances < distance_i+delta_rad ) ]

            # CLOSE TO THE EDGE (theta<~360°)
            if (angle_i>2.0*np.pi-delta_theta_corr):

                STRIP=input_image[(  (pixel_angles > angle_i-delta_theta_corr)
                                     | (pixel_angles < delta_theta_corr - (2.0*np.pi - angle_i) ) )
                                  & (pixel_distances > distance_i-delta_rad ) 
                                  & (pixel_distances < distance_i+delta_rad ) ]

            strip_average=np.mean(STRIP)

            #strip_percts=np.percentile(STRIP,[30,70])
            #strip_average=np.mean(STRIP[(STRIP>strip_percts[0]) & (STRIP<strip_percts[1])])
            #strip_average=(strip_percts[0]+strip_percts[1])/2.

            if np.abs(strip_average) > sigma_thresh*SIGMA_BOX:
                output_image[ii,jj] = strip_average

#            if output_image[ii,jj] != output_image[ii,jj]:
#                print('WARNING: Nan found in radial_background()')
#                stop()
#
    return output_image





# read image header and data
def read_fits_image(image_name):
    """
    Reads a FITS image and returns the data and header
    as separate variables.
    -----------
    PARAMETERS:
    image_name : str
        The file path of the FITS image.
    -----------
    RETURNS:
    data : numpy.ndarray
        The data array of the FITS image.
    header : astropy.io.fits.header.Header
        The header of the FITS image.
    --------
    CALL THIS FUNCTION AS FOLLOWS:
    data, header = read_fits_image('/path/to/fits/image.fits')
    """
    with fits.open(image_name) as hdulist:
        data = hdulist[0].data
        header = hdulist[0].header
    return data, header


def get_keyword_value(header, keyword):
    """
    Get the value of a keyword in a fits header, handling HISTORY entries as well.

    Parameters:
        header (astropy.io.fits.Header): The header object to search for the keyword.
        keyword (str): The name of the keyword to search for.

    Returns:
        The value of the keyword (str, float, int, etc.) or None if the keyword is not found.
    """
    # Try to get the keyword value from the header normally
    value = header.get(keyword)

    if value is None:
        # If the keyword isn't found in the header, look for it in the HISTORY entries

        for card in header.cards:
            if card.keyword.startswith('HISTORY'):
                print (card)
                imax=np.size(card)
                for ii in range(imax):
                    print(card[ii])
                    if keyword in card[ii]:
                        try:
                            value = float(card[ii].split(keyword + "='")[1].split("'")[0])
                        except ValueError:
                            value = card[ii].split(keyword + "='")[1].split("'")[0]

        if value is None:
        # If the keyword isn't found among the HISTORY entries with the previous search,
        # search using regular expression (it's a non general, specific solution)

            if keyword in ["FWHM_maj", "FWHM_min"]:
            
                match = re.search(rf"{keyword} used in restoration: (\d+\.\d+) by (\d+\.\d+)", card.comment)
                if match:
                    if keyword == "FWHM_maj":
                        value = float(match.group(1))
                    elif keyword == "FWHM_min":
                        value = float(match.group(2))
                elif keyword == "POS_ANGLE":
                    match = re.search(rf">2 \(arcsec\) at pa (-?\d+\.\d+) \(deg\)", card.comment)
                    if match:
                        value = float(match.group(1))

    return value

def get_filter_name(FWHM_PIX):
    # Automatically define the most appropriate filter name
    # given the FWHM of the image

    #Filter selection
    G_filters_poss=np.array([1.5,2.0,2.5,3.0,4.0,5.0])
    #------------------------------------------
    # GOOD FOR A CENTRAL CALIBRATOR (header FWHM)
    ID_FIL=np.where(abs(FWHM_PIX-G_filters_poss) == min(abs(FWHM_PIX-G_filters_poss)))
    #------------------------------------------
    # GOOD FOR REAL FWHM = 2 x header FWHM
    # ID_FIL=np.where(abs(2.*FWHM_PIX-G_filters_poss) == min(abs(2.*FWHM_PIX-G_filters_poss)))
    #------------------------------------------
    if ID_FIL[0] == 0 : filter='1.5_3x3'
    if ID_FIL[0] == 1 : filter='2.0_5x5'
    if ID_FIL[0] == 2 : filter='2.5_5x5'
    if ID_FIL[0] == 3 : filter='3.0_7x7'
    if ID_FIL[0] == 4 : filter='4.0_7x7'
    if ID_FIL[0] == 5 : filter='5.0_9x9'
    filt_name="gauss_"+filter+".conv"

    return filt_name


def run_sex1(imagename_input,imagename_output,SEx_folder,filter_name,PIXEL_SCALE,FWHM_PIX_x,FWHM_PIX_y,BACK_SIZE,BACK_FILTERSIZE, img_type="BACKGROUND", DETECT_THRESH=1.5, ANALYSIS_THRESH=1.5, DETECT_MINAREA_BEAM=2.0):
    #-------------------------------------------------------
    # This function outputs an image (imagename_output) of 
    # the large scale background of the input image
    #-------------------------------------------------------

    # Internal parameters #########################
    BEAM_AREA=(1./(4.*np.log(2.))) * (np.pi*(FWHM_PIX_x)*(FWHM_PIX_y))
    FWHM_PIX=(FWHM_PIX_x+FWHM_PIX_y)/2.
    FWHM_asec=FWHM_PIX/PIXEL_SCALE
    # SExtractor parameters #######################
    # DETECT_THRESH=1.5 # different from IDL test (was 1.2)
    # ANALYSIS_THRESH=1.5 # different from IDL test (was 1.2)
    # DETECT_MINAREA=2.0*np.pi*(FWHM_PIX_x/2.355)*(FWHM_PIX_y/2.355) #(=BEAM_AREA[pixels])
    DETECT_MINAREA=DETECT_MINAREA_BEAM*np.pi*(FWHM_PIX_x/2.355)*(FWHM_PIX_y/2.355) #(=BEAM_AREA[pixels])
    MAG_ZEROPOINT=8.9+2.5*np.log10(BEAM_AREA)
    # n_fwhm=4.
    # BACK_SIZE=n_fwhm*FWHM_PIX ####### PARAMETER CONTROLLED FROM OUTSIDE
    # BACK_FILTERSIZE=2 ####### PARAMETER  CONTROLLED FROM OUTSIDE
    ###############################################
    # set "img_type" to select the type of sextractor "check image" desired
    # (BACKGROUND, BACKGROUND_RMS, -BACKGROUND, FILTERED, OBJECTS, -OBJECTS,
    #  SEGMENTATION, or APERTURES
    # create temporary folder to store temporary files
    tmp_folder='tmp_folder'
    os.system('mkdir '+tmp_folder)
    # RUN SEXTRACTOR TO GET BACKGROUND
    os.system('source-extractor '+imagename_input+' -c '+SEx_folder+'config_A.txt -CATALOG_NAME '+tmp_folder+'/tmp_cat.fits -PARAMETERS_NAME '+SEx_folder+'default.param -FILTER_NAME '+SEx_folder+filter_name+' -STARNNW_NAME '+SEx_folder+'default.nnw -DETECT_THRESH '+str(DETECT_THRESH)+' -ANALYSIS_THRESH '+str(ANALYSIS_THRESH)+' -DETECT_MINAREA '+str(round(DETECT_MINAREA))+' -SATUR_LEVEL 50000.0'+' -MAG_ZEROPOINT '+str(MAG_ZEROPOINT)+' -GAIN 0.0 -PIXEL_SCALE '+str(PIXEL_SCALE)+' -SEEING_FWHM '+str(FWHM_asec)+' -BACK_SIZE '+str(BACK_SIZE)+' -BACK_FILTERSIZE '+str(BACK_FILTERSIZE) +' -BACKPHOTO_TYPE LOCAL -WEIGHT_TYPE BACKGROUND -CHECKIMAGE_TYPE '+img_type+' -CHECKIMAGE_NAME '+imagename_output)

    # remove temporary folder and temporary files
    os.system('rm -r tmp_folder')


def resample(x, y, N):
    x_res = np.array([])
    y_res = np.array([])
    
    for i in range(len(x) - 1):
        x_start = x[i]
        y_start = y[i]
        x_end = x[i + 1]
        y_end = y[i + 1]
        
        x_res = np.append(x_res, x_start)
        y_res = np.append(y_res, y_start)
        
        for j in range(1, N):
            x_interpolated = x_start + (x_end - x_start) * j / N
            y_interpolated = y_start + (y_end - y_start) * j / N
            x_res = np.append(x_res, x_interpolated)
            y_res = np.append(y_res, y_interpolated)
    
    x_res = np.append(x_res, x[-1])
    y_res = np.append(y_res, y[-1])
    
    return x_res, y_res



def get_histogram(input_array,perc_min=2.5,perc_max=97.5,SHOW_PLOT=False,get_perc=-99):
    # §§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
    # Automatically creates an histogram of the values
    # from a bi-dimensional input numpy array.
    # It automatically set the bin width once two percentile 
    # limits are set (perc_min and perc_max).
    # The values corresponding to some additional percentiles
    # can also be obtained in output by setting "get_perc" to
    # an array of values ranging from 0 to 100. The values are  
    # provided through "perc_val" (see below) only if get_perc
    # is set as described.
    # IT RETURNS:
    #  xhist= central x of the bins 
    #  yhist= value of the historgam bins
    #  bin_width= width of the bins (automatically selected)
    #  minmax_hs= min and max values used to bin the histogram
    #             (they correspond to the percentiles perc_min 
    #             and perc_max indicated in input)
    #  perc_val= values corresponding to the percentiles indicated
    #            in input (get_perc vector)
    #
    # §§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
    # REQUIRES:
    # import numpy as np
    # import matplotlib.pyplot as plt
    # §§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
    # --------------------------------------------
    # Flatten the multi-dimensional input array:
    F_input_array=input_array.flatten()
    # --------------------------------------------
    # Set histogram binning 
    minmax_hs=np.percentile(F_input_array,[perc_min,perc_max])# ex. 2.5% and 97.5% percentiles
    print("WARNING: if the distribution is not simmetrical, the binning will probably fail")
    print("         especially if you use percentiles 40-60%")

    # ORIGINAL (WRONG)
    #bin_vals = np.linspace(minmax_hs[0]*3., minmax_hs[1]*3., 100) # create 100 evenly spaced bins

    # REPLACED BY:
    minhist=minmax_hs[0]-(np.median(F_input_array)-minmax_hs[0])*2.
    maxhist=minmax_hs[1]+(minmax_hs[1]- np.median(F_input_array))*2.
    bin_vals = np.linspace(minhist, maxhist, 100) # create 100 evenly spaced bins

        # Compute histogram
    hist, bin_edges = np.histogram(F_input_array, bins=bin_vals)
    #------------------------------------------------------------------
    xhist=bin_vals[:-1] # X values of the histogram
    yhist=hist # Y values of the histogram
    bin_width=bin_vals[1]-bin_vals[0] # width of the histogram's bins
    #------------------------------------------------------------------
    print(perc_min,perc_max)
    print('======================================')
    if SHOW_PLOT==True:
        # Plot - histogram of the original image
        plt.bar(xhist,yhist, width=bin_width)
        ## Plot x and y hist to make sure they coincide
        #plt.plot(xhist,yhist,color="red")
        # Set the axis labels
        #plt.title('')
        plt.xlabel('Pixel Values')
        plt.ylabel('Count')
        # Show the plot
        plt.show()

    if np.sum(get_perc)==-99:
        return xhist, yhist, bin_width, minmax_hs
        
    if np.sum(get_perc)!=-99:
        # Compute additional percentile values
        perc_val=np.percentile(F_input_array,get_perc)
        return xhist, yhist, bin_width, minmax_hs, perc_val



   
def get_gauss_distrib(input_array,perc_min,perc_max,SHOW_PLOT=False,plot_title=''):
    # §§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
    # Computes the histogram of the values in a 
    # bi-dimensional input numpy array (input_array)
    # and returns the parameters of the Gaussian 
    # fit that better describes the distribution
    # RETURNS:
    #  popt= [Gauss_peak, Gauss_mean, Gauss_sigma]
    #  pcov= Covariances associated with previous params.
    #  x_fit= array of x values of the actual fit
    #  y_fit= array of y values of the actual fit
    # §§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
    # REQUIRES:
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from scipy.stats import norm
    # from scipy.optimize import curve_fit
    # §§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
    # --------------------------------------------
    # Flatten the multi-dimensional input array:
    F_input_array=input_array.flatten()
    # --------------------------------------------
    # Compute histogram
    xhist, yhist, bin_width, minmax_hs = get_histogram(input_array,perc_min,perc_max,SHOW_PLOT=False)
    # --------------------------------------------
    
    # Define the Gaussian function
    def gaussian(x, a, b, c):
        return a * np.exp(-(x - b)**2 / (2 * c**2))
    # Select only elements between perc_min and perc_max and put them in a flat array
    valid_idx=np.where((F_input_array > minmax_hs[0]) & (F_input_array < minmax_hs[1]))
    valid_data=F_input_array[valid_idx]
    # Estimate the INITIAL parameters for the Gaussian function
    G_mean = np.median(valid_data) #np.median(xhist)
    G_std_dev = np.std(valid_data)
    G_max_val = np.max(yhist)
    # Fit the Gaussian function to the data
    popt, pcov = curve_fit(gaussian, xhist, yhist, p0=[G_max_val, G_mean, G_std_dev])
    # Main FIT parameters corresponding to the fitted Gaussian function
    G_max_val_fit,G_mean_fit,G_std_dev_fit=popt
    x_fit = xhist
    y_fit = gaussian(xhist,G_max_val_fit,G_mean_fit,G_std_dev_fit)# gaussian(xhist, *popt)
    if SHOW_PLOT==True:
        # Plot - histogram of the original image
        plt.bar(xhist,yhist, width=bin_width)
        ## Plot x and y hist to make sure they coincide
        #plt.plot(xhist,yhist,color="blue")
        plt.plot(x_fit,y_fit,color="red",label='Gauss Fit')
        # Plot average and peak values (from fit)
        plt.plot(np.array([G_mean_fit,G_mean_fit]),np.array([0,G_max_val_fit]),color='yellow')
      # Plot +1 sigma (from fit)
        plt.plot(np.array([G_mean_fit+G_std_dev_fit,G_mean_fit+G_std_dev_fit]),np.array([0,G_max_val_fit]),color='yellow')
      # Plot -1 sigma (from fit)
        plt.plot(np.array([G_mean_fit-G_std_dev_fit,G_mean_fit-G_std_dev_fit]),np.array([0,G_max_val_fit]),color='yellow')
        
        # Set the axis labels
        #plt.title('')
        plt.xlabel('Pixel Values')
        plt.ylabel('Count')
        plt.title(plot_title)
        plt.legend()
        # Show the plot
        plt.show()
    return popt, pcov, x_fit, y_fit


def sim_pix_distrib(input_array,perc_min,perc_max,sim_img='replicate',SHOW_PLOT=False,plot_title=''):
    # §§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
    # Computes the histogram of the values in a bi-dimensional 
    # input numpy array (input_array) using get_histogram(), defined
    # in this same program, and returns:
    # 1) an output array with the same distribution of pixel values.
    # 2A) if sim_img is set to 'Gfit': the sigma of the best fitting
    #     Gaussian distribution
    # 2B) if sim_img is set to 'replicate': The simmetrized uncertainty
    #     (sigma-like) based computed from the percentiles 
    #     as (84% perc - 16% perc)/2 
    # 3) if sim_img is set to 'replicate': The values corresponding to 
    #    the 16% and 84% percentiles
    # - Using sim_img='Gfit', the distribution is computed as the 
    # Gaussian fit of the original distribution;
    # - Using sim_img='replicate', the distribution is identical
    # to the one in input
    # §§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
    # REQUIRES:
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from scipy.stats import norm
    # from scipy.optimize import curve_fit
    # §§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
    # Get x and y sizes of the input image
    ysize=input_array.shape[0] # [2] if input_array is an ALMA 2D imgs (and not just a 2d np array)
    xsize=input_array.shape[1] # [3] if input_array is an ALMA 2D imgs (and not just a 2d np array)
    # --------------------------------------------
    # Create a similar (void) output array
    output_array=np.copy(input_array) # Modifying the copy you will not modify the original
    output_array[:]=0
    # --------------------------------------------
    # Flatten the multi-dimensional input array:
    F_input_array=input_array.flatten()
    # --------------------------------------------
    # Compute histogram
    xhist, yhist, bin_width, minmax_hs = get_histogram(input_array,perc_min,perc_max,SHOW_PLOT=False)
    # --------------------------------------------

    if sim_img=='Gfit':
        popt, pcov, x_fit, y_fit = get_gauss_distrib(input_array,perc_min,perc_max,SHOW_PLOT=False)
        # Main FIT parameters corresponding to the fitted Gaussian function
        G_max_val_fit,G_mean_fit,G_std_dev_fit=popt
        #------------------------------------------------------------------
        #output_array = np.random.choice(x_fit, size=(ysize,xsize), p=y_fit/y_fit.sum())
        x_fit_res, y_fit_res=resample(x_fit,y_fit,100)
        rng = np.random.default_rng()
        output_array =rng.choice(x_fit_res, size=(ysize,xsize), replace=True, p=y_fit_res/y_fit_res.sum(), axis=0, shuffle=True)
        #------------------------------------------------------------------
        # COMPUTE MAIN PARAMETERS FOR SIM IMAGE
        # Flatten the output array:
        F_output_array=output_array.flatten()
        #------------------------------------------------------------------
        if SHOW_PLOT==True:
            # Compute histogram of the output array ONLY for visualization purposes
            minhist=minmax_hs[0]-(np.median(input_array)-minmax_hs[0])*2.
            maxhist=minmax_hs[1]+(minmax_hs[1]- np.median(input_array))*2.
            bin_vals = np.linspace(minhist, maxhist, 100) # create 100 evenly spaced bins

            # Compute histogram
            xhist_sim, yhist_sim, bin_width_sim, minmax_hs = get_histogram(output_array,perc_min,perc_max,SHOW_PLOT=False)


            #------------------------------------------------------------------

    if sim_img=='replicate':
        # OUTPUT PIXELS WITH SAME DISTRIBUTION AS INPUT ARRAY
        # Generate random pixel values with the same distribution as the 
        # original histogram. Histogram bin heights (normalized to sum to 1)
        # are passed as probabilities to numpy.random.choice
        #------------------------------------------------------------------
        # Resample histogram curve AND THEN pick up numbers from the original distribution 
        xhist_res, yhist_res=resample(xhist,yhist,100)
        rng = np.random.default_rng()
        output_array =rng.choice(xhist_res, size=(ysize,xsize), replace=True, p=yhist_res/yhist_res.sum(), axis=0, shuffle=True)

        #------------------------------------------------------------------
        # COMPUTE MAIN PARAMETERS FOR SIM IMAGE
        # Flatten the output array:
        F_output_array=output_array.flatten()
        #------------------------------------------------------------------
        # 16% and 84% percentiles of the input array distribution
        Perc_16_68=np.percentile(input_array,[16,84]) 
        # Simmetrized uncertainty (Gaussian sigma-like)
        Simm_Unc=(Perc_16_68[1]-Perc_16_68[0])/2.
        #------------------------------------------------------------------
        if SHOW_PLOT==True:
            # Compute histogram of the output array ONLY for visualization purposes
            minhist=minmax_hs[0]-(np.median(input_array)-minmax_hs[0])*2.
            maxhist=minmax_hs[1]+(minmax_hs[1]- np.median(input_array))*2.
            bin_vals = np.linspace(minhist, maxhist, 100) # create 100 evenly spaced bins
            #------------------------------------------------------------------
            # Compute histogram
            xhist_sim, yhist_sim, bin_width_sim, minmax_hs = get_histogram(output_array,perc_min,perc_max,SHOW_PLOT=False)

    if SHOW_PLOT==True:
        # Plot - histogram of the original image
        plt.bar(xhist,yhist, width=bin_width)
        ## Plot x and y hist to make sure they coincide
        #plt.plot(xhist,yhist,color="blue")
        if sim_img=='replicate':
            # Plot - histogram of the replicated image
            #plt.plot(xhist_sim,yhist_sim,color='red', label='Reproduced distribution')
            plt.plot(xhist_sim+bin_width/2.,yhist_sim,color='red', label='Reproduced distribution')
        # Plot the Gaussian fit
        if sim_img=='Gfit':
            plt.plot(x_fit,y_fit,color="black", label='Gauss fit')#,linestyle=':')
            plt.plot(xhist_sim+bin_width/2.,yhist_sim,color="red", label='Reproduced distribution')
            # Plot average and peak values (from fit)
            plt.plot(np.array([G_mean_fit,G_mean_fit]),np.array([0,G_max_val_fit]),color='yellow')
            # Plot +1 sigma (from fit)
            plt.plot(np.array([G_mean_fit+G_std_dev_fit,G_mean_fit+G_std_dev_fit]),np.array([0,G_max_val_fit]),color='yellow')
            # Plot -1 sigma (from fit)
            plt.plot(np.array([G_mean_fit-G_std_dev_fit,G_mean_fit-G_std_dev_fit]),np.array([0,G_max_val_fit]),color='yellow')
        
        # Set the axis labels
        #plt.title('')
        plt.xlabel('Pixel Values')
        plt.ylabel('Count')
        plt.title(plot_title)
        plt.legend()
        # Show the plot
        plt.show()
    if sim_img=='Gfit':
        return output_array,G_std_dev_fit

    if sim_img=='replicate':
        return output_array,Simm_Unc#,Perc_16_68[0],Perc_16_68[1]


def get_gaussian_PSF(xsize,ysize,FWHM_PIX_a,FWHM_PIX_b,P_ANGLE):
    # This function generates an image of the PSF given:
    # - a 2D numpy array with x and y size as big as the output image
    # - the FWHM of the Gaussian PSF, along the major axis [pixels]
    # - the FWHM of the Gaussian PSF, along the minor axis [pixels]
    # - the position angle [deg]
    #-----------------------------------------------------------------
    # A) create a void image filled with one's
    PSF_img=np.ones((ysize,xsize))
    # B) centre of the Gaussian PSF (x,y)
    midpoint_x=round(xsize/2)
    midpoint_y=round(ysize/2)
    # C) Position angle in radiants
    PA_rad=P_ANGLE*math.pi/180.0
    #------------------------------------------
    #  CORRECTION FACTOR
    # When convolving the pixel-scale random noise with the image PSF,
    # it should be considered that the pixels do not have size=0.
    # Therefore the size of the PSF to convolve with must be a little bit 
    # smaller than the actual PSF. The correction factor depends on the ratio
    # between the PSF size and size of the pixel (along the major an minor axes).
    # The correction is negligible when FWHM[pixels]>>1
    # VARIABLE CORRECTION FACTOR
    print("WARNING IVANO: THIS SHOULD BE CHECKED")
    corr_fact_a=1.0-1.0/ (abs(FWHM_PIX_a*math.sin(PA_rad)) + abs(FWHM_PIX_b*math.cos(PA_rad)))
    corr_fact_b=1.0-1.0/ (abs(FWHM_PIX_a*math.cos(PA_rad)) + abs(FWHM_PIX_b*math.sin(PA_rad)))
    # This one seems to work (but not always) MUST BE CHECKED TESTING DIFFERENT PA_ANGLES
#    corr_fact_a=1.0-1.0/abs( FWHM_PIX_a*math.cos(PA_rad) - FWHM_PIX_b*math.sin(PA_rad) )
#    corr_fact_b=1.0-1.0/abs( -FWHM_PIX_b*math.cos(PA_rad) - FWHM_PIX_a*math.cos(PA_rad) )
#    FIXED CORRECTION FACTOR:
#    corr_fact_a=0.755  
#    corr_fact_b=0.755  
#    print(corr_fact_a,corr_fact_b)
    #------------------------------------------
    WW_x=0
    while WW_x < xsize:
        WW_y=0
        while WW_y < ysize:
            ww_ax=(WW_y-midpoint_y)*math.cos(PA_rad)-(WW_x-midpoint_x)*math.sin(PA_rad)
            ww_ay=-(WW_y-midpoint_y)*math.sin(PA_rad)-(WW_x-midpoint_x)*math.cos(PA_rad)
            # Gaussian PSF with shape obtained from the actual FWHM
            G_ax=math.exp(-0.5*(  ww_ax/(corr_fact_a*FWHM_PIX_a/2.355)  )**2)
            G_ay=math.exp(-0.5*(  ww_ay/(corr_fact_b*FWHM_PIX_b/2.355)  )**2)
            G_tot=G_ax*G_ay
            # PSF_img[WW_x,WW_y]=PSF_img[WW_x,WW_y]*G_tot # works in IDL
            PSF_img[WW_y,WW_x]=PSF_img[WW_y,WW_x]*G_tot # x and y inverted in python
                                                             
            WW_y=WW_y+1
        WW_x=WW_x+1

    return PSF_img



def get_filtered_img(input_array,SMALL_SCALE_THR,LARGE_SCALE_THR, show_imgs=False, save_spec_to_file=False, filename='magnitude_spectrum.fits'):
    # Computes the Fast Fourier Transform (FFT) of an input image
    # (passed through an input array) and filters out the spatial 
    # frequencies above and below the thresholds specified by the
    # user using SMALL_SCALE_THR and LARGE_SCALE_THR, respectively
    # corresponding to the smallest and largest scales to be
    # filtered (i.e., selected), expressed in pixels.
    # Requires:
    #  from scipy.fft import fft2, ifft2, fftshift
    #-----------------------------------------------------------------

    # Compute the 2D FFT of the input image
    f = fft2(input_array)

    # Shift the zero-frequency component back to the center of the spectrum
    fshift = fftshift(f)

    # TEST AREA TEST AREA TEST AREA
    # # ikk=np.where(fshift.real<0.1)
    # # fshift.real[ikk]=0.1#fshift.real
    # fshift.real[0:55,:]=0.001
    # fshift.real[:,0:55]=0.001
    # fshift.real[200:256,:]=0.001
    # fshift.real[:,200:256]=0.001
    
    # TEST AREA TEST AREA TEST AREA
    ###################################
    ###################################
    # ADD GAUSSIAN FIT 2D HERE BELOW
    ###################################
    # Fit with 2D Gaussian
#    stop()
#    Gauss_2D_fit_img = fit_2d_gaussian(np.abs(fshift.real), show_imgs=True)
#    fshift.real = np.abs(Gauss_2D_fit_img) - np.abs(fshift.real)
    ###################################
    ###################################
#    stop()

    # Compute the magnitude spectrum of the shifted FFT
    magnitude_spectrum = np.log10(np.abs(fshift))


    if show_imgs==True:
        # Display the Fourier transform of the image
        plt.imshow(magnitude_spectrum, cmap='Greys')
        plt.title('Magnitude Spectrum')
        plt.gca().invert_yaxis()
        #plt.set_title('Input image Spectrum')
        plt.show()
    print('-------------------------------------------')

    print('Magnitude spectrum minimum, median, maximum')
    print(np.min(magnitude_spectrum))
    print(np.median(magnitude_spectrum))
    print(np.max(magnitude_spectrum))
    print('-------------------------------------------')

    ####################################################
    if save_spec_to_file==True:
        # Save image of the spectrum into a fits file
        hdu1E = fits.PrimaryHDU(data=magnitude_spectrum)
        hdu1E.writeto(filename, overwrite=True)
    ####################################################

    x_size = magnitude_spectrum.shape[0]
    y_size = magnitude_spectrum.shape[1]
    mean_size=(x_size+y_size) // 2

    use_central_rect='no'
    if use_central_rect=='yes':
    # Calculate the size of the central rectangle
        rect_width  = x_size // 5 # 51 # x_size // 20
        rect_height = y_size // 5 # 51 # y_size // 20
        # Select the central rectangle
        central_rect = magnitude_spectrum[(x_size // 2) - (rect_width // 2) : (x_size // 2) + (rect_width // 2),
                                          (y_size // 2) - (rect_height // 2) : (y_size // 2) + (rect_height // 2)]

        low_M_thresh = np.median(central_rect)


    # minimum scale below which we want to detect structures/variations:
    min_scale_struct=np.max( (SMALL_SCALE_THR*1., 0) )
    max_scale_struct=(LARGE_SCALE_THR*1.)

    ####################################################
    # Upper Magnitude threshold
    up_M_thresh = np.max(magnitude_spectrum)
    ####################################################
    # Lower Frequency threshold
    low_F_thresh= mean_size // max_scale_struct  # ((mean_size // 2) - FWHM_PIX*5.) // 2 # ~ 56
    # Upper Frequency threshold
    up_F_thresh=mean_size // min_scale_struct
    ####################################################

    # Get the shape of the magnitude spectrum
    M, N = magnitude_spectrum.shape
    # Compute the distance from the center of the magnitude spectrum to each pixel
    X, Y = np.meshgrid(np.arange(N) - N//2, np.arange(M) - M//2)
    distance = np.sqrt(X**2 + Y**2)

    #-------------------------------------------------------------
    # Set the lower and upper thresholds for mag and freq
    # THIS SOLUTION WORKS WITH POINT-LIKE SOURCES
    if use_central_rect=='no':
        low_magnitude_threshold  = np.median(magnitude_spectrum) # low_M_thresh # 
    if use_central_rect=='yes':
        low_magnitude_threshold  = low_M_thresh
    high_magnitude_threshold = up_M_thresh
    low_frequency_threshold  = low_F_thresh
    high_frequency_threshold = up_F_thresh
    #-------------------------------------------------------------

    print('-------------------------------------------')
    print('Magnitude limits set')
    print(low_magnitude_threshold)
    print(high_magnitude_threshold)
    print('Frequency limits set')
    print(low_frequency_threshold)
    print(high_frequency_threshold)
    print('-------------------------------------------')

    # Create masks for frequencies and magnitudes that fall outside the thresholds
    # Frequencies and magnitudes that should NOT be considered
    frequency_mask = np.logical_or(distance < low_frequency_threshold, distance > high_frequency_threshold)
    magnitude_mask = np.logical_or(magnitude_spectrum < low_magnitude_threshold, magnitude_spectrum > high_magnitude_threshold)

    # Combine the frequency and magnitude masks
    mask = np.logical_or(frequency_mask, magnitude_mask)
    # Apply the mask to the shifted FFT
    fshift_masked = fshift.copy()
    fshift_masked[mask] = 0
    # Shift the masked FFT back to the original position
    f_masked = fftshift(fshift_masked)
    # Compute the inverse FFT of the masked FFT
    img_masked = np.real(ifft2(f_masked))

    if show_imgs==True:
        # Display the input image and the masked image side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
        ax1.imshow(input_array, cmap='gray')
        ax1.set_title('Input image')
        ax2.imshow(img_masked, cmap='gray')
        ax2.set_title('Masked image')
        plt.show()

    return img_masked


def get_hf_patterns(input_array, FWHM_PIX, show_imgs=False, save_spec_to_file=False,filename='magnitude_spectrum.fits'):

    # Returns the high frequency patterns from an image.
    # The spatial scale below which the patterns are selected 
    # is defined by FWHM_PIX, corresponding to the Full Width
    # Half Maximum (in pixels) of the image considered)
    # The output is returned as an image with same shape as 
    # the input image
    # Note this function requires "get_filtered_img()", defined above
    
    LARGE_SCALE_THR=FWHM_PIX
    SMALL_SCALE_THR=0

    hf_patterns=get_filtered_img(input_array,SMALL_SCALE_THR,LARGE_SCALE_THR, show_imgs=show_imgs, save_spec_to_file=save_spec_to_file, filename=filename)

    return hf_patterns


def fill_image_holes(image, hole_mask, patch_scale=2, n_pix_contour=2):

    # Fills the "holes" in an input image (numpy array) with patches copied 
    # from the neighbooring areas. The area with the highest likelihood is
    # selected. The likelihood is computed using the difference between the 
    # non-masked pixels in the pacth and the pixels in the original image.
    # The function creates a transition effect between the original image and the 
    # patch, so that the borders of the patch (or the borders of the "holes")
    # are not visible at the end of the process.
    # The "holes" are identified by a boolean mask provided by the user (hole_mask)
    # pixels set to 1 in the hole_mask will be replaced with a patch.
    # The size of the patch is set using "patch_scale". By default, the size is 
    # set to twice the size of the "holes".
    # Using "n_pix_contour", a certain number of additional pixels can be considered
    # around each hole identified by the mask. This corresponds to increase the area 
    # of the regions (holes) initially masked.
    # The function can be run iteratively, by identifying residual pixels that should 
    # be masked (pixels set to 0 in the output image).
    # Requires:
    # import cv2
    # from scipy.ndimage import binary_dilation # expand a boolean mask of a few pixels

    hole_mask=hole_mask.astype(bool)
            
    #----------------------------------------------------------
    # Set the number of pixels to add around each hole
    n_pix_contour = 2
    # Perform binary dilation on the mask
    if n_pix_contour>0:
        dilated_mask = binary_dilation(hole_mask, iterations=n_pix_contour)
        hole_mask=np.copy(dilated_mask)
    #----------------------------------------------------------
    # set masked part of the image to 0
    image[hole_mask==1]=0.
    #----------------------------------------------------------

    image_out = np.copy(image)

    # Find connected components in the hole mask
    _, labels = cv2.connectedComponents(hole_mask.astype(np.uint8))

    # Iterate over each connected component (hole)
    for hole_label in range(1, np.max(labels) + 1):
        # Find pixels belonging to the current hole
        hole_pixels = np.where(labels == hole_label)
                
        # Calculate hole center and bounding box coordinates
        y_center = int(np.mean(hole_pixels[0]))
        x_center = int(np.mean(hole_pixels[1]))
        min_y, max_y = np.min(hole_pixels[0]), np.max(hole_pixels[0])
        min_x, max_x = np.min(hole_pixels[1]), np.max(hole_pixels[1])
                
        # Compute the patch size based on the hole dimensions and scale factor
        patch_height = int((max_y - min_y + 1) * patch_scale)
        patch_width = int((max_x - min_x + 1) * patch_scale)
        if patch_height <=5 : patch_height=5
        if patch_width <=5 : patch_width=5
                
        # Calculate the patch coordinates
        patch_min_y = max(0, y_center - patch_height // 2)
        patch_max_y = min(image.shape[0], y_center + patch_height // 2)
        patch_min_x = max(0, x_center - patch_width // 2)
        patch_max_x = min(image.shape[1], x_center + patch_width // 2)

        size_x=patch_max_x-patch_min_x
        size_y=patch_max_y-patch_min_y
                
        # Initialize patch
        BCK_IMAGE_STAMP=np.copy(image_out[patch_min_y:patch_max_y, patch_min_x:patch_max_x])
        MASK=np.copy(hole_mask[patch_min_y:patch_max_y, patch_min_x:patch_max_x].astype(bool))
        PATCH=np.copy(BCK_IMAGE_STAMP)          
        update_patch=0

        # Initialize reference RMS
        ref_RMS=10.*np.sum(image**2)

        for i in range(3):
            for j in range(3):
                ########################################################################################
                # Tweakreging around the i,j position
                for i_tweak in range(10):
                    for j_tweak in range(10):
                        patch_corr_min_y=(i_tweak-5)+patch_min_y+(i-1)*min(patch_height,(patch_max_y-patch_min_y))
                        patch_corr_max_y=(i_tweak-5)+patch_corr_min_y+min(patch_height,(patch_max_y-patch_min_y))
                        patch_corr_min_x=(j_tweak-5)+patch_min_x+(j-1)*min(patch_width,(patch_max_x-patch_min_x))
                        patch_corr_max_x=(j_tweak-5)+patch_corr_min_x+min(patch_width,(patch_max_x-patch_min_x))
                        
                        size_corr_x=patch_corr_max_x-patch_corr_min_x
                        size_corr_y=patch_corr_max_y-patch_corr_min_y
                                
                        if patch_corr_min_y >= 0 and patch_corr_max_y<image.shape[0] and patch_corr_min_x >= 0 and patch_corr_max_x<image.shape[1] and size_x==size_corr_x and size_y==size_corr_y and not( abs(patch_corr_min_y-patch_min_y) < size_corr_y  and abs(patch_corr_min_x-patch_min_x) < size_corr_x  ) :
                            diff=image[patch_min_y:patch_max_y, patch_min_x:patch_max_x]-image[patch_corr_min_y:patch_corr_max_y, patch_corr_min_x:patch_corr_max_x]
                            mask_diff=hole_mask[patch_min_y:patch_max_y, patch_min_x:patch_max_x]
                            #RMS=np.sqrt(np.sum(diff**2)/np.size(diff))
                            RMS=np.sqrt(np.sum(diff[mask_diff==0]**2)/(1.0+np.size(diff[mask_diff==0])))
                            if RMS<ref_RMS and RMS!=0:
                                update_patch=1
                                ref_RMS=RMS
                                #BCK_IMAGE_STAMP=image_out[patch_min_y:patch_max_y, patch_min_x:patch_max_x]
                                PATCH=np.copy(image_out[patch_corr_min_y:patch_corr_max_y, patch_corr_min_x:patch_corr_max_x])
                                #MASK=hole_mask[patch_min_y:patch_max_y, patch_min_x:patch_max_x].astype(bool)
                ########################################################################################
                
        ## Create Transition between image and patch
                                
        # Calculate the blending weights
        weights_patch = np.zeros_like(PATCH)
        weights_image = np.zeros_like(PATCH)
                            
        # Calculate the distance from the center of PATCH
        center_y = PATCH.shape[0] // 2
        center_x = PATCH.shape[1] // 2
        distance_y = np.abs(np.arange(PATCH.shape[0]) - center_y)
        distance_x = np.abs(np.arange(PATCH.shape[1]) - center_x)

        # Calculate the transition factor based on the distance and MASK
        transition_factor_y = distance_y / (PATCH.shape[0] - 1)
        transition_factor_x = distance_x / (PATCH.shape[1] - 1)

        # Resize transition_factor_y to match the shape of PATCH
        transition_factor_y = np.expand_dims(transition_factor_y, axis=1)
        transition_factor_y = np.tile(transition_factor_y, (1, PATCH.shape[1]))

        # Resize transition_factor_x to match the shape of PATCH
        transition_factor_x = np.expand_dims(transition_factor_x, axis=0)
        transition_factor_x = np.tile(transition_factor_x, (PATCH.shape[0], 1))

        # Calculate the transition factor by taking the minimum
        #transition_factor = np.minimum(transition_factor_y, transition_factor_x)
        transition_factor = np.maximum(transition_factor_y, transition_factor_x)
        #------------------------------------------------------------
        transition_factor =transition_factor -np.min(transition_factor)
        transition_factor =transition_factor/np.max(transition_factor)
        #------------------------------------------------------------
        transition_factor = np.clip(transition_factor, 0, 1)

        # noise is attenuated sqrt(2) when the average (weight=0.5) of two
        # uncorrelated images is considered. The following correction factor
        # keeps into account for this effect.
        correction_factor=1.0+(np.sqrt(2)-1)*(2.0*(0.5-abs(transition_factor-0.5)))
                
        weights_image = transition_factor #* MASK
        weights_patch = (1 - transition_factor) #* MASK
        
        # Perform the weighted blending
        PATCHED_IMAGE_STAMP = PATCH * weights_patch + BCK_IMAGE_STAMP * weights_image

        #--------------------------------------------------------------------------------
        # SET masked values to value of the patch (no transition effect, in this case)
        PATCHED_IMAGE_STAMP[MASK==1]=PATCH[MASK==1]
        correction_factor[MASK==1]=1.0
        #--------------------------------------------------------------------------------

        # Perform the weighted blending
        image_out[patch_min_y:patch_max_y, patch_min_x:patch_max_x] = PATCHED_IMAGE_STAMP*correction_factor

        # if show_imgs==True:
        # Display original image and patch side by side
        #fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
        #ax1.imshow(BCK_IMAGE_STAMP, cmap='gray')
        #ax1.set_title('original image stamp')
        #ax2.imshow(PATCH, cmap='gray')
        #ax2.set_title('patch')
        #ax3.imshow(PATCHED_IMAGE_STAMP, cmap='gray')
        #ax3.set_title('patched image')
        #plt.show()
                                
    return image_out


def get_wpeak(xx,yy,fract=0.5):
    # measures the weighted peak of a x, y distribution by considering 
    # all (and only) the x values corresponding to y values higher than
    # y > fract*peak
    # where peak is the maximum of the distribution
    # IT RETURNS:
    # - the value of x corresponding to the weighted peak of the
    #   distribution
    # --------------------------------------------------------------
    PEACK_IDX=np.where(yy>fract*max(yy))
    PEACK_IDX=PEACK_IDX[0]
    X_PEACK=np.sum(xx[PEACK_IDX]*yy[PEACK_IDX])/np.sum(yy[PEACK_IDX])
    return X_PEACK


def normalize_distribution2(data_norm, data_ref, perc1=16, perc2=84, recenter='Peak', precision=100): 
    # Normalizes and recenter the distribution of the values in a
    # numpy 2D array (data_norm) using a reference 2D array (data_ref)
    # as a reference. Even if the images look different, the distribution 
    # of values in them will be similar at the end of the process.
    # ------------------------------------------------------------
    # After an initial width normalization (performed using  
    # percentiles perc1 and perc2) and recentering of the 
    # distribution's peak, the function goes through an iterative 
    # procedure where the percentiles of the distribution are shifted
    # one by one to recreate the reference distribution
    # Available options:
    # precision=100 Set this value to the number of percentiles that 
    #               should be considered in the fine-tuning process.
    #               The higher the precision, the slower the process
    #               (The computational time scales linearly with this 
    #               parameter)
    # recenter='Peak' Initial recentering is performed using the peak
    #                   of the reference distribution as a reference
    #                   This option works better with Gaussian-like 
    #                   distributions
    # recenter='Median' Initial recentering is performed using the median
    #                   of the reference distribution as a reference.
    #                   This option works better with a general distributions
    ############################################################

    #-----------------------------------------
    # INITIAL NORMALIZATION AND RECENTERING
    #-----------------------------------------
    perc_min=10#2.5
    perc_max=90#97.5
    xhist_ref, yhist_ref, bin_width_ref, minmax_hs_ref,perc_val_ref = get_histogram(data_ref,perc_min,perc_max,SHOW_PLOT=False, get_perc=[perc1, perc2])
    xhist_norm, yhist_norm, bin_width_norm, minmax_hs_norm,perc_val_norm = get_histogram(data_norm,perc_min,perc_max,SHOW_PLOT=False, get_perc=[perc1, perc2])
        
    PERC_DIFF_ref=perc_val_ref[1]-perc_val_ref[0]
    PERC_DIFF_norm=perc_val_norm[1]-perc_val_norm[0]
        
    #-----------------------------------------
    # Get weighted peaks (x values) of the distributions
    #-----------------------------------------
    if recenter=='Peak':
        X_PEAK_ref=get_wpeak(xhist_ref,yhist_ref,fract=0.5)
        X_PEAK_norm=get_wpeak(xhist_norm,yhist_norm,fract=0.5)

    if recenter=='Median':
        X_PEAK_ref=np.median(data_ref)
        X_PEAK_norm=np.median(data_norm)

    # PRE-NORMALIZATION:
    data_norm=(((data_norm-X_PEAK_norm)/PERC_DIFF_norm)*PERC_DIFF_ref)+X_PEAK_ref

    #-----------------------------------------
    # PERCENTILES FINE-TUNING
    #-----------------------------------------
    tmp1=np.copy(data_norm)
    tmp1[:]=0
    #VECT_PERC=np.arange(101)
    VECT_PERC=(np.arange(precision+1)/float(precision))*100. 
    X_NORM_perc=np.percentile(data_norm,VECT_PERC)
    X_REF_perc=np.percentile(data_ref,VECT_PERC)
    max_perc=np.size(VECT_PERC)-1
    for ii in range(max_perc):
        perc_NORM_IDX=np.where((data_norm > X_NORM_perc[ii]) & (data_norm <= X_NORM_perc[ii+1]))
        perc_REF_IDX=np.where((data_ref > X_REF_perc[ii]) & (data_ref <= X_REF_perc[ii+1]))
        perc_NORM_mean=np.mean(data_norm[perc_NORM_IDX])
        perc_REF_mean=np.mean(data_ref[perc_REF_IDX])
        perc_NORM_IDX_B=np.where(data_norm >= X_NORM_perc[ii])
        tmp1[perc_NORM_IDX_B]=data_norm[perc_NORM_IDX_B]+(perc_REF_mean-perc_NORM_mean)
        # TEST:
        #AA, BB, CC, DD,EE = get_histogram(tmp1,perc_min,perc_max,SHOW_PLOT=False, get_perc=[perc1, perc2])
    data_norm=tmp1
        
    return data_norm


def plot_compare_distrib(array1, array2, array3=[], label1='Array1', label2='Array2', label3=None, xlabel='Values', ylabel='Quantity', title='Array1 Vs Array2', norm=None, ylog='no'):
    # given two (or threee) 2D arrays, it compares the distributions of 
    # their values.
    # Using the norm keyword, the user can chose the preferred way 
    # to compare the distributions
    # - norm=None # Default option: the distributions are not normalized
    # - norm=area # the distributions are normalized by their area
    # - norm=peak # the distributions are normalized by their peak
    #-----------------------------------------------------------
    perc_min=2.5
    perc_max=97.5
    xhist_1, yhist_1, bin_width_1, minmax_1 = get_histogram(array1,perc_min,perc_max,SHOW_PLOT=False)
    xhist_2, yhist_2, bin_width_2, minmax_2 = get_histogram(array2,perc_min,perc_max,SHOW_PLOT=False)
    if np.size(array3)!=0:
        xhist_3, yhist_3, bin_width_3, minmax_3 = get_histogram(array3,perc_min,perc_max,SHOW_PLOT=False)

    norm_val_1=1.
    norm_val_2=1.
    norm_val_3=1.
            
    if norm=='area':
        norm_val_1=np.sum(yhist_1*bin_width_1)
        norm_val_2=np.sum(yhist_2*bin_width_2)
        if np.size(array3)!=0:
            norm_val_3=np.sum(yhist_3*bin_width_3)

    if norm=='peak':
        norm_val_1=np.max(yhist_1)
        norm_val_2=np.max(yhist_2)
        if np.size(array3)!=0:
            norm_val_3=np.max(yhist_3)

    # Plot - histogram of image/array 1
    plt.plot(xhist_1, yhist_1/norm_val_1,color='blue', label=label1)
    # Plot - histogram of image/array 2
    plt.plot(xhist_2, yhist_2/norm_val_2,color='red', label=label2)
    if np.size(array3)!=0:
        plt.plot(xhist_3, yhist_3/norm_val_3,color='green', label=label3)
        
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel) 
    if ylog=='yes':
        plt.yscale('log')
    plt.legend()
    plt.show()
    return None



def main():

    #########################################################################
    # INITIAL TESTS ON CALIBRATOR IMAGES:
    # EXAMPLE:
    # python noisempire4.py default_config.txt TEST_imgs/uid___A002_Xef4e78_X19df.ms.split.cal.J0215-0222_B3.pseudocont.sc.fits
    #
    #  1) Simple case (no background, no noise patterns, ~circular PSF)
    # uid___A002_Xb5bd46_X5dc8.ms.split.cal.J1126-3828_B3.pseudocont.sc.fits
    #  2) small scale noise structuration well reproduced
    # uid___A002_Xb00171_X1904.ms.split.cal.J0438+3004_B3.pseudocont.sc.fits
    #  3) presence of large scale background 
    # uid___A002_Xb88fca_X7028.ms.split.cal.J1319-1217_B3.pseudocont.sc.fits
    #  4A) Some non-Gaussian, non-random noise where the IDL algorithm failed
    #    High frequency patterns are correctly replicated 
    # uid___A002_Xb5aa7c_X2937.ms.split.cal.J1041+0610_B3.pseudocont.sc.fits
    #  4B) High frequency patterns are correctly replicated - similar to 4A 
    # uid___A002_Xc8ed16_X9686.ms.split.cal.J0442-0017_B3.pseudocont.sc.fits
    #  5) High frequency patterns/fringing + DOMINANT large scale background
    # uid___A002_Xc3a8fe_X280b.ms.J1229+0203_B3.pseudocont.sc.fits
    #  6) High frequency patterns/fringing + non point-like source
    # uid___A002_Xef4e78_X19df.ms.split.cal.J0215-0222_B3.pseudocont.sc.fits

    # NEW TESTS ************************************************************

    # 7) background structuration and HF patterns --> It works very well here
    # uid___A002_X872bbc_X1747.ms.split.cal.J1215-1731_B3.pseudocont.sc.fits
    # 8A) Radial structuration --> NOT well reproduced !
    # uid___A002_Xa7b91c_X17a1.ms.split.cal.J0334-4008_B3.pseudocont.sc.fits
    # 8B) Radial structuration --> VERY BAD:
    # uid___A002_X99c183_X332b.ms.split.cal.J1427-4206_B3.pseudocont.sc.fits

    #########################################################################

    ## SIMULATE NOISE + REAL SKY
    # Case 6
    # python noisempire.py  default_config.txt --INPUT_IMAGE=TEST_imgs/uid___A002_Xef4e78_X19df.ms.split.cal.J0215-0222_B3.pseudocont.sc.fits --REAL_SKY_IMAGE=/home/baronchelli/cartella_lavoro/BRAIN_Study/Noise_sim/Test_imgs_From_Michele/test4/gaussian_simulations/clean_cube_6.fits
    ##

    # --------------------------/
    # -------------------------/
    # PARAMETERS              |
    # -------------------------\
    # --------------------------\
    
    Prog_Descript_Text='''
-------------------------------------------------
 Run noisempire.py using the following syntax:
-------------------------------------------------
 Configuration 1
 All the parameters are specified in the configuration file:
 $ python noisempire.py config_file.txt
-------------------------------------------------
 Configuration 2
 Some parameters are specified in the configuration file, 
 while others are specified in the command line. Example:
 $ python noisempire.py config_file.txt --INPUT_IMAGE=input_image.fits --REAL_SKY_IMAGE=real_sky.fits --DEBUG=False
-------------------------------------------------
    '''

    parser = argparse.ArgumentParser(description=Prog_Descript_Text, formatter_class=argparse.RawTextHelpFormatter)

    # POSITIONAL ARGUMENTS
    parser.add_argument('CONFIG_FILE', help='Configuration file /path/name')
    # OPTIONAL ARGUMENTS
    parser.add_argument('--INPUT_IMAGE',    help='Input image (/path/name)')
    parser.add_argument('--WORKDIR',        help='Working directory. Simulated images, temporary files \n and other stuff is put here (/path/dirname) ')
    parser.add_argument('--SHOW_PLOT',      help='Show useful plots during run (True or False)')
    parser.add_argument('--DEBUG',          help='Produce additional images for debugging purposes (True or False)')
    parser.add_argument('--REAL_SKY_IMAGE', help='Real sky image (/path/name or None)')
    parser.add_argument('--NSIGMA_RSKY',    help='Approximate significance of the detection in sigma')
    parser.add_argument('--REP_STRIP_LENGTH',help='Scale length over which the radial and/or elliptical \n patterns are computed. Measured in units of image (maximum) FWHM')
    parser.add_argument('--REP_THRESH',     help='Patterns are considered as existing only where the \n pixel value inside a strip is larger than REP_THRESH \n times the sigma computed over an isotropic (square) \n with similar area centered in the same position')
    parser.add_argument('--IMG_RAD_PATT',   help='Save radial patterns to a .fits image (True/False)')
    parser.add_argument('--RAD_PATT_ISOLATE',help='Isolate and remove the radial patterns component from \n the original image. They are added back into the simulated image (True or False)')
    parser.add_argument('--IMG_ELL_PATT',   help='Save elliptical (PSF-shaped) patterns to a .fits image \n (True/False)')
    parser.add_argument('--ELL_PATT_ISOLATE',help='Isolate and remove the elliptical (PSF_shaped) patterns \n component from the original image. They are added back \n into the simulated image (True or False)')
    parser.add_argument('--REM_REP',        help='Identify and remove radial-elliptical patterns (True or False)')
    parser.add_argument('--NOISE_CONV',     help='Convolve pixel-scale noise with PSF, local ACF, Global ACF or \n ACF and PSF combined (PSF/ACF/local_ACF/PSF_&_local_ACF)')
    parser.add_argument('--BCK1_TYPE',      help='keep the original Background measured on Larger scales or \n simulate it (Original or Simulate)')
    parser.add_argument('--BCK2_TYPE',      help='keep the original Background measured on small scale or \n simulate it (Original or Simulate)')
    parser.add_argument('--SEX_FOLDER',     help='SExtractor configuration files folder (/path/folder_name) ')
    parser.add_argument('--HFN_SCALE',      help='Maximum scale at which the high frequency patters are isolated.\n Measured in number of FWHM. Reasonable value should be 1.0')
#    parser.add_argument('--version', action='version', version='%(prog)s ' + Version)
    parser.add_argument('--version', action='version', version=Name+' '+ Version+'\n'+Developer+' '+ Years)

    args = parser.parse_args()


    print("----------------------------------------")
    if len(sys.argv) == 1:
        print("ERROR:")
        print("A configuration file must be provided!")
        print("Use the followig sintax:")
        print("python noisempire.py config_file.txt")
#        print("or:")
#        print("python noisempire.py config_file.txt input_image.fits")
        sys.exit(1)

    # Configuration file
    if len(sys.argv) >= 2:
        print("Configuration file: " + sys.argv[1] )
        if not(os.path.isfile(sys.argv[1])):
            print("NOT FOUND")
            sys.exit(1) 

    print("----------------------------------------")

    ###################################################
    # Read configuration file
    ###################################################
    #config_file=Readcol(sys.argv[1],'a,a',skipline=0,skipcol=0,string_length=200)
    config_file=Readcol(args.CONFIG_FILE,'a,a',skipline=0,skipcol=0,string_length=200)
    PARAMS=np.char.upper(config_file.col(0)) # ALl upper case
    VALUES=config_file.col(1)
    ############################################

    ##########################################################
    # PARAMETERS FROM CONFIG FILE
    ##########################################################

    #-------------------------------------------------------
    # Retrieve parameters from the configuration file. 
    #-------------------------------------------------------
    # Any parameter specified in the command line will override those from the configuration file."
    # Input image
    INPUT_IMAGE=get_param_value(PARAMS=PARAMS, VALUES=VALUES, param_name='INPUT_IMAGE', convert=True)

    # Input image
    WORKDIR=get_param_value(PARAMS=PARAMS, VALUES=VALUES, param_name='WORKDIR', convert=True)

    # See plots during processing
    SHOW_PLOT=get_param_value(PARAMS=PARAMS, VALUES=VALUES, param_name='SHOW_PLOT', convert=True)

    # Produces additional images for debugging purposes
    DEBUG=get_param_value(PARAMS=PARAMS, VALUES=VALUES, param_name='DEBUG', convert=True)

    # Real sky image
    REAL_SKY_IMAGE=get_param_value(PARAMS=PARAMS, VALUES=VALUES, param_name='REAL_SKY_IMAGE', convert=True)

    # Approximate significance of the detection
    NSIGMA_RSKY=get_param_value(PARAMS=PARAMS, VALUES=VALUES, param_name='NSIGMA_RSKY', convert=True)

    #Scale length over which the radial and/or elliptical patterns are computed.
    REP_STRIP_LENGTH=get_param_value(PARAMS=PARAMS, VALUES=VALUES, param_name='REP_STRIP_LENGTH', convert=True)

    # Patterns detection threshold
    REP_THRESH=get_param_value(PARAMS=PARAMS, VALUES=VALUES, param_name='REP_THRESH', convert=True)

    # Radial patterns image True/False
    IMG_RAD_PATT=get_param_value(PARAMS=PARAMS, VALUES=VALUES, param_name='IMG_RAD_PATT', convert=True)
    
    # Isolate and remove radial patterns from original image before noise simulation and add them back at the end to simulated image
    RAD_PATT_ISOLATE=get_param_value(PARAMS=PARAMS, VALUES=VALUES, param_name='RAD_PATT_ISOLATE', convert=True)

    # Elliptical patterns image True/False
    IMG_ELL_PATT=get_param_value(PARAMS=PARAMS, VALUES=VALUES, param_name='IMG_ELL_PATT', convert=True)

    # Isolate and remove elliptical patterns from original image before noise simulation and add them back at the end to simulated image
    ELL_PATT_ISOLATE=get_param_value(PARAMS=PARAMS, VALUES=VALUES, param_name='ELL_PATT_ISOLATE', convert=True)
    # Convolve pixel-scale noise with PSF, local ACF, Global ACF or ACF and PSF combined
    NOISE_CONV=get_param_value(PARAMS=PARAMS, VALUES=VALUES, param_name='NOISE_CONV', convert=True)

    # Background Large scale simulated / original? 
    BCK1_TYPE=get_param_value(PARAMS=PARAMS, VALUES=VALUES, param_name='BCK1_TYPE', convert=True)

    # Background small scale simulated / original? 
    BCK2_TYPE=get_param_value(PARAMS=PARAMS, VALUES=VALUES, param_name='BCK2_TYPE', convert=True)

    # SExtractor configuration files folder
    SEX_FOLDER=get_param_value(PARAMS=PARAMS, VALUES=VALUES, param_name='SEX_FOLDER', convert=True)

    # Maximum scale at which the high frequency patters are isolated
    # Measured in number of FWHM. Default should be 1.0
    HFN_SCALE=get_param_value(PARAMS=PARAMS, VALUES=VALUES, param_name='HFN_SCALE', convert=True)



    ##########################################################
    # PARAMETERS FROM CONFIG FILE - END
    ##########################################################


    ##########################################################
    # PARAMETERS FROM COMMAND LINE 
    ##########################################################

    if args.INPUT_IMAGE:
        INPUT_IMAGE=get_type_and_convert(args.INPUT_IMAGE)
    if args.WORKDIR:
        WORKDIR=get_type_and_convert(args.WORKDIR)
    if args.SHOW_PLOT:
        SHOW_PLOT=get_type_and_convert(args.SHOW_PLOT)
    if args.DEBUG:
        DEBUG=get_type_and_convert(args.DEBUG)
    #------------------------------------------------------------
    if args.REAL_SKY_IMAGE:
        REAL_SKY_IMAGE=get_type_and_convert(args.REAL_SKY_IMAGE)
    if args.NSIGMA_RSKY:
        NSIGMA_RSKY=get_type_and_convert(args.NSIGMA_RSKY)
    #------------------------------------------------------------
    if args.REP_STRIP_LENGTH:
        REP_STRIP_LENGTH=get_type_and_convert(args.REP_STRIP_LENGTH)
    if args.REP_THRESH:
        REP_THRESH=get_type_and_convert(args.REP_THRESH)
    if args.IMG_RAD_PATT:
        MG_RAD_PATT=get_type_and_convert(args.IMG_RAD_PATT)   
    if args.RAD_PATT_ISOLATE:
        RAD_PATT_ISOLATE=get_type_and_convert(args.RAD_PATT_ISOLATE)
    if args.IMG_ELL_PATT:
        IMG_ELL_PATT=get_type_and_convert(args.IMG_ELL_PATT)
    if args.ELL_PATT_ISOLATE:
        ELL_PATT_ISOLATE=get_type_and_convert(args.ELL_PATT_ISOLATE)
    #------------------------------------------------------------
    if args.NOISE_CONV:
        NOISE_CONV=get_type_and_convert(args.NOISE_CONV)
    if args.BCK1_TYPE:
        BCK1_TYPE=get_type_and_convert(args.BCK1_TYPE)
    if args.BCK2_TYPE:
        BCK2_TYPE=get_type_and_convert(args.BCK2_TYPE)
    #------------------------------------------------------------
    if args.SEX_FOLDER:
        SEX_FOLDER=get_type_and_convert(args.SEX_FOLDER)
    if args.HFN_SCALE:
        HFN_SCALE=get_type_and_convert(args.HFN_SCALE)


    ##########################################################
    # PARAMETERS FROM COMMAND LINE - END
    ##########################################################


    ###################################################
    # Read reference image 
    ###################################################
    # Check input image existence and read it
    print("-----------------------------")
    print("Input image: " + INPUT_IMAGE )
    if not(os.path.isfile(INPUT_IMAGE)):
        print("NOT FOUND")
        print("-----------------------------")
        sys.exit(1)
    else:
        # get input image name and directory
        directory_ii, original_image_name = os.path.split(INPUT_IMAGE)
    #...............................................
    # Read input (reference) image (data and header)
    #...............................................
    data, header = read_fits_image(INPUT_IMAGE)
    # get image WCS (from astropy.wcs import WCS)
    #wcs=WCS(header)

    # Compute flat reference image (CUBE --> 2D image)
    NAXIS_KEEP=2
    NAXIS_REMOVE=0
    while np.shape(np.shape(data))[0]>NAXIS_KEEP:
        # Keep flattening while the cube becomes a 2D image instead of a cube
        data = np.sum(data, axis=0)
        NAXIS_REMOVE=NAXIS_REMOVE+1

    # Update number of axes in the header
    header["NAXIS"]=2
    # Remove additional axes from header (to prevent errors when writing 2d images)
    while NAXIS_REMOVE>0:
        keyw_rem="NAXIS"+str(NAXIS_REMOVE+2)
        header.pop(keyw_rem, None)
        NAXIS_REMOVE=NAXIS_REMOVE-1

    # get image WCS (from astropy.wcs import WCS)
    wcs=WCS(header)

    # Write flattened data and modified header to a new FITS file
    # NECESSARY TO GET AN APPROPRIATE HEADER FROM SEXTRACTOR
    flat_original_image="FLAT_"+original_image_name
    fits.writeto(WORKDIR+'/'+flat_original_image, data, header, overwrite=True)


    ###################################################


    ###################################################
    # Read Real sky image (if present) 
    ###################################################
    if (REAL_SKY_IMAGE==None or REAL_SKY_IMAGE==' '):
        REAL_SKY_IMAGE=None
        print ('------------------------------------------------------------')
        print ('No real sky image indicated to be added to simulated noise')
        print ('------------------------------------------------------------')
    if REAL_SKY_IMAGE!=None:
        # Check real sky image existence and read it
        print ('------------------------------------------------------------')
        print("Real Sky image: " + REAL_SKY_IMAGE )
        if not(os.path.isfile(REAL_SKY_IMAGE)):
            print("NOT FOUND")
            print("If you don't want to add a real sky to the simulation, set REAL_SKY_IMAGE")
            print("in the configuration file to None or leave the keyword void")
            print ('------------------------------------------------------------')
            sys.exit(1)
        else:
            print ('------------------------------------------------------------')
            # get real sky image name and directory
            # Split the working image file path into the directory and file name
            directory_rsi, real_sky_image_name = os.path.split(REAL_SKY_IMAGE)
            # Read real sky image (data_rsi and header_rsi)
            data_rsi, header_rsi = read_fits_image(REAL_SKY_IMAGE)
            # get image WCS (from astropy.wcs import WCS)
            wcs_rsi=WCS(header_rsi)

            
            # Compute flat real sky image CUBE --> 2D image
            #flat_data_rsi = np.sum(data_rsi, axis=0)
            #flat_data_rsi = np.sum(flat_data_rsi, axis=0)

            # Compute flat real sky image CUBE --> 2D image
            flat_data_rsi = np.copy(data_rsi)
            while np.shape(np.shape(flat_data_rsi))[0]>2:
                # Keep flattening while the real sky is a 2D image instead of a cube
                flat_data_rsi = np.sum(flat_data_rsi, axis=0)

            # # Check flat image
            # flat_image_filename = 'TEST_imgs/flat_working_image.fits'
            # fits.writeto(flat_image_filename, flat_data_rsi, header=header_rsi, overwrite=True)

    ###################################################

    
    orig_image=np.copy(data)


    ###################################################
    # Compute/read relevant quantities from Reference original image 
    ###################################################

    # Get keyword values from the header
    CDELT1 = get_keyword_value(header,"CDELT1")# Pixel scale along x [deg/pixel]
    CDELT2 = get_keyword_value(header,"CDELT2")# Pixel scale along y [deg/pixel]
    BMAJ = get_keyword_value(header, "BMAJ")   # Beam FWHM along major axis [deg]
    BMIN = get_keyword_value(header, "BMIN")   # Beam FWHM along minor axis [deg]
    BPA  = get_keyword_value(header, "BPA")    # Beam position angle [deg]

    # convert keyword values to to appropriate units (header,"CDELT2")# Pixel scale along y [deg]
    pix_scale_x=abs(CDELT1*3600)  # Pixel scale along x [arcsec/pixel]
    pix_scale_y=abs(CDELT2*3600)  # Pixel scale along y [arcsec/pixel]
    pix_scale=(pix_scale_x+pix_scale_y)/2. # Average Pixel scale [arcsec/pixel]
    FWHM_x=BMAJ*3600              # Beam FWHM-x [arcsec]
    FWHM_y=BMIN*3600              # Beam FWHM-y [arcsec]
    FWHM=(FWHM_x+FWHM_y)/2.       # Average FWHM [arcsec]
    FWHM_MAX=max(FWHM_x,FWHM_y)   # Maximum FWHM [arcsec]
    FWHM_PIX_x=FWHM_x/pix_scale_x # Beam FWHM-x [pixels]
    FWHM_PIX_y=FWHM_y/pix_scale_y # Beam FWHM-y [pixels]
    FWHM_PIX=FWHM/pix_scale       # Average Beam FWHM [pixels]
    FWHM_PIX_max=max(FWHM_PIX_x,FWHM_PIX_y)   # Maximum Beam FWHM [pixels]
    P_ANGLE=BPA                   # Beam position angle [deg]

    # print image parameters
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("IMAGE technical details")
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("Pixel scale x [arcsec/pixel]:",pix_scale_x)
    print("Pixel scale y [arcsec/pixel]:",pix_scale_y)
    print("Average Pixel scale [arcsec/pixel]:",pix_scale)
    print("FWHM (x) [arcsec]:",FWHM_x)
    print("FWHM (y) [arcsec]:",FWHM_y)
    print("FWHM (x) [pixels]:",FWHM_PIX_x)
    print("FWHM (y) [pixels]:",FWHM_PIX_y)
    print("Position angle [deg]:",P_ANGLE)
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")


    ###################################################
    # Create void simulated image
    ###################################################
    # create a new image, identical in size, but with all the values replaced by zeroes
    sim_data = np.zeros_like(data)

    ###################################################
    # Compute and save image PSF
    ###################################################
    # Size of the image that will contain the synthetic PSF
    xsize_PSF=max((round(5*FWHM_PIX_max),8)) # =data.shape[2]
    ysize_PSF=max((round(5*FWHM_PIX_max),8)) # =data.shape[3]
    PSF_img=get_gaussian_PSF(xsize_PSF,ysize_PSF,FWHM_PIX_x,FWHM_PIX_y,P_ANGLE)
    if SHOW_PLOT==True:
        plt.imshow(PSF_img, cmap='gray')
        plt.title('Synthesized PSF (Beam)')
        # invert the y-axis (to make it consistent with visualization in ds9)
        plt.gca().invert_yaxis()
        plt.show()
    # SAVE PSF IMAGE TO A FILE
    data_PSF=PSF_img
    hdu_PSF = fits.PrimaryHDU(data_PSF,header) 
    hdu_PSF.header["COMMENT"] = "This is a simulated PSF created using "+ Name+" "+Version+" "+Developer+" "+Years
    hdu_PSF.header["COMMENT"] = "The image from which this PSF is computed is"
    hdu_PSF.header["COMMENT"] = original_image_name
    hdu_PSF.writeto(WORKDIR+'/PSF_sim_'+original_image_name, overwrite=True)




    ##################################################
    # Compute large scale background img (SExtractor)
    ##################################################
    mean_FWHM=np.mean([FWHM_PIX_x,FWHM_PIX_y])
    filter_name=get_filter_name(mean_FWHM) 
    img_bck1='bck1_'+original_image_name
    #-------------------------------------------------
    # SET BACKGROUND SCALE HERE:
    BACK_SIZE_LS=3*FWHM_PIX # 4*FWHM_PIX # increase to increase the scale over which the bck is computed
    BACK_FILTERSIZE=2 #3 (higher values to smooth the background more)
    #-------------------------------------------------
    # Run SExtractor here
    run_sex1(WORKDIR+'/'+flat_original_image,WORKDIR+'/'+img_bck1, SEX_FOLDER,filter_name,pix_scale,FWHM_PIX_x,FWHM_PIX_y,BACK_SIZE_LS,BACK_FILTERSIZE,img_type="BACKGROUND")
    # read bck image for successive computations:
    data_bck1, header_bck1 = read_fits_image(WORKDIR+'/'+img_bck1)
    # get image WCS (from astropy.wcs import WCS)
    wcs_bck1=WCS(header_bck1)
    ##################################################


    ###################################################
    # CONVOLVE LARGE SCALE (LS) BACKGROUND with PSF stamp
    ###################################################
    # Large scale background is convolved with PSF to improve the 
    # "boxy" shape of the background computed by SExtractor and make it
    # more "natural" i.e., similar to the original background pattern. 
    # Convolve bck and PSF (mode=same to obtain the same image size)
    med_bck1_orig=np.median(data_bck1) # median bck value (for successive re-normalization)
    sigma_bck1_orig=np.std(data_bck1)  # sigma of bck (for successive re-normalization)
    data_bck1_orig=np.copy(data_bck1)  # save original SExtractor bck to another array
    # -----------------------------------------------------------------
    # Convolve large scale BCK with PSF
    data_bck1 = scipy.signal.convolve2d(data_bck1,PSF_img,mode='same',boundary='wrap')
    #------------------------------------------------------------------
    if SHOW_PLOT==True:
        plt.imshow(data_bck1, cmap='gray')
        plt.title("Large scale background")
        # invert the y-axis (to make it consistent with visualization in ds9)
        plt.gca().invert_yaxis()
        plt.show()
    #-------------------------------------------------------------
    # re-normalize convoved BCK to its original Flux distribution
    #data_bck1 = (data_bck1/np.median(data_bck1))*  med_bck1_orig
    data_bck1 = ( ( (data_bck1-np.median(data_bck1)) / np.std(data_bck1) )*sigma_bck1_orig ) + med_bck1_orig
    #-------------------------------------------------------------
    # PLOT BCK DISTRIBUTIONS BEFORE AND AFTER PSF CONVOLUTION
    if SHOW_PLOT==True:
        perc_min=2.5
        perc_max=97.5
        # get histogram of original background distribution
        xhist_bck1_orig, yhist_bck1_orig, bin_width_bck1_orig, minmax_hs_bck1_orig= get_histogram(data_bck1_orig,perc_min,perc_max,SHOW_PLOT=False)
        # get histogram of the PSF_convolved background distribution
        xhist_bck1, yhist_bck1, bin_width_bck1, minmax_hs_bck1 = get_histogram(data_bck1,perc_min,perc_max,SHOW_PLOT=False)
        # Plot - histogram of the original background image
        plt.plot(xhist_bck1_orig,yhist_bck1_orig,color='blue', label='Bck from SExtractor')
        # Plot - histogram of the PSF-convolverd background image
        plt.plot(xhist_bck1,yhist_bck1,color='red', label='Bck from SExtractor convolved with PSF')
        plt.title("Large scale background Flux distribution")
        plt.ylabel("Number of  pixels")
        plt.xlabel("Flux [original image units]")
        plt.legend()
        plt.show()
    #-------------------------------------------------------------


    ##################################################
    # Compute LS background - subtracted img
    ##################################################
    #-------------------------------------------------
    data_bck1_sub=data-data_bck1
    header_bck1_sub=header#_bck1
    #-------------------------------------------------
    if SHOW_PLOT==True:
        plt.imshow(np.log(data_bck1_sub-1.01*np.min(data_bck1_sub)), cmap='gray')
        plt.title("log(Image WITHOUT Large scale BCK)")
        # invert the y-axis (to make it consistent with visualization in ds9)
        plt.gca().invert_yaxis()
        plt.show()
        
    # Save bck-sub img to a file
    img_bck1_sub='bck1_sub_'+original_image_name
    fits.writeto(WORKDIR+'/'+img_bck1_sub, data_bck1_sub, header_bck1, overwrite=True)
    ##################################################
        
    
    ###################################################
    # Fast Fourier transform of (image-sources)-bck 
    ###################################################
    opt1='B'
    if opt1=='A':
        # Compute Fourier Transform of the original image (faster)
        #data_to_filter=data 
        data_to_filter=np.imcopy(data_bck1_sub) # Alternative: img-background
    if opt1=='B':
        # Compute Fourier Transform of image-sources
        # Run sextractor to compute the image without sources
        img_nosrcs1='nosrcs_'+original_image_name
        # RUN SEXTRACTOR ON ORIGINAL IMAGE
        #run_sex1(WORKDIR+'/'+original_image_name,WORKDIR+'/'+img_nosrcs1, SEX_FOLDER,filter_name,pix_scale,FWHM_PIX_x,FWHM_PIX_y,BACK_SIZE,BACK_FILTERSIZE,img_type="-OBJECTS")

        # RUN SEXTRACTOR ON BACKGROUND SUBTRACTED IMAGE

        SExtractor_BUG='yes'
        #--------------------------------------------------------------------------------
        # BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG
        #--------------------------------------------------------------------------------
        # BUG: Source Extractor (version 2.25.0 (2018-02-08) doesn't seem to take into 
        #      account the DETECT_MINAREA parameter when computing the
        #      CHECKIMAGE "-OBJECTS" (image with no sources). This is not the case
        #      when computing, instead, the CHECKIMAGE "OBJECTS".
        #--------------------------------------------------------------------------------
        # BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG
        #--------------------------------------------------------------------------------
        if SExtractor_BUG=='no':
            print('WARNING: considering the -OBJECT image directly, in place of ')
            print('         the OBJECT option, could make the script faster.')
            print('         However, currently SExtractor does not correctly handle')
            print('         the "-OBJECT" option (and in particular, it does not ')
            print('         correctly take into account the parameter DETEC_MINAREA')
            print('         when computing the -OBJECT image).')
            print('         When this bug will be fixed, this section could be ')
            print('         rewritten to make the script more efficient.')
            print('         ------------------------------------------------------')
            print('             MODIFYING THE SCRIPT IS NOT NECESSARY!')
            print('         ------------------------------------------------------')

        if SExtractor_BUG=='yes':
            
            #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
            # POINT LIKE SOURCES
            #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
            # higher threshold in sigma, smaller minimum area [pixels] for detection
            img_srcs_only1='srcs_only1_'+original_image_name
            run_sex1(WORKDIR+'/'+img_bck1_sub,WORKDIR+'/'+img_srcs_only1, SEX_FOLDER,filter_name,pix_scale,FWHM_PIX_x,FWHM_PIX_y,BACK_SIZE_LS,BACK_FILTERSIZE,img_type="OBJECTS", DETECT_THRESH=4.0, ANALYSIS_THRESH=4.0,DETECT_MINAREA_BEAM=1)
            data_srcs_ony1, header_srcs_only1 = read_fits_image(WORKDIR+'/'+img_srcs_only1)
            no0idx1=np.where(data_srcs_ony1 != 0)

            #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
            # EXTENDED SOURCES
            #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
            # lower threshold in sigma, larger minimum area [pixels] for detection
            img_srcs_only2='srcs_only2_'+original_image_name

            run_sex1(WORKDIR+'/'+img_bck1_sub,WORKDIR+'/'+img_srcs_only2, SEX_FOLDER,filter_name,pix_scale,FWHM_PIX_x,FWHM_PIX_y,BACK_SIZE_LS,BACK_FILTERSIZE,img_type="OBJECTS", DETECT_THRESH=1.25, ANALYSIS_THRESH=1.25,DETECT_MINAREA_BEAM=10)
            data_srcs_ony2, header_srcs_only2 = read_fits_image(WORKDIR+'/'+img_srcs_only2)

            no0idx2=np.where(data_srcs_ony2 != 0)

            #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

            # Bi-dimensional image
            data_nosrcs_A= np.copy(data_bck1_sub)
            data_nosrcs_A[no0idx1]= 0 # Point like sources
            data_nosrcs_A[no0idx2]= 0 # extended sources

            # data cube
            data_nosrcs_B=np.copy(data_bck1_sub)
            data_nosrcs_B=data_nosrcs_A
            
            header_nosrcs_B = header_srcs_only1
            fits.writeto(WORKDIR+'/'+img_nosrcs1, data_nosrcs_B, header_nosrcs_B, overwrite=True)


            #######################

#        #---------------------------------------------------------------
#        # Create a binary mask indicating regions to fill
#        mask = np.zeros_like(data_nosrcs_B)
#        mask[data_nosrcs_B == 0] = 1 #  zero values in the image will be filled
#        # Fill zero regions with synthetic noise
#        #---------------------------------------------------------------


        filled_image=np.copy(data_nosrcs_B) # img without sources
        #filled_image=np.copy(data_nosrcs_B-data_bck1)# Alternative: img-background
        n_max_iter=5
        for tt in range(n_max_iter):
            mask = np.zeros_like(filled_image) # Create a binary mask indicating regions to fill
            mask[filled_image == 0] = 1 #  zero values in the image will be filled
            filled_image[mask==1]=1000. # TEST TEST TEST 
            # Fill zero regions with synthetic noise
            filled_image=fill_image_holes(filled_image, mask, patch_scale=2,n_pix_contour=2 )
            if np.size(np.where(mask==1))==0 : tt=n_max_iter

        data_to_filter=np.copy(filled_image)

        # Save image with no sources and patches replacing them to a file
        img_nosrcs_patch='nosrcs_patch_'+original_image_name
        img_nosrcs_patch_save=np.copy(data)
        img_nosrcs_patch_save=np.copy(data_to_filter)
        fits.writeto(WORKDIR+'/'+img_nosrcs_patch, img_nosrcs_patch_save, header, overwrite=True)

        # Save image with no sources and patches + bck to a file only for debugging purposes
        if DEBUG==True:
            img_nosrcs_patch_bck='nosrcs_patch_bck_'+original_image_name
            img_nosrcs_patch_bck_save=np.copy(data)
            img_nosrcs_patch_bck_save=np.copy(data_to_filter)
            img_nosrcs_patch_bck_save=img_nosrcs_patch_bck_save+data_bck1
            fits.writeto(WORKDIR+'/'+img_nosrcs_patch_bck, img_nosrcs_patch_bck_save, header, overwrite=True)


    ###################################################
    # COMPUTE High Frequency patterns (HFN = High Frequency Noise)
    ###################################################
    img_filtered=get_hf_patterns(data_to_filter,HFN_SCALE*FWHM_PIX, show_imgs=SHOW_PLOT, save_spec_to_file=DEBUG, filename='TEST_imgs/magnitude_spectrum.fits')

    # Save high frequency patterns to a file 
    img_HFN='HFN_'+original_image_name
    img_filtered_save=np.copy(data)
    img_filtered_save=np.copy(img_filtered)
    fits.writeto(WORKDIR+'/'+img_HFN, img_filtered_save, header, overwrite=True)


    ##################################################
    # Compute [ background_L & HF pattern ] subtracted img
    ##################################################
    #-------------------------------------------------
    data_bck1_HF_sub=data_bck1_sub-img_filtered
    header_bck1_HF_sub=header
    #-------------------------------------------------
    # Save [bck+HF pattern]-subtracted img to a file 
    img_bck1_HF_sub='bck1_HF_sub_'+original_image_name
    fits.writeto(WORKDIR+'/'+img_bck1_HF_sub, data_bck1_HF_sub, header_bck1_HF_sub, overwrite=True)
    ##################################################


    ##################################################
    # Compute [ background, sources & HF pattern ] subtracted img
    # SOURCES REPLACED BY PATCHES!
    ##################################################
    #-------------------------------------------------
    data_nosrcs_bck1_HF_sub=img_nosrcs_patch_save-img_filtered_save
    header_nosrcs_bck1_HF_sub=header
    #-------------------------------------------------
    # Save [bck + sources + HF pattern] subtracted img to a file 
    img_nosrcs_bck1_HF_sub='nosrcs_bck1_HF_sub_'+original_image_name
    fits.writeto(WORKDIR+'/'+img_nosrcs_bck1_HF_sub, data_nosrcs_bck1_HF_sub, header_nosrcs_bck1_HF_sub, overwrite=True)

            
    ##################################################
    # Compute [HF & sources ] subtracted img
    # SOURCES REPLACED BY PATCHES!
    ##################################################
    if DEBUG==True:
        # NOTE : THIS SECTION IS NOT NECESSARY, AND IT IS USED ONLY
        #        FOR TEST / DEBUGGING PURPOSES
        # --------------------------------------------------
        data_nosrcs_HF_sub=data_nosrcs_bck1_HF_sub+data_bck1
        header_nosrcs_HF_sub=header
        # Save [sources + HF] subtracted img to a file 
        img_nosrcs_HF_sub='nosrcs_HF_sub_'+original_image_name
        fits.writeto(WORKDIR+'/'+img_nosrcs_HF_sub,data_nosrcs_HF_sub ,header_nosrcs_HF_sub , overwrite=True)
    ##################################################


    ##################################################
    # Compute Small scale background
    ##################################################
    # THIS OPERATION CAN NOT BE PERFORMED BEFORE, AS 
    # THE SMALL SCALE USED WOULD INTERFERE WITHE THE 
    # IDENTIFICATION OF THE SOURCES !
    img_bck2='bck2_'+original_image_name # SMALL SCALE BCK
    BACK_SIZE_SS=1*FWHM_PIX # SMALL SCALE BACK SIZE
    #-------------------------------------------------
    # Run SExtractor here
    run_sex1(WORKDIR+'/'+img_nosrcs_bck1_HF_sub, WORKDIR+'/'+img_bck2, SEX_FOLDER,filter_name,pix_scale,FWHM_PIX_x,FWHM_PIX_y,BACK_SIZE=BACK_SIZE_SS,BACK_FILTERSIZE=2,img_type="BACKGROUND")
    # read bck image for successive computations:
    data_bck2, header_bck2 = read_fits_image(WORKDIR+'/'+img_bck2)
    # get image WCS (from astropy.wcs import WCS)
    wcs_bck2=WCS(header_bck2)
    # ------------------------------------------------
    # bck3: Background large sacale + small scale 
    data_bck3=data_bck1+data_bck2
    # ------------------------------------------------
    # SUBTRACT refined/small scale BCKground from img_nosrcs_bck1_HF_sub
    data_nosrcs_bck3_HF_sub=data_nosrcs_bck1_HF_sub-data_bck2 # data_bck2 NOT data_bck3! It is already bck1 subtracted!
    header_nosrcs_bck3_HF_sub=header
    #-------------------------------------------------
    # Save [bck3 + sources + HF pattern] subtracted img to a file 
    img_nosrcs_bck3_HF_sub='nosrcs_bck3_HF_sub_'+original_image_name
    fits.writeto(WORKDIR+'/'+img_nosrcs_bck3_HF_sub, data_nosrcs_bck3_HF_sub, header_nosrcs_bck3_HF_sub, overwrite=True)
    ##################################################

    ##################################################
    # Compute [ L+S scale background, sources ] subtracted img
    # HF patterns are kept
    # SOURCES REPLACED BY PATCHES!
    ##################################################
    if DEBUG==True:
        # NOTE : THIS SECTION IS NOT NECESSARY, AND IT IS USED ONLY
        #        FOR TEST / DEBUGGING PURPOSES
        # --------------------------------------------------
        data_nosrcs_bck3_sub=img_nosrcs_patch_save-data_bck2
        header_nosrcs_bck3_sub=header
        # Save [bck3 + sources] subtracted img to a file 
        img_nosrcs_bck3_sub='nosrcs_bck3_sub_'+original_image_name
        fits.writeto(WORKDIR+'/'+img_nosrcs_bck3_sub,data_nosrcs_bck3_sub ,header_nosrcs_bck3_sub , overwrite=True)
    ##################################################
        

    data_nosrcs_bck1_HF_sub=None # Not to be used anymore


    ########################################
    # DETECT RADIAL AND ELLIPTICAL PATTERNS
    ########################################
    # Detect radial and PSF-shaped (elliptical) patterns
    # from the image by averaging pixel values along strips
    # oriented radially or along ellipses concentrical with
    # the image itself. The size and width of the strips is 
    # set using the parameters here below. 
    # Narrower and/or shorter strips will be more sensitive 
    # to patterns but will also increase the possibility of
    # "allucination" artifacts.
    # Wider or longer strips will be less sensitive to
    # patterns but will be more stable over the possible 
    # introduction of "allucination" artifacts.
    # Increase the pattern detection threshold to reduce
    # artifacts.

    #-------------------------------------------------------------------
    # If radial or elliptical patterns are not computed, 
    # this image wil be used for the next steps of the process:
    img_nosrcs_bck3_HF_REP_sub='nosrcs_bck3_HF_sub_'+original_image_name
    data_nosrcs_bck3_HF_REP_sub=np.copy(data_nosrcs_bck3_HF_sub)
    #-------------------------------------------------------------------

    #------------------------------------------------------
    # PARAMETERS
    #------------------------------------------------------
    # Strips half length
    Strip_Half_length=0.5*REP_STRIP_LENGTH*int(FWHM_PIX_max) # default: 3 times the max FWHM in pixels
    # Strips half widths
    Strip_Half_width= 0.5 
    # Sigma threshold (consider patterns only when above this thresold)
    sigma_thresh=REP_THRESH # 1.25
    #------------------------------------------------------

    if (RAD_PATT_ISOLATE==True or IMG_RAD_PATT==True):
        
        #LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL
        # DETECT radial PATTERNS
        #LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL
        # Transform the linear widths and lengths
        # of the strips into angular and radial widths.
        # Needed to set the inputs of "rad_bck_img()".
        #--------------------------------------
        # Image radius in pixels
        #img_radius_pix=np.size(data_nosrcs_bck3_HF_sub[0,0,:,0]+data_nosrcs_bck3_HF_sub[0,0,0,:])//2
        img_radius_pix=np.size(data_nosrcs_bck3_HF_sub[:,0]+data_nosrcs_bck3_HF_sub[0,:])//2
        # Image circumference in pixels
        img_circ_pix=2.*np.pi*img_radius_pix
        #---------------------------------------
        DELTA_THETA=2.*np.pi * Strip_Half_width/ img_circ_pix
        DELTA_RAD= Strip_Half_length
        #---------------------------------------
        rad_bck_img=radial_patterns(data_nosrcs_bck3_HF_sub,sigma_thresh=sigma_thresh, delta_theta=DELTA_THETA,delta_rad=DELTA_RAD, SHOW_PLOT=SHOW_PLOT)

        if RAD_PATT_ISOLATE==True:
            # Update image used for the next steps
            # REP = Radial Ellliptical Patterns 
            data_nosrcs_bck3_HF_REP_sub=np.copy(data_nosrcs_bck3_HF_sub)
            data_nosrcs_bck3_HF_REP_sub=data_nosrcs_bck3_HF_sub-rad_bck_img

        if SHOW_PLOT==True:
            plt.imshow(rad_bck_img, cmap='gray')
            plt.title('Residual radial patterns')
            # invert the y-axis (to make it consistent with visualization in ds9)
            plt.gca().invert_yaxis()
            plt.show()

        if IMG_RAD_PATT==True:
            # Save Radial patterns img to a file 
            img_rad_pat_name='Radial_Patterns_'+original_image_name
            fits.writeto(WORKDIR+'/'+img_rad_pat_name,rad_bck_img ,header , overwrite=True)

    if (ELL_PATT_ISOLATE==True or IMG_ELL_PATT==True):
        #LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL
        # DETECT PSF-like PATTERNS (elliptical patterns)
        #LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL
        ell_bck_img=compute_elliptical_structure(data_nosrcs_bck3_HF_sub,sigma_thresh=sigma_thresh, stripes_width=Strip_Half_width*2, stripes_length=Strip_Half_length*2., AXES_RATIO=FWHM_PIX_x/FWHM_PIX_y, P_ANGLE=P_ANGLE)

        if ELL_PATT_ISOLATE==True:
            # Update image used for the next steps
            # REP = Radial Ellliptical Patterns 
            data_nosrcs_bck3_HF_REP_sub=np.copy(data_nosrcs_bck3_HF_sub)
            data_nosrcs_bck3_HF_REP_sub=data_nosrcs_bck3_HF_sub-ell_bck_img
            
        if SHOW_PLOT==True:
            plt.imshow(ell_bck_img, cmap='gray')
            plt.title('Residual elliptical patterns')
            # invert the y-axis (to make it consistent with visualization in ds9)
            plt.gca().invert_yaxis()
            plt.show()

        if IMG_ELL_PATT==True:
            
            # Save Elliptical patterns img to a file 
            img_ell_pat_name='Elliptical_Patterns_'+original_image_name
            fits.writeto(WORKDIR+'/'+img_ell_pat_name,ell_bck_img ,header , overwrite=True)

    # ------------------------------------------------------------------------
    # Save [bck3 + sources + HF pattern + radial res] subtracted img to a file 
    # ------------------------------------------------------------------------
    # Note: if no radial and/or elliptical patterns are subtracted, the image is
    #       identical to img_nosrcs_bck3_HF_sub
    img_nosrcs_bck3_HF_REP_sub='nosrcs_bck3_HF_REP_sub_'+original_image_name
    header_nosrcs_bck3_HF_REP_sub=header_nosrcs_bck3_HF_sub
    fits.writeto(WORKDIR+'/'+img_nosrcs_bck3_HF_REP_sub, data_nosrcs_bck3_HF_REP_sub, header_nosrcs_bck3_HF_sub, overwrite=True)
    # idx0=np.where(rad_and_ell_bck_img==0)
    # idx_no_0=np.where(rad_and_ell_bck_img!=0)
    # np.size(idx_no_0)/(np.size(idx0)+np.size(idx_no_0)) # ~0.5697 for random noise ~0.382 with patterns

    ###################################################


    ###################################################
    # Compute underlying local RMS (using sextractor)
    ###################################################
    img_rms1='rms1_'+original_image_name
    #run_sex1(WORKDIR+'/'+img_nosrcs_bck3_HF_sub,WORKDIR+'/'+img_rms1, SEX_FOLDER,filter_name,pix_scale,FWHM_PIX_x,FWHM_PIX_y,BACK_SIZE_LS,BACK_FILTERSIZE,img_type="BACKGROUND_RMS")
    run_sex1(WORKDIR+'/'+img_nosrcs_bck3_HF_REP_sub,WORKDIR+'/'+img_rms1, SEX_FOLDER,filter_name,pix_scale,FWHM_PIX_x,FWHM_PIX_y,BACK_SIZE_LS,BACK_FILTERSIZE,img_type="BACKGROUND_RMS")
    # read rms image for successive computations:
    data_rms1, header_rms1 = read_fits_image(WORKDIR+'/'+img_rms1)
    # get image WCS (from astropy.wcs import WCS)
    wcs_rms1=WCS(header_rms1)


    ###################################################
    # Obtain spatially "flat" noise (normalize noise 
    # by the local RMS (obtained from SExtractor)
    ###################################################
    # Flat_noise=(data_nosrcs_bck3_HF_sub/data_rms1)*np.median(data_rms1)
    Flat_noise=(data_nosrcs_bck3_HF_REP_sub/data_rms1)*np.median(data_rms1)
    header_Flat_noise=header
    #-------------------------------------------------
    # Save spatially "flat" noise img to a file 
    img_flat_noise='Flat_noise_'+original_image_name
    fits.writeto(WORKDIR+'/'+img_flat_noise, Flat_noise, header_Flat_noise, overwrite=True)





    ####################################################
    ####################################################
    #              SIMULATE NEW IMAGE 
    ####################################################
    ####################################################




    ##################################################
    # Fill sim image with random values replicating the 
    # Flux distribution in the flat pure noise image 
    ##################################################
    perc_min=2.5
    perc_max=97.5
    # Note: the data array is not just x by y. It has multiple dimensions.
    #       dimensions 2 ad 3 correspond to y and x
    #-------------------------------------------------
    # Replicate original distribution as it is (Replicate_opt='replicate') 
    # or replicate the best fitting gaussian distribution(Replicate_opt='Gfit')
    Replicate_opt='replicate' # 'Gfit' 
    #-------------------------------------------------
    # pixel-to pixel noise, same distrib as real flat noise + bck small sale
    #data_1pix_noise,sigma_Gnoise=sim_pix_distrib(data_nosrcs_bck1_HF_sub,perc_min,perc_max,SHOW_PLOT=True,sim_img=Replicate_opt,plot_title='Pixel-flux distribution (img no srcs, no bck, no HFN)')

    # pixel-to pixel noise, same distrib as real flat noise
    data_1pix_noise,sigma_Gnoise=sim_pix_distrib(Flat_noise,perc_min,perc_max,SHOW_PLOT=SHOW_PLOT,sim_img=Replicate_opt,plot_title='Pixel-scale flux distribution ("Flattened" pure noise img)')

    #-------------------------------------------------
    if SHOW_PLOT==True:
        plt.imshow(data_1pix_noise, cmap='Greys')
        plt.title('Noise at pixel scale level')
        # invert the y-axis (to make it consistent with visualization in ds9)
        plt.gca().invert_yaxis()
        plt.show()

    # Save simulated Noise on a pixel noise scale
    if DEBUG==True:
        # NOTE : THIS SECTION IS NOT NECESSARY, AND IT IS USED ONLY
        #        FOR TEST / DEBUGGING PURPOSES
        header_sim_Flat_pix_noise=header
        # Save flat pixel-scale noise img to a file 
        img_sim_Pixscale_noise='sim_Pixscale_noise_'+original_image_name
        fits.writeto(WORKDIR+'/'+img_sim_Pixscale_noise,data_1pix_noise,header_sim_Flat_pix_noise, overwrite=True)
    

    ##################################################
    # CONVOLVE SMALL SCALE pixel-noise with 
    # A) PSF
    # B) local noise auto-correlation function, or
    # C) Global noise auto-correlation function, or
    # D) (A+B) PSF & local noise auto-correlation function
    ##################################################
    # set at the beginning of the main() function
    # NOISE_CONV="PSF_&_local_ACF" #"local_ACF" # "PSF" # "ACF"
    ##################################################

    #------------------------------------------------
    # Option A: CONVOLVE SMALL SCALE pixel-noise with
    #           PSF stamp
    #------------------------------------------------
    if (NOISE_CONV=="PSF" or NOISE_CONV=="PSF_&_local_ACF") :
        # Convolve img and PSF (mode=same to obtain the same image size)
        test_image_A = scipy.signal.convolve2d(data_1pix_noise,PSF_img,mode='same',boundary='wrap')

        if SHOW_PLOT==True:
            plt.imshow(test_image_A, cmap='gray')
            plt.title('pixel-scale noise convolved with PSF')
            # invert the y-axis (to make it consistent with visualization in ds9)
            plt.gca().invert_yaxis()
            plt.show()

        # Save NOT-NORMALIZED flat PSF-scale noise img to a TEMPORARY FILE (tmp)
        #(needed by SExtractor below) 

        header_sim_tmp_Flat_PSF_noise=header
        sim_tmp_Flat_PSFscale_noise='sim_tmp_Flat_PSFscale_noise_'+original_image_name
        fits.writeto(WORKDIR+'/'+sim_tmp_Flat_PSFscale_noise,test_image_A,header_sim_tmp_Flat_PSF_noise, overwrite=True)

        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # Remove Small scale background from PSF-convolved
        # pixel noise SIM image
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        sim_bck_tmp='sim_bck_tmp_'+original_image_name # SMALL SCALE BCK on sim img , TEMPORARY FILE (tmp)
        #-------------------------------------------------
        # Run SExtractor here (to compute small scale background image)
        run_sex1(WORKDIR+'/'+sim_tmp_Flat_PSFscale_noise, WORKDIR+'/'+sim_bck_tmp, SEX_FOLDER,filter_name,pix_scale,FWHM_PIX_x,FWHM_PIX_y,BACK_SIZE=BACK_SIZE_SS,BACK_FILTERSIZE=2,img_type="BACKGROUND")
        # read bck image for successive computations:
        data_bck_sim_tmp, header_bck_sim_tmp = read_fits_image(WORKDIR+'/'+sim_bck_tmp)
        # get image WCS (from astropy.wcs import WCS)
        wcs_bck_sim_tmp=WCS(header_bck_sim_tmp)
        # ------------------------------------------------
        # SUBTRACT small scale BCKground from simulated, not normalized, image
        test_image_A =test_image_A-data_bck_sim_tmp
        #-------------------------------------------------
        # Save bck subtracted sim img to a file 
        if DEBUG==True:
            # NOTE : THIS SECTION IS NOT NECESSARY, AND IT IS USED ONLY
            #        FOR TEST / DEBUGGING PURPOSES
            sim_bck_sub_tmp2='sim_bck_sub_tmp2_'+original_image_name
            fits.writeto(WORKDIR+'/'+sim_bck_sub_tmp2, test_image_A, header, overwrite=True)
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # ITERATIVE FLUX DISTRIBUTION NORMALIZATION:
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # Re-normalize Flux distribution in the simulated image
        # (no sources, no bck, no HF patterns) to make it similar
        # to the distribution in the original pure noise image
        # (original image with no background, no sources 
        # and no High Frequency Noise (HFN)
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        test_image_A=normalize_distribution2(test_image_A,Flat_noise, perc1=16, perc2=84, recenter='Peak', precision=200)
        if SHOW_PLOT==True:
            plot_compare_distrib(Flat_noise,test_image_A,label1='Img No sources, no bck and no HFN. Flat rms', label2='Sim img No sources, no bck and no HFN. Flat rms)',xlabel='Flux [original image units]', ylabel='Peak normalized distribution', title='Flux distribution [original vs simulated] (no srcs, no bck, and no HFN) flat rms' , norm='peak')
        #-------------------------------------------------------------

#        # Save simulated FLAT Noise on a PSF scale
#        if DEBUG==True:
#            # NOTE : THIS SECTION IS NOT NECESSARY, AND IT IS USED ONLY
#            #        FOR TEST / DEBUGGING PURPOSES
#            header_sim_Flat_PSF_noise=header
#            # Save flat pixel-scale noise img to a file 
#            img_sim_Flat_PSFscale_noise='sim_Flat_PSFscale_noise_'+original_image_name
#            fits.writeto(WORKDIR+'/'+img_sim_Flat_PSFscale_noise,test_image_A,header_sim_Flat_PSF_noise, overwrite=True)

        if NOISE_CONV=="PSF" :
            sim_data = test_image_A
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@




    #------------------------------------------------
    # Option B: CONVOLVE SMALL SCALE pixel-noise with
    #           LOCAL noise auto-correlation function
    #------------------------------------------------
    if (NOISE_CONV == "local_ACF" or NOISE_CONV=="PSF_&_local_ACF") :

        if NOISE_CONV == "local_ACF":
            base_image=np.copy(Flat_noise)
            base_image[:]=0.
        if NOISE_CONV == "PSF_&_local_ACF":
            base_image=np.copy(test_image_A)

        N_FWHM_ACF=10 # Size, in FWHM, of the blocks inside which the ACF is computed
        shrink_factor=1.0 # initial ACF shrink factor (increases or decreases based on granularity coparison) 
        tolerance=0.02 # 2% tolerance in granularity (comparing original ad sim. image)
        granul_thresh=1.0 # N Sigma at which the granularity is checked
        # Reference granularity
        Flat_noise_Gr=get_granularity(Flat_noise, N_sigma=granul_thresh)

        N_cyg=7 # Max number of cycles
        i=0
        while (i<N_cyg):

            test_image_B=base_image+convolve_local_ACF(data_1pix_noise,Flat_noise,box_size=round(N_FWHM_ACF*max([FWHM_PIX_x,FWHM_PIX_y])),SHOW_PLOT=False, norm_blocks='Global',shrink_factor=shrink_factor)

            # Check granualrity:
            test_image_B_Gr=get_granularity(test_image_B, N_sigma=granul_thresh)

            print(i, test_image_B_Gr , Flat_noise_Gr, shrink_factor)

            # Compare granularity with reference
            if (test_image_B_Gr > Flat_noise_Gr*(1.0-tolerance) and test_image_B_Gr < Flat_noise_Gr*(1.0+tolerance)):
                #print('EXIT CYCLE')
                i=N_cyg
            else:
                #shrink_factor=shrink_factor * (1-0.5*(1-(test_image_B_Gr/Flat_noise_Gr)))
                #shrink_factor=(shrink_factor + shrink_factor * (1-(1-(test_image_B_Gr/Flat_noise_Gr))))/2.
                if NOISE_CONV == "local_ACF":
                    shrink_factor=shrink_factor*(1 + (test_image_B_Gr/Flat_noise_Gr) )/2.
                if NOISE_CONV == "PSF_&_local_ACF":
                    shrink_factor=shrink_factor*test_image_B_Gr/Flat_noise_Gr

            i=i+1

        
        ################################################
        
        sim_data=test_image_B 
        
        #############################


    #------------------------------------------------
    # Option C: CONVOLVE SMALL SCALE pixel-noise with
    #           global noise auto-correlation function
    #------------------------------------------------
    if NOISE_CONV=="ACF":

        # Compute the noise ACF
        ACF = scipy.signal.correlate2d(Flat_noise, Flat_noise, mode='same', boundary='wrap')
        # SELECT ACF CENTRAL AREA ONLY
        #ACF=ACF[int(np.size(ACF[:,0])/2)-int(np.size(PSF_img[:,0])):int(np.size(ACF[:,0])/2)+int(np.size(PSF_img[:,0])), int(np.size(ACF[0,:])/2)-int(np.size(PSF_img[0,:])):int(np.size(ACF[0,:])/2)+int(np.size(PSF_img[0,:]))]
        #ACF=ACF[int(np.size(ACF[:,0])/2)-int(np.size(PSF_img[:,0])*2/3):int(np.size(ACF[:,0])/2)+int(np.size(PSF_img[:,0])*2/3), int(np.size(ACF[0,:])/2)-int(np.size(PSF_img[0,:])*2/3):int(np.size(ACF[0,:])/2)+int(np.size(PSF_img[0,:])*2/3)]
        ACF=ACF[int(np.size(ACF[:,0])/2)-int(np.size(PSF_img[:,0])*0.5):int(np.size(ACF[:,0])/2)+int(np.size(PSF_img[:,0])*0.5), int(np.size(ACF[0,:])/2)-int(np.size(PSF_img[0,:])*0.5):int(np.size(ACF[0,:])/2)+int(np.size(PSF_img[0,:])*0.5)]

        # Normalize the ACF
        ACF = ACF / np.max(ACF)
        
        if SHOW_PLOT==True:
            # Create a plot to visualize the ACF
            plt.figure(figsize=(8, 8))
            plt.imshow(ACF, cmap='viridis', origin='lower', extent=[-ACF.shape[1]//2, ACF.shape[1]//2, -ACF.shape[0]//2, ACF.shape[0]//2])
            plt.colorbar(label='Normalized ACF')
            plt.title('Noise Auto-Correlation Function (ACF)')
            plt.xlabel('Offset (Pixels)')
            plt.ylabel('Offset (Pixels)')
            plt.show()

        # SAVE ACF IMAGE TO A FILE
        hdu_ACF = fits.PrimaryHDU(ACF,header)
        #    hdu_ACF = fits.PrimaryHDU(data_masked,header)
        hdu_ACF.header["COMMENT"] = "This is the image of the ACF created by noisempire.py (Ivano Baronchelli 2023)."
        hdu_ACF.header["COMMENT"] = "The image from which this ACF is computed is"
        hdu_ACF.header["COMMENT"] = original_image_name
        hdu_ACF.writeto(WORKDIR+'/ACF_sim_'+original_image_name, overwrite=True)

        # Convolution with ACF
        sim_data = scipy.signal.convolve2d(data_1pix_noise,ACF,mode='same',boundary='wrap') # DEFAULT
 



    #######################################################
    # ITERATIVE FLUX DISTRIBUTION NORMALIZATION:
    #######################################################
    # Re-normalize Flux distribution in the simulated image
    # (no sources, no bck, no HF patterns) to make it similar
    # to the distribution in the original pure noise image
    # (original image with no background, no sources 
    # and no High Frequency Noise (HFN)
    ###################################################
    sim_data=normalize_distribution2(sim_data,Flat_noise, perc1=16, perc2=84, recenter='Peak', precision=200)
    if SHOW_PLOT==True:
        plot_compare_distrib(Flat_noise,sim_data,label1='Img No sources, no bck and no HFN. Flat rms', label2='Sim img No sources, no bck and no HFN. Flat rms)',xlabel='Flux [original image units]', ylabel='Peak normalized distribution', title='Flux distribution [original vs simulated] (no srcs, no bck, and no HFN) flat rms' , norm='peak')
    #-------------------------------------------------------------

    # Save simulated FLAT Noise on a PSF scale
    if DEBUG==True:
        # NOTE : THIS SECTION IS NOT NECESSARY, AND IT IS USED ONLY
        #        FOR TEST / DEBUGGING PURPOSES
        header_sim_Flat_PSF_noise=header
        # Save flat pixel-scale noise img to a file 
        img_sim_Flat_PSFscale_noise='sim_Flat_PSFscale_noise_'+original_image_name
        fits.writeto(WORKDIR+'/'+img_sim_Flat_PSFscale_noise,sim_data,header_sim_Flat_PSF_noise, overwrite=True)

    if SHOW_PLOT==True:
        plt.imshow(sim_data, cmap='Greys')
        plt.title('pixel-scale noise convolved with PSF, ACF or both')
        # invert the y-axis (to make it consistent with visualization in ds9)
        plt.gca().invert_yaxis()
        plt.show()

    ###################################################



    ###################################################
    # Remodulate noise on the underlying RMS 
    # (Previously computed using sextractor)
    ###################################################
    #print('WARNING 1: IVANO, bisogna dividere per np.median(data_rms1) oppure per np.std(sim_data)???')
    sim_data = (sim_data/np.median(data_rms1))*data_rms1
    #sim_data = (sim_data/np.std(sim_data))*data_rms1    
    if SHOW_PLOT==True:
        plot_compare_distrib(data_nosrcs_bck3_HF_REP_sub,sim_data, label1="1) Img No sources, no bck, no HFN, no REP. Real rms distrib.", label2="Sim img no sources, no bck and no HFN. Remodulated rms", xlabel="Flux [original image units]", ylabel="Peak normalized distribution", title="Flux distrib. [orig vs sim] (no srcs, no bck, and no HFN) realistic rms distrib.", norm='peak')
    # RENORMALIZATION
    sim_data=normalize_distribution2(sim_data,data_nosrcs_bck3_HF_REP_sub, perc1=16, perc2=84, recenter='Peak', precision=200)
    if SHOW_PLOT==True:
        plot_compare_distrib(data_nosrcs_bck3_HF_sub,sim_data, label1="2) Img No sources, no bck and no HFN. Real rms distrib.", label2="Sim img no sources, no bck and no HFN. Remodulated rms", xlabel="Flux [original image units]", ylabel="Peak normalized distribution", title="Flux distrib. [orig vs sim] (no srcs, no bck, and no HFN) realistic rms distrib.", norm='peak')
    

    # Save simulated Noise on a PSF scale
    if DEBUG==True:
        # NOTE : THIS SECTION IS NOT NECESSARY, AND IT IS USED ONLY
        #        FOR TEST / DEBUGGING PURPOSES
        header_sim_PSF_noise=header
        # Save flat pixel-scale noise img to a file 
        img_sim_PSFscale_noise='sim_PSFscale_noise_'+original_image_name
        fits.writeto(WORKDIR+'/'+img_sim_PSFscale_noise,sim_data,header_sim_PSF_noise, overwrite=True)



    #-------------------------------------------------------------

    
    ###################################################
    # Add back RADIAL AND ELLIPTICAL PATTERNS
    ###################################################

    if RAD_PATT_ISOLATE==True:
        sim_data=sim_data+rad_bck_img
        rad_and_ell_patterns=np.copy(rad_bck_img)

    if ELL_PATT_ISOLATE==True:
        sim_data=sim_data+ell_bck_img
        rad_and_ell_patterns=rad_and_ell_patterns+ell_bck_img

    if (RAD_PATT_ISOLATE==True or ELL_PATT_ISOLATE==True):
        # Compare distributions 
        plot_compare_distrib(data_nosrcs_bck3_HF_sub, sim_data, rad_and_ell_patterns, label1="1) Original Image with radial/elliptical patterns (no srcs, no bck, no HF patterns", label2="Simulated image with radial/elliptical patterns (no srcs, and no HFN)", label3="Radial/Elliptical Patterns", xlabel="Flux [original image units]", ylabel="Peak normalized distribution", title="Flux distrib. [orig vs sim] (yes bck, no srcs, no HFN) realistic rms distrib.", norm='peak')

        # RENORMALIZATION
        sim_data=normalize_distribution2(sim_data,data_nosrcs_bck3_HF_sub, perc1=16, perc2=84, recenter='Peak', precision=200)

        # Save simulated Image with radial/elliptical patterns
        if DEBUG==True:
            # NOTE : THIS SECTION IS NOT NECESSARY, AND IT IS USED ONLY
            #        FOR TEST / DEBUGGING PURPOSES
            header_sim_nosrcs_nobck_noHF=header
            # Save flat pixel-scale noise img to a file 
            img_sim_nosrcs_nobck_noHF='sim_nosrcs_nobck_noHF_'+original_image_name
            fits.writeto(WORKDIR+'/'+img_sim_nosrcs_nobck_noHF,sim_data,header_sim_nosrcs_nobck_noHF, overwrite=True)


    ##################################################
    # Add background to simulated image
    ##################################################
    if BCK1_TYPE=='Original':
        #-------------------------------------------------
        # Add real small scale bck to the simulated image
        #-------------------------------------------------
        sim_data = sim_data + data_bck1

    if BCK2_TYPE=='Original':
        #-------------------------------------------------
        # Add real small scale bck to the simulated image
        #-------------------------------------------------
        sim_data = sim_data + data_bck2

    if BCK1_TYPE=='Simulate':
        #-------------------------------------------------
        # Simulate Large scale background
        #-------------------------------------------------

        # simulate pixel-to pixel noise with same distrib as bck on Large scale 
        data_1pix_bck_Ls,sigma_Gnoise_bck_Ls=sim_pix_distrib(data_bck1,perc_min,perc_max,SHOW_PLOT=SHOW_PLOT,sim_img=Replicate_opt,plot_title='Pixel-flux distribution (large-scale bck)')

        # TEST
        # data_1pix_bck_Ls=data_1pix_noise

        # visualize pixel-to pixel flux simulating Large scale background
        if SHOW_PLOT==True:
            plt.imshow(data_1pix_bck_Ls, cmap='Greys')
            plt.title('Large scale back simulated at pixel scale level')
            # invert the y-axis (to make it consistent with visualization in ds9)
            plt.gca().invert_yaxis()
            plt.show()

        # TEST TEST TEST
        # Compute downsalmpling scale (we are not interested on the smallest scales
        downsamp_shape = np.array(np.shape(data_bck1))/(BACK_SIZE_LS/2)
        downsamp_fact = downsamp_shape / np.array(np.shape(data_bck1))
        # Downsample bck image 
        downsamp_bck1 = zoom(data_bck1, downsamp_fact)
        # A) Downsample stochastic pix-to-pix distrib
        downsamp_1pix_bck_Ls= zoom(data_1pix_bck_Ls, downsamp_fact)
        # B1) Compute auto-correlation function for downsampled bck image
        downsamp_acf_bck1 = scipy.signal.correlate2d(downsamp_bck1, downsamp_bck1, mode="same", boundary="wrap")
        # B2) Normalize downsampled LS-bck ACF   
        downsamp_acf_bck1 = downsamp_acf_bck1/np.max(downsamp_acf_bck1)
        # Convolve A) Downsampled stochastic pix-to-pix distrib with B) ACF for downsampled bck image
        downsamp_sim_bck1 = scipy.signal.correlate2d(downsamp_1pix_bck_Ls, downsamp_acf_bck1, mode="same", boundary="wrap")
        # Compute the required upsampling factor to match the desired output shape
        upsample_factor = np.ceil(np.array(np.shape(data_bck1)) / np.array(np.shape(downsamp_sim_bck1))).astype(int)
        # Upsample the convolved image to match the original image size
        sim_bck_Ls = zoom(downsamp_sim_bck1, upsample_factor, order=1)[:data_bck1.shape[0], :data_bck1.shape[1]]
        #sim_bck_Ls= zoom(downsamp_sim_bck1, 1./downsamp_fact)
        # Normalize bck ACF
        #resamp_acf_bck1 = downsamp_acf_bck1/np.max(downsamp_acf_bck1)
        #fits.writeto(WORKDIR+'/'+'TEST.fits',sim_bck_Ls,header, overwrite=True)
        # TEST TEST TEST

        # RENORMALIZATION
        sim_bck_Ls=normalize_distribution2(sim_bck_Ls,data_bck1, perc1=16, perc2=84, recenter='Peak', precision=200)
        if SHOW_PLOT==True:
            plot_compare_distrib(data_bck1,sim_bck_Ls, label1="Original bck on large scale", label2="Sim bck on large scale", xlabel="Flux [original image units]", ylabel="Peak normalized distribution", title="Large scale background - flux distrib. [orig vs sim]", norm='peak')

        # Save simulated large scale background to a file (DEBUG)
        if DEBUG==True:
            # NOTE : THIS SECTION IS NOT NECESSARY, AND IT IS USED ONLY
            #        FOR TEST / DEBUGGING PURPOSES
            header_sim_bck1=header#_bck1
            # Save background small scale to a file 
            img_sim_bck1='sim_bck1_'+original_image_name
            fits.writeto(WORKDIR+'/'+img_sim_bck1,sim_bck_Ls,header_sim_bck1, overwrite=True)

        # Add simulated bck small scale to simulated image
        sim_data = sim_data + sim_bck_Ls
        

    
    if BCK2_TYPE=='Simulate':
        #-------------------------------------------------
        # Simulate small scale background
        #-------------------------------------------------

        # simulate pixel-to pixel noise with same distrib as bck on small scale 
        data_1pix_bck_ss,sigma_Gnoise_bck_ss=sim_pix_distrib(data_bck2,perc_min,perc_max,SHOW_PLOT=SHOW_PLOT,sim_img=Replicate_opt,plot_title='Pixel-flux distribution (small-scale bck)')

        # TEST
        # data_1pix_bck_ss=data_1pix_noise
    
        # visualize pixel-to pixel flux simulating small scale background
        if SHOW_PLOT==True:
            plt.imshow(data_1pix_bck_ss, cmap='Greys')
            plt.title('Small scale back simulated at pixel scale level')
            # invert the y-axis (to make it consistent with visualization in ds9)
            plt.gca().invert_yaxis()
            plt.show()

        # Convolve pixel-scale simulated bck with local ACF computed on the original small scale bck
        # NOTE: 
        # - Box size set to same size used for the local ACF computed on the underlying noise
        # - shrink_factor set to 1.0 and no check on the granularity on the final image is performed
        
        #sim_bck_ss1=convolve_local_ACF(data_1pix_bck_ss,data_bck2,box_size=1.0*round(N_FWHM_ACF*max([FWHM_PIX_x,FWHM_PIX_y])),SHOW_PLOT=False, norm_blocks='Global',shrink_factor=1.0)

        sim_bck_ss2=convolve_local_ACF(data_1pix_bck_ss,data_bck2,box_size=1.2*round(N_FWHM_ACF*max([FWHM_PIX_x,FWHM_PIX_y])),SHOW_PLOT=False, norm_blocks='Global',shrink_factor=1.0)

        sim_bck_ss3=convolve_local_ACF(data_1pix_bck_ss,data_bck2,box_size=3*round(N_FWHM_ACF*max([FWHM_PIX_x,FWHM_PIX_y])),SHOW_PLOT=False, norm_blocks='Global',shrink_factor=1.0)

        #sim_bck_ss=(sim_bck_ss1)#+sim_bck_ss2)#+sim_bck_ss3)/3.
        sim_bck_ss=(sim_bck_ss2+sim_bck_ss3)/2.

#        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#        # Compute small scale background from local 
#        # ACF-convolved simulated background small scale 
#        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#        sim_bck_ss_tmp='sim_bck_ss_tms_'+original_image_name
#        fits.writeto(WORKDIR+'/'+sim_bck_ss_tmp,sim_bck_ss , header, overwrite=True)
#        #-------------------------------------------------
#        # Run SExtractor here (to compute small scale background image)
#        img_sim_bck2_tmp='sim_bck2_'+original_image_name
#        run_sex1(WORKDIR+'/'+sim_bck_ss_tmp, WORKDIR+'/'+img_sim_bck2_tmp, SEX_FOLDER,filter_name,pix_scale,FWHM_PIX_x,FWHM_PIX_y,BACK_SIZE=BACK_SIZE_SS,BACK_FILTERSIZE=2,img_type="BACKGROUND")
#        # read bck image for successive computations:
#        sim_bck_ss, header_sim_bck_ss = read_fits_image(WORKDIR+'/'+img_sim_bck2_tmp)
#        # get image WCS (from astropy.wcs import WCS)
#        wcs_sim_bck_ss=WCS(header_sim_bck_ss)
#        # Remove the temporary files
#        try:
#            os.remove(WORKDIR+'/'+img_sim_bck2_tmp) # Remove temporary ss background
#        except FileNotFoundError:
#            print(WORKDIR+'/'+img_sim_bck2_tmp+" \n File not found. No action taken.")
#        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # Remove large scale background from local 
        # ACF-convolved simulated background small scale 
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # Save simulated background ss (at stage to a TEMPORARY file)
        sim_bck_ss_tmp='sim_bck_ss_tms_'+original_image_name
        fits.writeto(WORKDIR+'/'+sim_bck_ss_tmp,sim_bck_ss , header, overwrite=True)
        # LS background subtracted ss background: 
        sim_bck_ss_bckrem_tmp='sim_bck_ss_bckrem_tmp_'+original_image_name # TEMPORARY FILE (tmp)
        #-------------------------------------------------
        # Run SExtractor here (to compute large scale background from small scale bck sim image)
        run_sex1(WORKDIR+'/'+sim_bck_ss_tmp, WORKDIR+'/'+sim_bck_ss_bckrem_tmp, SEX_FOLDER,filter_name,pix_scale,FWHM_PIX_x,FWHM_PIX_y,BACK_SIZE=0.9*BACK_SIZE_LS,BACK_FILTERSIZE=2,img_type="BACKGROUND")
        # read bck image for successive computations:
        data_LSbck_on_ss_tmp, header_LSbck_on_ss_tmp = read_fits_image(WORKDIR+'/'+sim_bck_ss_bckrem_tmp)
        # get image WCS (from astropy.wcs import WCS)
        wcs_LSbck_on_ss_tmp=WCS(header_LSbck_on_ss_tmp)
        # ------------------------------------------------
        # SUBTRACT small scale BCKground from simulated, not normalized, image
        sim_bck_ss =sim_bck_ss-data_LSbck_on_ss_tmp
        #-------------------------------------------------try:
        # Remove the temporary files
        try:
            os.remove(WORKDIR+'/'+sim_bck_ss_tmp) # Remove temporary ss background
        except FileNotFoundError:
            print(WORKDIR+'/'+sim_bck_ss_tmp+" \n File not found. No action taken.")
        try:
            os.remove(WORKDIR+'/'+sim_bck_ss_bckrem_tmp) # remove LS background in ss background
        except FileNotFoundError:
            print(WORKDIR+'/'+sim_bck_ss_bckrem_tmp+" \n File not found. No action taken.")
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        # RENORMALIZATION
        sim_bck_ss=normalize_distribution2(sim_bck_ss,data_bck2, perc1=16, perc2=84, recenter='Peak', precision=200)
        if SHOW_PLOT==True:
            plot_compare_distrib(data_bck2,sim_bck_ss, label1="Original bck on small scale", label2="Sim bck on small scale", xlabel="Flux [original image units]", ylabel="Peak normalized distribution", title="small scale background - flux distrib. [orig vs sim]", norm='peak')

        # Save simulated small scale background to a file (DEBUG)
        if DEBUG==True:
            # NOTE : THIS SECTION IS NOT NECESSARY, AND IT IS USED ONLY
            #        FOR TEST / DEBUGGING PURPOSES
            header_sim_bck2=header#_bck2
            # Save background small scale to a file 
            img_sim_bck2='sim_bck2_'+original_image_name
            fits.writeto(WORKDIR+'/'+img_sim_bck2,sim_bck_ss,header_sim_bck2, overwrite=True)

        # Add simulated bck small scale to simulated image
        sim_data = sim_data + sim_bck_ss
        

    

    #--------------------------------------------------
    # sim_data = sim_data + data_bck3 # data_bck3 = data_bck1 + data_bck2
    #--------------------------------------------------

    # Compare simulated image + simulated background before renormalization
    if SHOW_PLOT==True:
        plot_compare_distrib(data_nosrcs_bck3_HF_sub+data_bck3,sim_data, data_bck3, label1="1) Original Image with bck (no srcs and no HF patterns", label2="Simulated image with bck (no srcs, and no HFN)", label3="Background", xlabel="Flux [original image units]", ylabel="Peak normalized distribution", title="Flux distrib. [orig vs sim] (yes bck, no srcs, no HFN) realistic rms distrib.", norm='peak')

    #-------------------------------------------------
    # RENORMALIZATION
    #-------------------------------------------------
    sim_data=normalize_distribution2(sim_data,data_nosrcs_bck3_HF_sub+data_bck3, perc1=16, perc2=84, recenter='Peak', precision=200)

    # Visualize simulated image + background after renormalization
    if SHOW_PLOT==True:
        plot_compare_distrib(data_nosrcs_bck3_HF_sub+data_bck3,sim_data, data_bck3, label1="2) Original Image with bck (no srcs and no HF patterns", label2="Simulated image with bck (no srcs, and no HFN)", label3="Background", xlabel="Flux [original image units]", ylabel="Peak normalized distribution", title="Flux distrib. [orig vs sim] (yes bck, no srcs, no HFN) realistic rms distrib.", norm='peak')
    #-------------------------------------------------------------

    #-------------------------------------------------
    # Save simulated Noise + background (Debug)
    #-------------------------------------------------
    if DEBUG==True:
        # NOTE : THIS SECTION IS NOT NECESSARY, AND IT IS USED ONLY
        #        FOR TEST / DEBUGGING PURPOSES
        header_noise_and_bck=header
        # Save  to a file 
        img_sim_noise_and_bck='sim_noise_and_bck_'+original_image_name
        fits.writeto(WORKDIR+'/'+img_sim_noise_and_bck,sim_data,header_noise_and_bck, overwrite=True)

    ###################################################
    # Add back HF patterns originally removed
    ###################################################
    sim_data = sim_data + img_filtered
    #--------------------------------------------------

    if SHOW_PLOT==True:
        plot_compare_distrib(data_nosrcs_bck3_HF_sub+data_bck3+img_filtered,sim_data, label1="1) Original Image with bck and HF patterns (no srcs)", label2="Simulated image with bck and HF patterns (no srcs)", xlabel="Flux [original image units]", ylabel="Peak normalized distribution", title="Flux distrib. [orig vs sim] (yes bck, yes HFN, no srcs) realistic rms distrib.", norm='peak')

    # RENORMALIZATION
    sim_data=normalize_distribution2(sim_data,data_nosrcs_bck3_HF_sub+data_bck3+img_filtered, perc1=16, perc2=84, recenter='Peak', precision=200)

    if SHOW_PLOT==True:
        plot_compare_distrib(data_nosrcs_bck3_HF_sub+data_bck3+img_filtered,sim_data, label1="2) Original Image with bck and HF patterns (no srcs)", label2="Simulated image with bck and HF patterns (no srcs)", xlabel="Flux [original image units]", ylabel="Peak normalized distribution", title="Flux distrib. [orig vs sim] (yes bck, yes HFN, no srcs) realistic rms distrib.", norm='peak')
    #-------------------------------------------------------------

    # Save simulated Noise + background + HF patterns
    if DEBUG==True:
        # NOTE : THIS SECTION IS NOT NECESSARY, AND IT IS USED ONLY
        #        FOR TEST / DEBUGGING PURPOSES
        header_sim_noise_bck_HFN=header
        # Save to a file 
        img_sim_noise_bck_HFN='sim_noise_bck_HFN_'+original_image_name
        fits.writeto(WORKDIR+'/'+img_sim_noise_bck_HFN,sim_data,header_sim_noise_bck_HFN, overwrite=True)


    ###################################################
    # Check original and simulated Flux Distribution
    ###################################################
    if SHOW_PLOT==True:
        plot_compare_distrib(orig_image,sim_data, label1="Original Image", label2="Simulated image", xlabel="Flux [original image units]", ylabel="Peak normalized distribution", title="Flux distrib. [orig vs sim]", norm='peak')

    # DO NOT RENORMALIZE HERE (The original image contains sources...)
    #-------------------------------------------------------------


    # To be added or to be modified
    # -  REP (Radial-elliptical Patterns): 
    #   - The anti-allucination feature does not allow to correctly transfer 
    #      the REP patterns to the simulated image. Deactivating it... it allucinates.
    # - The image simualted is even TOO similar to the input image. A mechanism
    #    for the simulation of the various patterns must be implemented (at the moment 
    #    the patterns/background/noise isolated in the original image are simply put
    #    into the simlated image as they are (except for the pixel-scale noise, before 
    #    convolving it with PSF and/or ACF). As an initial step towards this goal, 
    #    the same patterns can be put into the simulated image but amplified (or 
    #    attenuated). Additionally, the tolerance considered in some functions to 
    #    isolate/reproduce the various patterns can be left to vary inside pre-determined
    #    ranges, in order to add some randomization to the various steps.

#    stop()

    ###################################################
    # Update header and save simulated image (NOISE ONLY!)
    ###################################################
    # ----------------------------------------
    # Update header 
    # ----------------------------------------
    hdu = fits.PrimaryHDU(sim_data, header=header) 
    hdu.header["COMMENT"] = "This is a simulated image created using "+ Name+" "+Version+" "+Developer+" "+Years
    hdu.header["COMMENT"] = "The values in the original reference image were replaced by simulated noise and background."
    # Copy the WCS information from the original image to the new image
    #    hdu.header.update()
    hdu.header.update(wcs.to_header())
    # ----------------------------------------
    # Write simulated image only noise
    # ----------------------------------------
    hdu.writeto(WORKDIR+'/sim_'+original_image_name, overwrite=True)
    # ----------------------------------------
    
    ###################################################
    # ADD real sky to simulation and save
    ###################################################
    if REAL_SKY_IMAGE!=None:

        # Compute approximated STDdev in simulated image (square boxes used)
        #BOX_STDEV=stdv_of_array_blocks_means(sim_data, FWHM):

        # Convolve Real sky with PSF
        flat_data_rsi_conv = scipy.signal.convolve2d(flat_data_rsi,PSF_img,mode='same',boundary='wrap') 
        
        # Normalize maximum to 1
        flat_data_rsi_conv = (flat_data_rsi_conv-np.min(flat_data_rsi_conv))/np.max(flat_data_rsi_conv)
        #flat_data_rsi = (data_rsi-np.min(data_rsi))/np.max(data_rsi)

        # Normalize so that brightest simulated source is at "NSIGMA_RSKY" sigma above noise
        flat_data_rsi_conv = NSIGMA_RSKY*flat_data_rsi_conv*np.std(sim_data)/np.max(flat_data_rsi_conv)
        
        # Add simulated noise
        sim_data=sim_data+flat_data_rsi_conv

        if SHOW_PLOT==True:
            plt.imshow(sim_data, cmap='gray')
            plt.title("Simulated image + convolved REAL SKY")
            # invert the y-axis (to make it consistent with visualization in ds9)
            plt.gca().invert_yaxis()
            plt.show()

        # ----------------------------------------
        # Write simulated image noise + real sky
        # ----------------------------------------
        FINAL_SIM_IMG=np.copy(sim_data)
        # Write simulated image to file
        fits.writeto(WORKDIR+'/sim1_'+original_image_name,FINAL_SIM_IMG,header, overwrite=True)
        # ----------------------------------------
        
    stop()

if __name__ == '__main__':
    main()

