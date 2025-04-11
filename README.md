# NOISEMPIRE 1.0.3 Noce
Empirical noise simulator

Simulates pure noise 2D images given a real ALMA cube or image in input.  
A real sky image (same size as reference image) can also be added.

Requirements
------------
- Source Extractor version 2.25.0 (2018-02-08)
- Python 3.7
- Python non standard packages: pbd, astropy, numpy, scipy, matplotlib, math, cv2, skimage, re, argparse 

Call this script as follows:
----------------------------
$ python noisempire.py config_file.txt  
If you want, you can specify some/all the parameters from command line (they will take precedence over those in the config file:  
Example:
$ python noisempire.py config_file.txt --INPUT_IMAGE=input_image.fits --REAL_SKY_IMAGE=real_sky.fits --DEBUG=False

Examples of input images and real sky are provided for a test run.

Additional information by calling:
---------------------------------

$ python noisempire.py -h  
or  
$ python noisempire.py --help  

The default configuration file is self-explanatory, and it is recommended to read it before using noisempire.

OUTPUTS
--------------------------
This is a complete list of all the outputs produced (some of them can be obtained ONLY using the DEBUG option)

MAIN IMAGE COMPONENTS (noise, sources, rms etc.)
------------------------------------------------
Original image background - large scale  
bck1_imagename.fits

Original image background - small scale  
bck2_imagename.fits

Original image after all noise patterns are subtracted  
Flat_noise_imagename.fits

High frequency Noise patterns isolated from the original image  
HFN_imagename.fits

simulated PSF of the input image  
PSF_sim_imagename.fits

rms map of the input image  
rms1_imagename.fits

sources identified in the original image (step 1 - mostly point like)  
srcs_only1_imagename.fits

sources identified in the original image (step 2 - point like and more extended components)  
srcs_only2_imagename.fits

Elliptical (PSF-like patterns) isolated from the original image  
Elliptical_Patterns_imagename.fits

Radial patterns isolated from the original image  
Radial_Patterns_imagename.fits


MAIN OUTPUTS
------------

simulated image 
sim_imagename.fits  


OTHER OUTPUTS
-------------

original image subtracted of background on large scale and high frequency patterns  
bck1_HF_sub_imagename.fits  

original image subtracted of background on large scale  
bck1_sub_imagename.fits  

original cube flattened to a 2D image  
FLAT_imagename.fits  

original image subtracted of sources, background on large scale and high frequency patterns  
nosrcs_bck1_HF_sub_imagename.fits  

original image subtracted of sources, background on small and large scale, high frequency patterns and Radial and/or elliptical patterns (REP)  
nosrcs_bck3_HF_REP_sub_imagename.fits  

original image subtracted of sources, background on large scale and high frequency patterns  
nosrcs_bck3_HF_sub_imagename.fits  

original image subtracted of sources, background on large and small scales  
nosrcs_bck3_sub_imagename.fits  

original image subtracted of sources and high frequency patterns  
nosrcs_HF_sub_imagename.fits  

original image subtracted of sources only  
nosrcs_patch_bck_imagename.fits  

original image subtracted of sources and large scale background  
nosrcs_patch_imagename.fits  

original image subtracted of large scale background and sources set to 0.  
nosrcs_imagename.fits  

simulated background on large scale (if BCK1_TYPE Simulate)  
sim_bck1_imagename.fits  

simulated background on small scale (if BCK2_TYPE Simulate)  
sim_bck2_imagename.fits  

simulated image with noise at PSF scale, noise small scale subtracted  (temporary file not to be considered)  
sim_bck_sub_tmp2_imagename.fits  

simulated image background on small scale (temporary file not to be considered)  
sim_bck_tmp_imagename.fits  

simulated flat image with noise at PSF scale   
sim_Flat_PSFscale_noise_imagename.fits  

simulated noise and background (no HF patterns yet)  
sim_noise_and_bck_imagename.fits  

simulated noise, background and HF patterns  
sim_noise_bck_HFN_imagename.fits  

xxxxx  
sim_nosrcs_nobck_noHF_imagename.fits    

simulated image with noise at PSF scale remodulated using the true rms of the original image  
sim_PSFscale_noise_imagename.fits  

simulated image with noise at pixels-scale only  
sim_Pixscale_noise_imagename.fits  

simulated flat image (temporary file not to be considered)  
sim_tmp_Flat_PSFscale_noise_imagename.fits  

# History

V1.0.2 --> V1.0.3
- noisempire can now properly treat images with odd sizes (the high frequency patterns
   were not correctly computed before, in these cases).
- noisempire can now properly work on rectangular images (it couldn't before due to some
   little bugs)
- NaN and repeated pixels are masked in the original image, at the beginning of the process
   A "non problematic" image is computed (IMG_NP in the configuration file).
   The process ignores areas occupied by "problematic" pixels. 
- Pixels below MIN_VAL are considered as problematic and masked pefore processing
- Pixels above MAX_VAL are considered as problematic and masked pefore processing
- The background at large and small scales is now simulated keeping into account possible underlying
   patterns at these same scales. These patterns are detected/measured computing the ACF at different
   scales, zooming out the original backgorund image.
- The image parameters (Pixel scale and Beam shape) can be read directly from the header or they
      can be provided by the user
- A prefix specified by the user is addedd to all the images created (parameter ALL_IMG_PREFIX)
- only selected images are saved, unless DEBUG option is active (list specified inside the configuration file)
- Parameter IMG_ELL_PATT replaced by IMG_ELL
- Parameter IMG_RAD_PATT replaced by IMG_RAD
- Execution time is measured and printed at the end of the process
- Some little bugs corrected

V1.0.1 --> V1.0.2 May 3 2024
- corrected bug (arising flattening cubes problem when extracting radial - elliptical patterns)

V1.0.0 --> V1.0.1 May 2 2024
- removed duplicated import (re)
- removed misleading comments 

