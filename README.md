# NOISEMPIRE
Empirical noise simulator

Simulates pure noise 2D images given a real ALMA cube or image in input.  
A real sky image (same size as reference image) can also be added.

Requirements
------------
- Source Extractor version 2.25.0 (2018-02-08)
- Python 3.7
- Python non standard packages: astropy, numpy, scipy, matplotlib, cv2, skimage

Call this script as follows:
----------------------------
$ python noisempire.py config_file.txt  
or, if you want to specify the image name in the command line:  
$ python noisempire.py config_file.txt --INPUT_IMAGE input_image.fits  

Examples of input images and real sky are provided for test run.

Additional information by calling:
---------------------------------

$ python noisempire.py -h  
or  
$ python noisempire.py --help  

Description of the outputs
--------------------------


