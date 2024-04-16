# NOISEMPIRE
Empirical noise simulator

Simulates a pure noise 2D image given a real ALMA cube or image in input
A real sky image (same size as reference image) can also be added.

Call this script as follows:
$ python noisempire.py config_file.txt
or, if you want to specify the image name in the command line:
$ python noisempire.py config_file.txt --INPUT_IMAGE input_image.fits

Additional information by calling:
$ python noisempire.py -h
or 
$ python noisempire.py --help

Examples of input images and real sky are provided
