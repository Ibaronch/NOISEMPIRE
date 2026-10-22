# NOISEMPIRE 1.0.4 — Belice

DOI: XXX

**Empirical simulator for astronomical noise and background maps**

NOISEMPIRE uses an observed astronomical FITS image or cube as a reference to build a synthetic 2D realization of its noise and background. It estimates and separates source emission, large- and small-scale backgrounds, high-frequency patterns, and the local RMS. It then simulates noise using the reference image's flux statistics and spatial correlations, including the PSF and/or the local or global auto-correlation function (ACF).

The standard simulated product does not copy the detected source population from the reference image. Optional stages can add a real-sky image and/or a newly generated population of point-like and extended sources. Radial and PSF-like elliptical patterns can also be isolated from the input and added back to the simulation when requested.

## Requirements

- Python 3.
- Python packages: `numpy`, `astropy`, `scipy`, `matplotlib`, `opencv-python` (`cv2`) and `scikit-image` (`skimage`).
- Source Extractor / SExtractor, available from the command line as `source-extractor`.
- The Source Extractor configuration files expected by the script: `config_A.txt`, `default.param`, `default.nnw`, and the relevant Gaussian convolution filter file (for example, `gauss_2.5_5x5.conv`). These files must be in the folder supplied through `SEX_FOLDER`.

The script invokes shell commands such as `mkdir`, `rm`, and `source-extractor`, so run it in a Unix-like environment with Source Extractor on `PATH`. The output directory supplied with `WORKDIR` should already exist.

## Running NOISEMPIRE

A configuration file is mandatory. Run the program with:

```bash
python noisempire.py config_file.txt
```

All configuration parameters can also be specified on the command line. Non-empty command-line options override selected values in the configuration file.For example:

```bash
python noisempire.py config_file.txt --INPUT_IMAGE=/path/to/input_image.fits --DEBUG=False
```

To add a real-sky image to the simulated noise:

```bash
python noisempire.py config_file.txt --INPUT_IMAGE=/path/to/input_image.fits --REAL_SKY_IMAGE=/path/to/real_sky.fits --NSIGMA_RSKY=5 --DEBUG=False
```

List the available command-line arguments with:

```bash
python noisempire.py -h
# or
python noisempire.py --help
```

Check the program version with:

```bash
python noisempire.py --version
```

Read the configuration file before running the program. It contains the parameters needed for the run; command-line arguments are overrides, not a replacement for the configuration file.

## Processing overview

1. **Read and prepare the reference image.** FITS cubes with more than two axes are reduced to a 2D image by summing over the extra axes. With `CUTOUT=True`, the program can instead process a rectangular sub-image and adjust its WCS reference pixels.
2. **Identify problematic pixels and non-scientific areas.** NaNs, values repeated more than five times, and values below `MIN_VAL` or above `MAX_VAL` are masked during processing. A further check identifies uniform 3×3 pixel blocks, which can indicate areas outside the scientific field. The `NP_...` product has problematic pixels set to zero; internally, a separate mask is retained so those pixels can be excluded from calculations and restored in outputs.
3. **Model the input image.** Source Extractor is used to estimate the large-scale background and to identify compact, extended, and very extended source components or residuals. The script also estimates the small-scale background, local RMS map, high-frequency patterns and PSF-like/radial patterns. Source-affected holes in intermediate maps are patched to improve background estimation.
4. **Generate the synthetic noise/background.** The simulation reproduces the flux distribution and spatial structure of the reference noise. In version 1.0.4, PSF-scale noise and background structure are synthesized directly from the PSF/ACF reference rather than by convolving a random pixel image with it. ACF-based stamps are generated and pixel-scale artefacts are subsequently removed. The result is renormalized to reproduce the relevant input statistics and RMS variations.
5. **Apply optional additions.** Depending on the configuration, the program can simulate the large- and/or small-scale background instead of retaining the corresponding original component, restore isolated radial or elliptical patterns, add a real-sky image, and inject a simulated source population.
6. **Write requested products and clean up.** The `IMG_*` parameters select which component images are retained. Temporary and diagnostic files are normally removed unless they are requested or `DEBUG=True`.

## Main simulated products

The following products are controlled by the output parameters in the configuration file:

| Product | Contents |
| --- | --- |
| `<ALL_IMG_PREFIX>sim_0_<input_filename>` | Simulated noise/background without the input's detected source population. Radial and/or elliptical patterns are included if their restoration options are enabled. Controlled by `IMG_SIM`. |
| `sim_1_<ALL_IMG_PREFIX><input_filename>` | Simulated noise plus the optional real-sky image, convolved with the synthesized PSF and scaled using `NSIGMA_RSKY`. Requires `REAL_SKY_IMAGE` and the relevant output options. Controlled by `IMG_SIM1`. |
| `<ALL_IMG_PREFIX>sim_2_<input_filename>` | Simulated noise, any real-sky component added to the run, and the optional injected source population. Written when source injection is enabled and `IMG_SIM2=True`. |
| `<ALL_IMG_PREFIX>sim_2_input_cat_<input_basename>.txt` | Catalog of injected sources, written when `ADD_SIM_SOURCES=True`. Columns are RA, Dec, pixel X/Y, flux, significance, size in PSF units, axis ratio, and position angle (PA). |

`ALL_IMG_PREFIX` is applied to most output names. The current code places it before `sim_0_` and `sim_2_`, while the `sim_1_` product is named `sim_1_<ALL_IMG_PREFIX><input_filename>`, as shown in the table.

## Component and analysis images

These FITS images can be retained through the corresponding `IMG_*` options. The original input filename, including its FITS extension, is represented below by `<input_filename>`.

| Filename pattern | Contents |
| --- | --- |
| `FLAT_<input_filename>` | Input cube reduced to 2D, or a copy of an already 2D input (`IMG_FLC`). |
| `NP_<input_filename>` | Input image with problematic pixels set to zero (`IMG_NP`). |
| `bck1_<input_filename>` | Estimated large-scale background of the original image (`IMG_BCK1`). |
| `bck2_<input_filename>` | Estimated small-scale background of the original image (`IMG_BCK2`). |
| `Flat_noise_<input_filename>` | Original image after the modeled noise patterns have been subtracted (`IMG_FLAT_NOISE`). |
| `HFN_<input_filename>` | High-frequency noise patterns isolated from the original (`IMG_HFN`). |
| `PSF_sim_<input_filename>` | Synthesized PSF/beam based on the input image parameters (`IMG_PSF`). |
| `rms1_<input_filename>` | Estimated local RMS map (`IMG_RMS`). |
| `srcs_only1_<input_filename>` | First source-extraction stage, mainly compact/point-like sources (`IMG_SRCS1`). |
| `srcs_only2_<input_filename>` | Second source-extraction stage, targeting extended components (`IMG_SRCS2`). |
| `srcs_only3_<input_filename>` | Very extended source components and/or large-scale residuals (`IMG_SRCS3`). |
| `srcs_only_<input_filename>` | Combined image of all detected source components: compact, extended and very extended/residual components (`IMG_SRCS_ALL`). |
| `srcs_only_deep_<input_filename>` | Source-extraction image at `SIM_REF_DEPTH`, used as a reference when `ADD_SIM_SOURCES=True` and `SRCS_SIM_METHOD=AUTO` (`IMG_SRCS_4SIM`). |
| `Radial_Patterns_<input_filename>` | Isolated radial patterns (`IMG_RAD`). |
| `Elliptical_Patterns_<input_filename>` | Isolated PSF-like elliptical patterns (`IMG_ELL`). |
| `sim_bck1_<input_filename>` | Simulated large-scale background, only when `BCK1_TYPE=Simulate` and its output is requested (`IMG_SIM_BCK1`). |
| `sim_bck2_<input_filename>` | Simulated small-scale background, only when `BCK2_TYPE=Simulate` and its output is requested (`IMG_SIM_BCK2`). |

The run also creates intermediate products used for source subtraction, background estimation and normalization, for example `bck1_sub_...`, `bck1_HF_sub_...`, `nosrcs_bck1_HF_sub_...`, `nosrcs_bck3_HF_sub_...`, and `nosrcs_bck3_HF_REP_sub_...`. Some simulated intermediate maps, such as `sim_noise_and_bck_...` and `sim_noise_bck_HFN_...`, are intended mainly for diagnostics. Do not assume every intermediate filename will remain after a run: retained products depend on the `IMG_*` options and `DEBUG`.

## Important configuration parameters

- `INPUT_IMAGE`: path to the reference FITS image or cube.
- `WORKDIR`: existing directory for output products.
- `SEX_FOLDER`: directory containing Source Extractor configuration and filter files.
- `IMG_PARAMS`: `HEADER` to read pixel scale and beam parameters from the FITS header, or `USER` to use `USER_CDELT1`, `USER_CDELT2`, `USER_BMAJ`, `USER_BMIN`, and `USER_BPA`. The header keywords used in `HEADER` mode are controlled by `CDELT1_KEY`, `CDELT2_KEY`, `BMAJ_KEY`, `BMIN_KEY`, and `BPA_KEY`.
- `MIN_VAL` / `MAX_VAL`: optional lower and upper input-pixel limits; pixels outside these limits are treated as problematic.
- `CUTOUT`, `PIX_X_MIN`, `PIX_X_MAX`, `PIX_Y_MIN`, `PIX_Y_MAX`: enable processing of a sub-image. The maximum X/Y coordinates follow Python slice semantics and are excluded from the cutout.
- `BCK1_TYPE` / `BCK2_TYPE`: retain the original or simulate the large-scale and small-scale background (`Original` or `Simulate`).
- `NOISE_CONV`: choose the PSF, local ACF, global ACF, or combined PSF/local-ACF reference for the correlated-noise model (`PSF`, `LOCAL_ACF`, `ACF`, or `PSF_&_LOCAL_ACF`). Version 1.0.4 uses direct PSF/ACF-based synthesis rather than the former random-image convolution approach.
- `RAD_PATT_ISOLATE` / `ELL_PATT_ISOLATE`: isolate the radial and/or PSF-like elliptical patterns and add the isolated components back to the simulated image. `IMG_RAD` and `IMG_ELL` control retention of their component maps; they do not, by themselves, request pattern restoration.
- `REAL_SKY_IMAGE` and `NSIGMA_RSKY`: optionally add a real-sky FITS image and set the approximate significance of its brightest feature relative to the simulated noise. Supply a real-sky image with compatible spatial dimensions; the script does not reproject it onto the reference WCS.
- `ADD_SIM_SOURCES`: enable injection of a simulated source population.
- `SRCS_SIM_METHOD`: `AUTO` derives the simulated flux/size population from source extraction on the input; `USER` uses user-specified source-generation parameters. Parameters include `SIM_REF_DEPTH`, `N_SIM_SRCS`, `SIM_SRCS_SIGMA_MIN`, `SIM_SRCS_SIGMA_MAX`, `SIM_POWER_LAW_IDX`, `SIM_SRCS_SIZE_MIN`, `SIM_SRCS_SIZE_MAX`, and `SIM_SRCS_BETA`.
- `ALL_IMG_PREFIX`: prefix used for most output files.
- `SHOW_PLOT`: display diagnostic plots during processing.
- `DEBUG`: retain additional diagnostic/intermediate products. **Note:** the current code also calls `pdb.set_trace()` at the end when `DEBUG=True`, so it enters an interactive debugger rather than simply exiting. Use `DEBUG=False` for unattended/batch runs.

The general source-extraction depth and some background-subtraction scales are internal code parameters in this release. `SIM_REF_DEPTH` specifically controls the reference extraction used for the `AUTO` simulated-source workflow; it is not a general override for every extraction stage.

## Version history

### V1.0.4 — Belice

- Replaced the previous random-pixel-image convolution approach for PSF-scale noise and background simulation with direct synthesis using the reference PSF/ACF. The new ACF-stamp methods are `deterministic_stamp_from_acf()` and `random_stamp_from_acf()`; residual pixel-scale artefacts are removed afterward.
- Moved small-scale background (`BCK2`) computation before the high-frequency-pattern computation.
- Improved identification of very extended, low-surface-brightness sources as background excesses.
- Improved hole filling after source removal. In addition to copying patches from nearby image regions, `fill_image_holes()` can use patches generated with an ACF similar to that in the surrounding region or the global image.
- Deepened compact-source extraction from approximately 4σ to 2.5σ.
- Added a combined output map containing compact, extended and very extended/source-residual components (`srcs_only_...`).
- Added the option to process a smaller image cutout.
- Improved patching after source removal and subtraction of sources/residuals from large- and small-scale background maps.
- Added optional injection of point-like and extended PSF-shaped simulated sources, including `AUTO` and `USER` population-generation modes.
- Added a strategy for identifying non-scientific image areas and corrected minor bugs.

### V1.0.3 — Noce

- Fixed handling of odd-sized and rectangular images.
- Added masking of NaNs, repeated pixel values, and values outside `MIN_VAL` / `MAX_VAL`; the `NP_...` image records problematic pixels as zero.
- Allowed image pixel scale and beam parameters to be read from the FITS header or supplied by the user.
- Added `ALL_IMG_PREFIX` and user selection of retained output images; `DEBUG` controls additional diagnostic products.
- Renamed `IMG_ELL_PATT` to `IMG_ELL` and `IMG_RAD_PATT` to `IMG_RAD`.
- Added execution-time reporting and fixed minor bugs.

### V1.0.2

- Fixed a bug affecting cube flattening when extracting radial and elliptical patterns.

### V1.0.1

- Removed a duplicated import and misleading comments.
