# Coronal Field Extrapolation Dataset Generator

## Description

This dataset emulate the physics-based ADAPT-WSA PFSS model. The parameters to predict are spherical harmonic coefficients which represent the magnetic potential for a domain between the photosphere and te source surface (set to 2.51 Rs). 

## Project Structure

`make_train_test_index.py`: Make training and validation indeces from a directory holding the coronal field extrapolation benchmark data
`dataloader.py`: Dataset which can be used with PyTorch

## Features

For each timestamp, there 12 realization FITS files (labeled R000 - R011) identifying members of an ensemble. Each FITS file has five HDU's. We describe them below in zero-based order:

### HDU #0: Arrays over the Sphere
Merged array of shape (10, 90, 180), representing 10 arrays over spherical coordinates. The arrays are:

| Array Index | Description |
|-------------|-------------|
| 0 | Coronal field at outer boundary (nT) |
| 1 | Flux tube expansion factor eval. at the source surface |
| 2 | Colat. of open field footpoints at photosphere (rad) |
| 3 | Long. of open field footpoints (rad) (Take abs. value of Long and add carrLongitude HDU attribute to get correct value) |
| 4 | Photospheric Field (G) |
| 5 | Dist. from open field footpoint to nearest coronal bndry (deg) |
| 6 | Open (1,2,3) and closed (0) regions on the photosphere (1=in-to-out tracing; 2=out-to-in tracing; 3=both) |
| 7 | Dist. to current sheet at outer boundary |
| 8 | Coronal field at user defined radius (nT) |
| 9 | Squashing factor at outer boundary |

### HDU #1: Arrays over Subsatellite Track +/- 1 Grid Cell
Merged array of shape (8, 3, 180), representing data alond the L1 subsatellite track +/- one grid cell (dimension of 3).The dimension 8 arrays are:

| Array Index | Description |
|-------------|-------------|
| 0 | Coronal field at outer boundary (nT) |
| 1 | Flux-tube expansion factor eval. at the source surface |
| 2 | Colat. of open field footpoints at photosphere (rad) |
| 3 | Long. of open field footpoints at photosphere (rad) |
| 4 | Photospheric field (G) |
| 5 | Dist. from open field footpoint to nearest coronal bndry (deg) |
| 6 | Dist. from subsatellite point to current sheet (deg) |
| 7 | Squashing factor eval. at outer boundary |

## Usage

## Requirements


## Contact
Daniel da Silva, [daniel.e.dasilva@nasa.gov](daniel.e.dasilva@nasa.gov)
