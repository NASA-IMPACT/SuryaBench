<!---- Provide an overview of what is being achieved in this repo ----> 

# SuryaBench


**SuryaBench** is a standardized test benchmark for evaluating the performance of foundational AI models in Heliophysics and Space Weather Science across various tasks.  

The complete dataset along with input files and application-specific instances is available through the Hugging Face platform at [nasa-impact/SuryaBench]((https://huggingface.co/collections/nasa-impact/suryabench-68265ce306fc2470c121af7b)).

## Application Dataset Details

SuryaBench contains application datasets related to six key tasks that are structured as prediction, segmentation, or reconstruction. These tasks are related to various aspects of solar and heliospheric events ecosystem and are listed below:
1. **[Active Region Segmentation](ar_segmentation/)**
2. **[Active Region Emergence Prediction](ar_emergence/)**
3. **[Coronal Field Extrapolation](CoronalExtrapolation/)**
4. **[Solar Flare Forecasting](flare/)**
5. **[Solar Wind Forecasting](solarwind/)**
6. **[Solar EUV Spectra Modeling](euv_spectra/)**

### Short Descriptions of Applications and Code

## 1. [Active Region Segmentation](ar_segmentation/)
This collection includes segmentation maps of active regions with magnetic polarity inversion lines (also referred to as ARPILs). The ARPIL data generation is applied on line of sight component of the solar magnetic field. 
<!-- The ARPIL segmentation maps are created using the following steps:
 - Generate two polarity maps (positive and negative polarity regions) with field strength thresholds of +50 and −50 Gauss, respectively. 
 - To remove small, noisy patches in polarity region maps, apply a size filter excluding regions
>100 px (approx. 13.3 Mm2 of photospheric area). 
 - Dilate the polarity regions (10 px). 
 - Identify the intersection of the dilated positive and negative polarity regions, which corresponds to areas containing PILs. 
 - Only ARs that include PILs are kept.  -->
The code is designed to process magnetogram data efficiently in C++. Various libraries including OpenCV, CUDA for efficient image processing, and CFITS10 and HDF5 for data I/O. Refer to the application's [README](ar_segmentation/README.md) for more details.


## 2. [Active Region Emergence Prediction](ar_emergence/)
Building early forecasts of the manifestation of solar activity, specifically active regions is crucial to mitigate the impact of space weather disturbances. For this application, 50 active region emergence instances that appear on the solar surface within 30 deg of central meridian are selected. The selected active region instances are persisted for more than four days and include helioseismic and magnetic information (magnetic flux, Doppler velocity, and continuum intensity). Refer to the application's [README](ar_emergence/README.md) for more details.

## 3. [Coronal Field Extrapolation](CoronalExtrapolation/)
 The solar corona is a highly magnetic environment, characterized by complex and time-varying magnetic fields. A common task for scientists studying the Sun is to model the 3D magnetic field.  This application aims to evaluate the effectiveness of AI foundation models to emulate the physics-based ADAPT-WSA PFSS model. The parameters to predict are spherical harmonic coefficients which represent the magnetic potential for a domain between the photosphere and te source surface (set to 2.51 Rs). Refer to the application's [README](CoronalExtrapolation/README.md) for more details.

## 4. [Solar Flare Forecasting](flare/)
This application focuses on a key space weather prediction task -- solar flare forecasting. The provided script generates labeled solar flare time-series data using a rolling window approach. The output includes binary labels based on both the maximum flare class and cumulative flare intensity within a given time window. Refer to the application's [README](flare/README.md) for more details.

## 5. [Solar Wind Forecasting](solarwind/)
The solar wind is a stream of particles that emanates from the Sun. The interaction of solar wind with Earth's magnetic field results in the formation of near-Earth space weather. Solar wind interactions are well-known to drive geomagnetic storms, wherein the Earth's magnetic field is perturbed, causing induction of currents into systems including satellites, power grids, oil pipelines, leading to potential socio-economic impacts. This application contains generation code for solar wind forecasting labels, specifically scripts for downloading the solar wind data from OMNI and provide splits. Refer to the application's [README](solarwind/README.md) for more details.

## 6. [Solar EUV Spectra Modeling](euv_spectra/)
The solar Extreme Ultraviolet (EUV) spectral irradiance is one of the most influential radiative component emitted from the chromosphere and corona, with wavelengths ranging from approximately 5 to 120 nm. These emissions play an important role in controlling the thermodynamic state of the Earth's upper atmosphere, governing processes such as photoionization, photodissociation, and heating in the thermosphere and ionosphere. Fluctuations in EUV irradiance, whether driven by the 11-year solar cycle, 27-day solar rotation, or impulsive flare events, introduce significant variability in space weather, influencing satellite drag, GPS signal accuracy, and communications infrastructure. This application contains generation code for high-energy EUV spectra from SDO/EVE (MEGS-A) with a 10s cadence. Refer to the application's [README](euv_spectra/README.md) for more details.


## Citation
**BibTeX:**
```
@misc{suryabench,
      title={SuryaBench: Benchmark Dataset for Advancing Machine Learning in Heliophysics and Space Weather Prediction}, 
      author={ Sujit Roy and 
               Dinesha Vasanta Hegde and
               Johannes Schmude and
               Amy Lin and
               Vishal Gaur and
               Talwinder Singh and
               Rohit Lal and 
               Andrés Muñoz-Jaramillo and 
               Kang Yang and 
               Chetraj Pandey and
               Jinsu Hong and 
               Berkay Aydin and 
               Ryan McGranaghan and 
               Spiridon Kasapis and 
               Vishal Upendran and 
               Shah Bahauddin and 
               Daniel da Silva and 
               Marcus Freitag and 
               Nikolai Pogorelov and
               Manil Maskey and 
               Rahul Ramachandran},
      year={2025}
}
```

## Dataset Authors
- [Sujit Roy](https://github.com/thesujitroy)*
- [Dinesha Vasanta Hegde](https://github.com/dinesh-hegde)*
- Johannes Schmude 
- Amy Lin 
- Vishal Gaur 
- Talwinder Singh 
- Rohit Lal  
- Andrés Muñoz-Jaramillo  
- Kang Yang  
- Chetraj Pandey 
- Jinsu Hong  
- Berkay Aydin  
- Ryan McGranaghan  
- Spiridon Kasapis  
- Vishal Upendran  
- Shah Bahauddin  
- Daniel da Silva  
- Marcus Freitag  
- Nikolai Pogorelov 
- Manil Maskey  
- Rahul Ramachandran

## Application Dataset Contacts
For each task, please contact:
- **[Active Region Segmentation](ar_segmentation/)** - [Jinsu Hong](https://github.com/JinsuHongg)
- **[Active Region Emergence Prediction](ar_emergence/)** - [Spyros Kasapis](https://github.com/skasapis)
- **[Coronal Field Extrapolation](CoronalExtrapolation/)** - [Daniel da Silva](https://github.com/ddasilva)
- **[Solar Flare Forecasting](flare/)** - [Jinsu Hong](https://github.com/JinsuHongg)
- **[Solar Wind Forecasting](solarwind/)** - [Vishal Upendran](https://github.com/Vishal-Upendran)
- **[Solar EUV Spectra Modeling](euv_spectra/)** - [Shah Bahauddin](shah.bahauddin@lasp.colorado.edu)
