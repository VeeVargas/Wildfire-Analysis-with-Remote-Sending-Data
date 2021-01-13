# Wildfire-Analysis-with-Remote-Sending-Data

## High-Level Overview
Current industry level wildfire models use computational fluid dynamics to simulate landscape-scale fires, typically on a 1m cubic voxel grid. While they do a great job at modeling the fires, the have two major limitations. First, they require an increduble amount of resources  to run. Second, even with supercomputers, they run slower than real time, making them useless for prediction.   

Forecasting fire size at the right time can help firefighters allocate resources and getting an area estimate of an ongoing fire can be very beneficial in preparing the mitigation strategies.

This project will assess the use of remote sensing technology and deep learning techniques to detect active fires and estimate they impact. Remote sensors provide a wealth of data on a global scale and can provide predictions at near real time.  

## Objectives and Constraints
There are two families of remote sensors; Passive sensors measure reflected energy emitted from the sun, whereas active sensors illuminate their target and measure its backscatter.  This project will only look as the application of passive sensors

In remote sensing, we can interpret features by looking at differences in spectral signatures for specific objects.  
Normalized Burn Ratio and Burn Area Index,have been employed to detect burned areas and assess the degree of impact of the disturbance usingdifferent satellite sensors

Three different types of resolution have to be considered:
* Spatial resolution
* Spectral resolution
* Temporal resolution


## Preliminary Assessment 
Spot the fire: The first part of this project will focus on spotting or detecting wildfires.
> Fire and non-fire labels were derived from the Terra MODIS Terra Thermal Anomalies & Fire product, which maps fires based on their pre- and post-fire infrared signals.

Predict the Spread: The second part will attempt to predict the the spread by forecasting its potential area coverage as soon as the fire is detected. (WIP-FUTURE)
> Target: Incorporated time series analysis in order to systematically forecast post-fire scars (burned area). Normalized Burn Ratio.

## Dataset Overview

* `Landsat 8`: multispectral, medium resolution, 11 bands, 15-100 meters, 16 day cycle.
* `Visible and Infrared Scanner (VIRS)`
* `Sentinel-2`
* `MODIS`: 1 to 2 days cycles 
> MOD13Q1 product, which consists of both the Normalized Difference Vegetation Index (NDVI) and the Enhanced Vegetation Index (EVI), 16 day cycle. 


The dataset consists of ecological, hydrological, and meteorological variables derived largely from remote sensing.  Values were  extracted for each variable at all pixels and scaled to equivalent spatial resolution to be represented by identical arrays of pixels.   

Addional processing tasks included:  
- cloud masking 
- Smoke removal
- Atmospheric correction
- upsampling the features (bilinear upsample)
- normalized_difference
- augmentations
 
**Features**
* `NDVI`(Normalized Difference Vegetation Index): Compares Near-Infrared (NIR) and Visible (VIS) bands, Landsat bands 5 and 4. 
* `EVI`(Enhanced Vegetation Index)
* `NBR`(Normalized Burn Ratio): target variable for post-fire scars. (FUTURE)
* `MIRBI`(Mid Infrared Burn Index) 
* `BAI`(Burn Area Index)
* `NDMI`(Normalized Difference Moisture Index)
* `LFMC` (live fuel moisture content): forest dryness, indicates the amount of water in fuels relative to their dry biomass
* `Topology` 
* `Fuel Load`
* `Land Surface Temp`

(WIP)








