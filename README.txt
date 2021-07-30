
Scripts for: 'Increased probability of hot and dry weather extremes during the growing season threatens global crop yields'
By: Matias Heino1, Pekka Kinnunen, Weston Anderson, Deepak Ray, Michael J. Puma, Olli Varis, Stefan Siebert, Matti Kummu

(Paper submitted)

The analysis is composed of eight Python scripts:

1) agmerra_data_mpi.py
  - Import and process temperature and precipitation data from
    AgMerra (1). The script produces growing season average precipitation,
	average temperature, as well as growing season temperature histograms
	for each crop (wheat, maize, rice, soybean) and grid cell.

2) soil_moisture_data_ERA5_mpi.py
  - Import and process soil moisture data from ERA5 (2). The script produces
    growing season soil moisture histograms for each crop and grid cell.

3) soil_moisture_data_gleam_mpi.py
  - Import and process soil moisture data from GLEAM (3). The script produces
    growing season soil moisture histograms for each crop and grid cell.

4) standardize_soil_moisture_mpi.py
  - Transform the soil moisture data to soil moisture deficit, and conduct a min-max
    scaling. Also, calculate and export average growing season soil moisture.
  
5) climate_bin_analysis_raster_mpi.py
  - Import crop yield data (4), and further process the climate data. Run the XGBoost
    regression. Exports model performance as well as
	partial dependence results.
  
6) climate_trend.py
  - Calculates historical trends in hot-cold and cold-wet events.
  
7) visualize_results.py
  - Visualize the numerical results produced in the above scripts.
  
8) general_functions.py
  - A set of functions used in many of the other scripts.


Most of the scripts were run in a cluster and MPI is required for them
scripts to run properly in managable time. Further, although the script run with
a coherent file structure, they require all the data to be downloaded separately.

References:
(1) Ruane, Alex C., Richard Goldberg, and James Chryssanthacopoulos.
    "Climate forcing datasets for agricultural modeling: Merged products
    for gap-filling and historical climate series estimation."
	Agricultural and Forest Meteorology 200 (2015): 233-248.

(2) Hersbach, Hans, et al. "The ERA5 global reanalysis."
    Quarterly Journal of the Royal Meteorological Society 146.730 (2020): 1999-2049.
	
(3) Martens, Brecht, et al. "GLEAM v3: Satellite-based land evaporation and root-zone
    soil moisture." Geoscientific Model Development 10.5 (2017): 1903-1925.
	
(4) Ray, Deepak K., et al. "Climate change has likely already affected global
    food production." PloS one 14.5 (2019): e0217148.