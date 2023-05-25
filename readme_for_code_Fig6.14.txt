##########################################################################
# ---------------------------------------------------------------------------------------------------------------------
# This is Python code to produce IPCC AR6 WGI Figure 6.14
# Creator: Steven Turnock, Met Office Hadley Centre, UK 
# Contact: steven.turnock@metoffice.gov.uk
# Last updated on: May 20th, 2021
# --------------------------------------------------------------------------------------------------------------------
#
# - Code functionality: 
#   The script reads in 2D surface O3 concentrations from different CMIP6 models at different temperature threshold levels in two different scenarios
#   to produce a 2D contour plot of the multi-model difference in surface O3 concentrations betwween the two scenarios at each temperature level
# - Input data:
#   Original data files used in plotting can be found on the ESGF but the data files directly relevant to this figure can be found  
#   Processed data files used in this script can be found on the CEDA archive at doi:10.1017/9781009157896.008 and are as follows:
#   datafile: ‘MRI-ESM2-0_monthly_annual_seasonal_mean_surf_o3_for_CMIP6_ssp370pdSST_2015_2099_all.nc’
#   datafile: ‘MRI-ESM2-0_monthly_annual_seasonal_mean_surf_o3_for_CMIP6_ssp370SST_2015_2099_all.nc’
#   datafile: ‘GFDL-ESM4_monthly_annual_seasonal_mean_surf_o3_for_CMIP6_ssp370pdSST_2015_2099_all.nc’
#   datafile: ‘GFDL-ESM4_monthly_annual_seasonal_mean_surf_o3_for_CMIP6_ssp370SST_2015_2099_all.nc’
#   datafile: ‘GISS-E2-1-G_monthly_annual_seasonal_mean_surf_o3_for_CMIP6_ssp370pdSST_2015_2099_all.nc’
#   datafile: ‘GISS-E2-1-G_monthly_annual_seasonal_mean_surf_o3_for_CMIP6_ssp370SST_2015_2099_all.nc’
#   datafile: ‘UKESM1-0-LL_monthly_annual_seasonal_mean_surf_o3_for_CMIP6_ssp370pdSST_2015_2099_all.nc’
#   datafile: ‘UKESM1-0-LL_monthly_annual_seasonal_mean_surf_o3_for_CMIP6_ssp370pdSST_2015_2099_all.nc’
# - Output variables: 
#   The code plots the figure 6.14 in IPCC AR6 WGI Chapter 6 
#
# ----------------------------------------------------------------------------------------------------
# Information on  the software used
# - Software Version: Python3.6
# - Landing page to access the software: https://www.python.org/downloads/release/python-360/
# - Operating System: N/A
# - Environment required to compile and run: No specific environment required but uses Python packages: Iris, Numpy, cartopy and Matplotlib
#  ----------------------------------------------------------------------------------------------------
#
#  License: Apache 2.0
# ----------------------------------------------------------------------------------------------------
# How to cite:
# When citing this code, please include both the code citation and the following citation for the related report component:
########################################################################