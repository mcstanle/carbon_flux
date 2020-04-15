# --- import modules ---
from mpl_toolkits.basemap import Basemap, cm
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from when_TCCON_measurements import is_cloudy_many_dir
import datetime
from netCDF4 import Dataset
import glob, os

# These are not used
month_as_day_per_year = np.array([0,31,31+28,31+28+31,31+28+31+30,31+28+31+30+31,31+28+31+30+31+30,31+28+31+30+31+30+31,31+28+31+30+31+30+31+31,31+28+31+30+31+30+31+31+30,31+28+31+30+31+30+31+31+30+31,31+28+31+30+31+30+31+31+30+31+30])
GC_plevels = np.array([0.010,0.066,0.211,0.617,1.651,4.077,9.293,19.792,28.368,40.175,56.388,78.512,92.366,108.663,127.837,150.393,176.930,208.244,245.246,288.927,339.005,377.070,415.155,453.269,491.401,529.550,567.706,605.880,644.054,682.239,707.699,733.160,758.621,784.088,809.556,829.929,845.211,860.493,875.776,891.059,906.342,921.626,936.911,952.195,967.480,982.765,998.051,1013.250])
GC_players = np.array([0.038,0.139,0.414,1.134,2.864,6.685,14.542,24.080,34.272,48.282,67.450,85.439,100.514,118.250,139.115,163.661,192.587,226.745,267.087,313.966,358.038,396.112,434.212,472.335,510.475,548.628,586.793,624.967,663.146,694.969,720.429,745.890,771.354,796.822,819.743,837.570,852.852,868.135,883.418,898.701,913.984,929.268,944.553,959.837,975.122,990.408,1005.650])

# Path to ACOS OCO-2 Files
#dir_path = '/users/jk/13/bbyrne/GEOSchem_adjoint/OCO2_data/'
dir_path = '/data/ctm/g350/'
os.chdir(dir_path)
aaa = glob.glob("*_2014*.nc")
aaa.extend(glob.glob("*_2015*.nc"))
#aaa.extend(glob.glob("*_2011*.nc"))
#aaa.extend(glob.glob("*_2012*.nc"))
#aaa.extend(glob.glob("*_2013*.nc"))
for file in aaa:
    print(file)
    completePath =dir_path+file
    f=Dataset(completePath,mode='r')
    latitude_day=f.variables['latitude'][:]
    longitude_day=f.variables['longitude'][:]
    time_day=f.variables['time'][:]
    quality_flag_day=f.variables['xco2_quality_flag'][:]
    soundings=f.groups['Sounding']
    retrieval=f.groups['Retrieval']
    gain_day=soundings.variables['gain'][:]
    warn_level_day=retrieval.variables['warn_level'][:]
    sounding_id_day=f.variables['sounding_id'][:]
    #sounding_id_day = np.floor(sounding_id_day_temp/100)
    unc_day=f.variables['xco2_uncertainty'][:]
    column_averaging_kernel=f.variables['xco2_averaging_kernel'][:] # The normalized column averging kernel for the retrieved XCO2 (dimensionless).
    pressure_levels=f.variables['pressure_levels'][:] #The retrieval pressure level grid for each sounding in hPa. Note that is simply equal to SigmaB multiplied by the surface pressure.
    # -----
    xco2_day=f.variables['xco2'][:] # units = "ppm"
    xco2_unc_day=f.variables['xco2_uncertainty'][:]
    co2_profile_apriori_day=f.variables['co2_profile_apriori'][:]
    pwf_day=f.variables['pressure_weight'][:]
    retrieval=f.groups['Retrieval']
    psurf_day=retrieval.variables['psurf'][:] # Surface pressure retrieved by the Level-2 retrieval 

    # --- DATA FILTERING ---
    latitude_day_good=latitude_day[np.where(quality_flag_day == 0)]
    longitude_day_good=longitude_day[np.where(quality_flag_day == 0)]
    time_day_good=time_day[np.where(quality_flag_day == 0)]
    quality_flag_day_good=quality_flag_day[np.where(quality_flag_day == 0)]
    sounding_id_day_good=sounding_id_day[np.where(quality_flag_day == 0)]
    warn_level_day_good=warn_level_day[np.where(quality_flag_day == 0)]   
    gain_day_good=gain_day[np.where(quality_flag_day == 0)]   
    unc_day_good=unc_day[np.where(quality_flag_day == 0)]  
    column_averaging_kernel_good=column_averaging_kernel[np.where(quality_flag_day == 0)]   
    pressure_levels_good=pressure_levels[np.where(quality_flag_day == 0)]
    xco2_good = xco2_day[np.where(quality_flag_day == 0)]
    xco2_unc_good = xco2_unc_day[np.where(quality_flag_day == 0)]
    pwf_good = pwf_day[np.where(quality_flag_day == 0)]
    psurf_good = psurf_day[np.where(quality_flag_day == 0)]
    co2_profile_apriori_good = co2_profile_apriori_day[np.where(quality_flag_day == 0)]
    # --- WL <= 15 ---
    latitude_day_good2=latitude_day_good[np.where(warn_level_day_good <= 15)]
    longitude_day_good2=longitude_day_good[np.where(warn_level_day_good <= 15)]
    time_day_good2=time_day_good[np.where(warn_level_day_good <= 15)]
    quality_flag_day_good2=quality_flag_day_good[np.where(warn_level_day_good <= 15)]
    sounding_id_day_good2=sounding_id_day_good[np.where(warn_level_day_good <= 15)]
    gain_day_good2=gain_day_good[np.where(warn_level_day_good <= 15)]
    unc_day_good2=unc_day_good[np.where(warn_level_day_good <= 15)]
    column_averaging_kernel_good2=column_averaging_kernel[np.where(warn_level_day_good <= 15)]   
    pressure_levels_good2=pressure_levels[np.where(warn_level_day_good <= 15)] 
    xco2_good2 = xco2_good[np.where(warn_level_day_good <= 15)]
    xco2_unc_good2 = xco2_unc_good[np.where(warn_level_day_good <= 15)]
    pwf_good2 = pwf_good[np.where(warn_level_day_good <= 15)]
    psurf_good2 = psurf_good[np.where(warn_level_day_good <= 15)]
    co2_profile_apriori_good2 = co2_profile_apriori_good[np.where(warn_level_day_good <= 15)]
    # --- Use M-Gain ---
    latitude_day_good1=latitude_day_good2#[np.where(gain_day_good2 != 'M')]
    longitude_day_good1=longitude_day_good2#[np.where(gain_day_good2 != 'M')]
    time_day_good1=time_day_good2#[np.where(gain_day_good2 != 'M')]
    quality_flag_day_good1=quality_flag_day_good2#[np.where(gain_day_good2 != 'M')]
    sounding_id_day_good1=sounding_id_day_good2#[np.where(gain_day_good2 != 'M')]
    unc_day_good1=unc_day_good2#[np.where(gain_day_good2 != 'M')]
    column_averaging_kernel_good1=column_averaging_kernel_good2#[np.where(gain_day_good2 != 'M')]
    pressure_levels_good1=pressure_levels_good2#[np.where(gain_day_good2 != 'M')]
    xco2_good1 = xco2_good2#[np.where(gain_day_good2 != 'M')]
    xco2_unc_good1 = xco2_unc_good2#[np.where(gain_day_good2 != 'M')]
    pwf_good1 = pwf_good2#[np.where(gain_day_good2 != 'M')]
    psurf_good1 = psurf_good2#[np.where(gain_day_good2 != 'M')]
    co2_profile_apriori_good1 = co2_profile_apriori_good2#[np.where(gain_day_good2 != 'M')]
    # ------------

    
    # ---------- Append daily data to array
    if 'latitude' in locals():   
        latitude = np.append(latitude,latitude_day_good1)
        longitude = np.append(longitude,longitude_day_good1)
        time = np.append(time,time_day_good1)
        sounding_id = np.append(sounding_id,sounding_id_day_good1)
        unc = np.append(unc,unc_day_good1)
        averaging_kernel = np.append(averaging_kernel,column_averaging_kernel_good1, axis=0)
        pressure = np.append(pressure,pressure_levels_good1, axis=0)
        xco2 = np.append(xco2, xco2_good1)
        xco2_unc = np.append( xco2_unc, xco2_unc_good1)
        pwf = np.append(pwf,pwf_good1, axis=0)
        psurf = np.append(psurf,psurf_good1)
        co2_profile_apriori = np.append(co2_profile_apriori,co2_profile_apriori_good1, axis=0)
    else:
        latitude = latitude_day_good1
        longitude = longitude_day_good1
        time = time_day_good1
        sounding_id = sounding_id_day_good1
        unc = unc_day_good1
        averaging_kernel = column_averaging_kernel_good1
        pressure = pressure_levels_good1
        xco2 = xco2_good1
        xco2_unc = xco2_unc_good1
        pwf = pwf_good1
        psurf = psurf_good1
        co2_profile_apriori = co2_profile_apriori_good1
        
 #   print 'mean unc ' + str(np.mean(unc_day_good1))+ ' min unc ' + str(unc_day_good1.min()) +' max unc ' + str(unc_day_good1.max())
    


# time period of interest
time_min = 20040101000000
time_max = 20160101000000

# Get rid of observations outside time period of interest
latitude_1 = latitude[np.where(sounding_id >= time_min)]
longitude_1 = longitude[np.where(sounding_id >= time_min)]
time_1 = time[np.where(sounding_id >= time_min)]
unc_1 = unc[np.where(sounding_id >= time_min)]
averaging_kernel_1 = averaging_kernel[np.where(sounding_id >= time_min)]
pressure_1 = pressure[np.where(sounding_id >= time_min)]
xco2_1 = xco2[np.where(sounding_id >= time_min)]
pwf_1 = pwf[np.where(sounding_id >= time_min)]
psurf_1 = psurf[np.where(sounding_id >= time_min)]
co2_profile_apriori_1 = co2_profile_apriori[np.where(sounding_id >= time_min)]
sounding_id_1 = sounding_id[np.where(sounding_id >= time_min)]

latitude_2 = latitude_1[np.where(sounding_id_1 < time_max)]
longitude_2 = longitude_1[np.where(sounding_id_1 < time_max)]
time_2 = time_1[np.where(sounding_id_1 < time_max)]
unc_2 = unc_1[np.where(sounding_id_1 < time_max)]
averaging_kernel_2 = averaging_kernel_1[np.where(sounding_id_1 < time_max)]
pressure_2 = pressure_1[np.where(sounding_id_1 < time_max)]
xco2_2 = xco2_1[np.where(sounding_id_1 < time_max)]
pwf_2 = pwf_1[np.where(sounding_id_1 < time_max)]
psurf_2 = psurf_1[np.where(sounding_id_1 < time_max)]
co2_profile_apriori_2 = co2_profile_apriori_1[np.where(sounding_id_1 < time_max)]
sounding_id_2 = sounding_id_1[np.where(sounding_id_1 < time_max)]

# -- GEOS-Chem Grids --
#XMid = np.arange(-180, 177.5, 2.5)
#YMid = np.append(-89.5,np.arange(-88, 88, 2),89.5)
#XMid = np.arange(-180, 178, 2.5)
#YMid = np.append(np.append(-89.5,np.arange(-88, 89, 2)),89.5)
XMid = np.arange(-180, 178, 5)
YMid = np.append(np.append(-89.,np.arange(-86, 89, 4)),89.)

# Map obs to Grid cells
X_ind = np.ones(len(latitude_2))*1.0e5
Y_ind = np.ones(len(latitude_2))*1.0e5
for index in enumerate(time_2,start=0):
    X_ind[int(index[0])]=np.argmin(np.abs(XMid-longitude_2[int(index[0])]))+1
    Y_ind[int(index[0])]=np.argmin(np.abs(YMid-latitude_2[int(index[0])]))+1

Z_ind = Y_ind*1000+X_ind # Array of unique values for both x and y.

# ---- Aggregate to 1 observation/Grid/hour ----
temp_arr=np.unique(np.floor(sounding_id_2/10000))*10000
for sarr in temp_arr:
   # print sarr
    ind = np.where(np.logical_and(sounding_id_2>=sarr, sounding_id_2<sarr+10000)) # select 1 hour time range
    Z_at_time = Z_ind[ind] # limit to time period of interest
    Y_at_time = Y_ind[ind]
    X_at_time = X_ind[ind]
    unc_at_time = unc_2[ind]
    xco2_at_time = xco2_2[ind]
    longitude_at_time = longitude_2[ind]
    latitude_at_time = latitude_2[ind]
    psurf_at_time = psurf_2[ind]
    pressure_at_time = pressure_2[ind[0],:]
    pwf_at_time = pwf_2[ind[0],:]
    averaging_kernel_at_time = averaging_kernel_2[ind[0],:]
    co2_profile_apriori_at_time = co2_profile_apriori_2[ind[0],:]
    sounding_id_at_time = sounding_id_2[ind]

    uuu, indices = np.unique(Z_at_time, return_index=True) # Find single obs for each hour and grid
    # Loop over aggregated observations
    if len(ind[0])>0:
        ii=ind[0][0]+indices
        if len(ii)>0:
            ii_ind=0
            for vs in uuu:
                unc_temp = unc_at_time[np.where(Z_at_time == vs)] # uncertaitnies for time and grid cell
                unc_temp = unc_temp.astype(float)
                unc_temp = unc_temp[np.where(np.logical_and(np.isfinite(unc_temp),(unc_temp > 0.1)))]
                if unc_temp.size>0:
                    unc_super = (0.6**2+1.7**2/unc_temp.size)**0.5 # From Kulawick discussion paper abstract
                    #AK_temp = np.flipud(averaging_kernel[ii[ii_ind]])
                    if 'unc_unique' in locals():   
                        unc_unique = np.append(unc_unique,unc_super)
                        Y_ind_unique = np.append(Y_ind_unique,Y_at_time[indices[ii_ind]])
                        X_ind_unique = np.append(X_ind_unique,X_at_time[indices[ii_ind]])
                        lon_unique = np.append(lon_unique,np.mean(longitude_at_time[np.where(Z_at_time == vs)]))
                        lat_unique = np.append(lat_unique,np.mean(latitude_at_time[np.where(Z_at_time == vs)]))
                        out_xco2 = np.append(out_xco2, np.mean(xco2_at_time[np.where(Z_at_time == vs)]))
                        out_psurf = np.append(out_psurf,np.mean(psurf_at_time[np.where(Z_at_time == vs)]))
                        out_P = np.append(out_P,np.expand_dims(np.flipud(np.mean(pressure_at_time[np.where(Z_at_time == vs)[0]],axis=0)), axis=0), axis=0)
                        out_pwf = np.append(out_pwf,np.expand_dims(np.flipud(np.mean(pwf_at_time[np.where(Z_at_time == vs)[0]],axis=0)), axis=0), axis=0)
                        out_AK = np.append(out_AK,np.expand_dims(np.flipud(np.mean(averaging_kernel_at_time[np.where(Z_at_time == vs)[0]],axis=0)), axis=0), axis=0)
                        sounding_id_unique = np.append(sounding_id_unique,sounding_id_at_time[indices[ii_ind]])
                        out_xa = np.append(out_xa,np.expand_dims(np.flipud(np.mean(co2_profile_apriori_at_time[np.where(Z_at_time == vs)[0]],axis=0)), axis=0), axis=0)
                        num_obs = np.append(num_obs,unc_temp.size)
                    else:
                        unc_unique = unc_super
                        Y_ind_unique = Y_at_time[indices[ii_ind]]
                        X_ind_unique = X_at_time[indices[ii_ind]]
                        lon_unique = np.mean(longitude_at_time[np.where(Z_at_time == vs)])
                        lat_unique = np.mean(latitude_at_time[np.where(Z_at_time == vs)])
                        out_xco2 = np.mean(xco2_at_time[np.where(Z_at_time == vs)])#xco2[ii[ii_ind]]
                        out_psurf = np.mean(psurf_at_time[np.where(Z_at_time == vs)])
                        out_P = np.expand_dims(np.flipud(np.mean(pressure_at_time[np.where(Z_at_time == vs)[0]],axis=0)), axis=0)
                        out_pwf =np.expand_dims(np.flipud(np.mean(pwf_at_time[np.where(Z_at_time == vs)[0]],axis=0)), axis=0)
                        out_AK = np.expand_dims(np.flipud(np.mean(averaging_kernel_at_time[np.where(Z_at_time == vs)[0]],axis=0)), axis=0)
                        sounding_id_unique = sounding_id_at_time[indices[ii_ind]]
                        out_xa = np.expand_dims(np.flipud(np.mean(co2_profile_apriori_at_time[np.where(Z_at_time == vs)[0]],axis=0)), axis=0)
                        num_obs = unc_temp.size
                ii_ind += 1

# ---- Write out aggregated obsrvations into daily files ----
sounding_id_str=["%.0f" % (x) for x in sounding_id_unique]

temp_arr2oo=np.unique(np.floor(sounding_id/10000000))*10000000
dailyXCO2 = np.ones(len(temp_arr2oo))*0
kjkj =0
for sarr in temp_arr2oo:
    IIIst = np.where(np.logical_and(sounding_id_unique>=sarr, sounding_id_unique<sarr+10000000))
    IIIs = IIIst[0]
    #
    TEMP_lon = lon_unique[IIIs]
    TEMP_lat = lat_unique[IIIs]
    TEMP_xco2 = out_xco2[IIIs]
    xco2_inds = np.where( np.logical_and(TEMP_lat>=40, TEMP_lat<=90) )
    dailyXCO2[kjkj]= np.mean(TEMP_xco2[xco2_inds])
    kjkj=kjkj+1
fig = plt.figure()
plt.plot(dailyXCO2)
plt.show()

temp_arr2=np.unique(np.floor(sounding_id/1000000))*1000000
for sarr in temp_arr2:
    IIIst = np.where(np.logical_and(sounding_id_unique>=sarr, sounding_id_unique<sarr+1000000))
    IIIs = IIIst[0]
    fname = '/users/jk/13/bbyrne/python_codes/gosat35_4x5_' + str(int(sarr/1000000)) + '.txt'
    with open(fname,'wb') as csvfile:
        spamwriter = csv.writer(csvfile,delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        iii=0
        while iii < len(IIIs):
           # print fname + ': ' + str(iii+1) + '/' + str(len(IIIs))
#            print str(iii) + '/' + str(len(sounding_id_unique))
            if (np.isfinite(unc_unique[iii]) and (unc_unique[iii] != 0)):
                spamwriter.writerow(['20',('%.4E' % lon_unique[IIIs[iii]]),('%.4E' % lat_unique[IIIs[iii]]),('%.4E' % out_psurf[IIIs[iii]]),('%.4E' % out_xco2[IIIs[iii]]),('%.4E' % unc_unique[IIIs[iii]]),sounding_id_str[IIIs[iii]][0:4],sounding_id_str[IIIs[iii]][4:6],sounding_id_str[IIIs[iii]][6:8],sounding_id_str[IIIs[iii]][8:10],sounding_id_str[IIIs[iii]][10:12],sounding_id_str[IIIs[iii]][12:14],sounding_id_str[IIIs[iii]][0:14]])
                spamwriter.writerow([('%.4E' % out_AK[IIIs[iii],19]),('%.4E' % out_AK[IIIs[iii],18]),('%.4E' % out_AK[IIIs[iii],17]),('%.4E' % out_AK[IIIs[iii],16]),('%.4E' % out_AK[IIIs[iii],15]),
                                     ('%.4E' % out_AK[IIIs[iii],14]),('%.4E' % out_AK[IIIs[iii],13]),('%.4E' % out_AK[IIIs[iii],12]),('%.4E' % out_AK[IIIs[iii],11]),('%.4E' % out_AK[IIIs[iii],10]),
                                     ('%.4E' % out_AK[IIIs[iii],9]),('%.4E' % out_AK[IIIs[iii],8]),('%.4E' % out_AK[IIIs[iii],7]),('%.4E' % out_AK[IIIs[iii],6]),('%.4E' % out_AK[IIIs[iii],5]),
                                     ('%.4E' % out_AK[IIIs[iii],4]),('%.4E' % out_AK[IIIs[iii],3]),('%.4E' % out_AK[IIIs[iii],2]),('%.4E' % out_AK[IIIs[iii],1]),('%.4E' % out_AK[IIIs[iii],0])])

                spamwriter.writerow([('%.4E' % out_pwf[IIIs[iii],19]),('%.4E' % out_pwf[IIIs[iii],18]),('%.4E' % out_pwf[IIIs[iii],17]),('%.4E' % out_pwf[IIIs[iii],16]),('%.4E' % out_pwf[IIIs[iii],15]),
                                     ('%.4E' % out_pwf[IIIs[iii],14]),('%.4E' % out_pwf[IIIs[iii],13]),('%.4E' % out_pwf[IIIs[iii],12]),('%.4E' % out_pwf[IIIs[iii],11]),('%.4E' % out_pwf[IIIs[iii],10]),
                                     ('%.4E' % out_pwf[IIIs[iii],9]),('%.4E' % out_pwf[IIIs[iii],8]),('%.4E' % out_pwf[IIIs[iii],7]),('%.4E' % out_pwf[IIIs[iii],6]),('%.4E' % out_pwf[IIIs[iii],5]),
                                     ('%.4E' % out_pwf[IIIs[iii],4]),('%.4E' % out_pwf[IIIs[iii],3]),('%.4E' % out_pwf[IIIs[iii],2]),('%.4E' % out_pwf[IIIs[iii],1]),('%.4E' % out_pwf[IIIs[iii],0])])

                spamwriter.writerow([('%.4E' % out_xa[IIIs[iii],19]),('%.4E' % out_xa[IIIs[iii],18]),('%.4E' % out_xa[IIIs[iii],17]),('%.4E' % out_xa[IIIs[iii],16]),('%.4E' % out_xa[IIIs[iii],15]),
                                     ('%.4E' % out_xa[IIIs[iii],14]),('%.4E' % out_xa[IIIs[iii],13]),('%.4E' % out_xa[IIIs[iii],12]),('%.4E' % out_xa[IIIs[iii],11]),('%.4E' % out_xa[IIIs[iii],10]),
                                     ('%.4E' % out_xa[IIIs[iii],9]),('%.4E' % out_xa[IIIs[iii],8]),('%.4E' % out_xa[IIIs[iii],7]),('%.4E' % out_xa[IIIs[iii],6]),('%.4E' % out_xa[IIIs[iii],5]),
                                     ('%.4E' % out_xa[IIIs[iii],4]),('%.4E' % out_xa[IIIs[iii],3]),('%.4E' % out_xa[IIIs[iii],2]),('%.4E' % out_xa[IIIs[iii],1]),('%.4E' % out_xa[IIIs[iii],0])])

                spamwriter.writerow([('%.4E' % out_P[IIIs[iii],19]),('%.4E' % out_P[IIIs[iii],18]),('%.4E' % out_P[IIIs[iii],17]),('%.4E' % out_P[IIIs[iii],16]),('%.4E' % out_P[IIIs[iii],15]),
                                     ('%.4E' % out_P[IIIs[iii],14]),('%.4E' % out_P[IIIs[iii],13]),('%.4E' % out_P[IIIs[iii],12]),('%.4E' % out_P[IIIs[iii],11]),('%.4E' % out_P[IIIs[iii],10]),
                                     ('%.4E' % out_P[IIIs[iii],9]),('%.4E' % out_P[IIIs[iii],8]),('%.4E' % out_P[IIIs[iii],7]),('%.4E' % out_P[IIIs[iii],6]),('%.4E' % out_P[IIIs[iii],5]),
                                     ('%.4E' % out_P[IIIs[iii],4]),('%.4E' % out_P[IIIs[iii],3]),('%.4E' % out_P[IIIs[iii],2]),('%.4E' % out_P[IIIs[iii],1]),('%.4E' % out_P[IIIs[iii],0])])
            iii+=1
f.close



OBS_XCO2 = np.zeros((len(XMid), len(YMid)))
i_ind=0
for i in XMid:
    print i_ind
    j_ind=0
    for j in YMid:
        xco2_inds = np.where( np.logical_and(X_ind_unique-1==i_ind, Y_ind_unique-1==j_ind) )
        OBS_XCO2[i_ind,j_ind] = np.mean(out_xco2[xco2_inds])
        j_ind+=1
    i_ind+=1

OBS_XCO2[np.where(~np.isfinite(OBS_XCO2))]=0

y, x = np.meshgrid(YMid,XMid)
# ================== TCCON ==================
fig = plt.figure(figsize=(20,10))
#ax1=plt.subplot(3, 1, 1)
# -----------------------------------
# miller projection
m = Basemap(projection='mill',lon_0=0)
# plot coastlines, draw label meridians and parallels.
m.drawcoastlines(zorder=0)
xx,yy=m(x,y)
# --- Try plotting ---
im1 = m.pcolormesh(xx,yy,OBS_XCO2,shading='flat',cmap='gist_earth_r')
plt.clim(392,400)
plt.colorbar(orientation='horizontal')  
plt.show()
