import nead
import pandas as pd
import numpy as np
import os
import glob
import re
workingdir ='/home/shl/OneDrive/projects/aws_processing_v1.0/'  

station = 'zac_l'
filelist = glob.glob(workingdir+'data_v1.0/L0M/'+station+'/'+station+'**.csv')
fig, ax = plt.subplots(3,1, figsize = (10,15))
for infile in filelist:
    print(str(infile))
    #infile = filelist[0]
    ds = nead.read(infile)
    ds = ds.set_coords(['time'])
    ds = ds.set_index({'index':'time'})
    ds = ds.rename({'index':'time'})
    ds['time'] = pd.to_datetime(ds.time.values)
    ds['n'] = (('time'), np.arange(ds.time.size)+1)
    
    # Remove duplicate dates
    _, index_dublicates = np.unique(ds['time'], return_index=True)
    ds = ds.isel(time=index_dublicates)
    
    # Remove inf
    for column in ds.keys():
        ds[column][ds[column]==np.inf] = np.nan
    
    
    def add_variable_metadata(ds):
        """Uses the variable DB (variables.csv) to add metadata to the xarray dataset."""
        df = pd.read_csv("./variables.csv", index_col=0, comment="#")
    
        for v in df.index:
            if v == 'time': continue # coordinate variable, not normal var
            if v not in list(ds.variables): continue
            for c in ['standard_name', 'long_name', 'units']:
                if isinstance(df[c][v], np.float) and np.isnan(df[c][v]): continue
                ds[v].attrs[c] = df[c][v]
                
        return ds
    ds = add_variable_metadata(ds)
    ds_raw = ds.copy()

    ds_raw[['tilt_x','tilt_y']].to_dataframe().plot()

station = 'zac_u'
filelist = glob.glob(workingdir+'data_v1.0/L0M/'+station+'/'+station+'**.csv')

for infile in filelist:
    print(str(infile))
    #infile = filelist[0]
    ds = nead.read(infile)
    ds = ds.set_coords(['time'])
    ds = ds.set_index({'index':'time'})
    ds = ds.rename({'index':'time'})
    ds['time'] = pd.to_datetime(ds.time.values)
    ds['n'] = (('time'), np.arange(ds.time.size)+1)
    
    # Remove duplicate dates
    _, index_dublicates = np.unique(ds['time'], return_index=True)
    ds = ds.isel(time=index_dublicates)
    
    # Remove inf
    for column in ds.keys():
        ds[column][ds[column]==np.inf] = np.nan
    
    
    def add_variable_metadata(ds):
        """Uses the variable DB (variables.csv) to add metadata to the xarray dataset."""
        df = pd.read_csv("./variables.csv", index_col=0, comment="#")
    
        for v in df.index:
            if v == 'time': continue # coordinate variable, not normal var
            if v not in list(ds.variables): continue
            for c in ['standard_name', 'long_name', 'units']:
                if isinstance(df[c][v], np.float) and np.isnan(df[c][v]): continue
                ds[v].attrs[c] = df[c][v]
                
        return ds
    ds = add_variable_metadata(ds)
    ds_raw = ds.copy()
ds_raw[['tilt_x','tilt_y']].to_dataframe().plot(ax = ax[1])

station = 'zac_a'
filelist = glob.glob(workingdir+'data_v1.0/L0M/'+station+'/'+station+'**.csv')

for infile in filelist:
    print(str(infile))
    #infile = filelist[0]
    ds = nead.read(infile)
    ds = ds.set_coords(['time'])
    ds = ds.set_index({'index':'time'})
    ds = ds.rename({'index':'time'})
    ds['time'] = pd.to_datetime(ds.time.values)
    ds['n'] = (('time'), np.arange(ds.time.size)+1)
    
    # Remove duplicate dates
    _, index_dublicates = np.unique(ds['time'], return_index=True)
    ds = ds.isel(time=index_dublicates)
    
    # Remove inf
    for column in ds.keys():
        ds[column][ds[column]==np.inf] = np.nan
    
    
    def add_variable_metadata(ds):
        """Uses the variable DB (variables.csv) to add metadata to the xarray dataset."""
        df = pd.read_csv("./variables.csv", index_col=0, comment="#")
    
        for v in df.index:
            if v == 'time': continue # coordinate variable, not normal var
            if v not in list(ds.variables): continue
            for c in ['standard_name', 'long_name', 'units']:
                if isinstance(df[c][v], np.float) and np.isnan(df[c][v]): continue
                ds[v].attrs[c] = df[c][v]
                
        return ds
    ds = add_variable_metadata(ds)
    ds_raw = ds.copy()
ds_raw[['tilt_x','tilt_y']].to_dataframe().plot(ax = ax[2])

import nead
import pandas as pd
import numpy as np
import os
import glob
import re

workingdir ='/home/shl/OneDrive/projects/aws_processing_v1.0/'  
station = 'zac_l'
filelist = glob.glob(workingdir+'data_v1.0/L0M/'+station+'/'+station+'**.csv')

for infile in filelist:
    print(str(infile))
    #infile = filelist[0]
    ds = nead.read(infile)
    ds = ds.set_coords(['time'])
    ds = ds.set_index({'index':'time'})
    ds = ds.rename({'index':'time'})
    ds['time'] = pd.to_datetime(ds.time.values)
    ds['n'] = (('time'), np.arange(ds.time.size)+1)
    
    # Remove duplicate dates
    _, index_dublicates = np.unique(ds['time'], return_index=True)
    ds = ds.isel(time=index_dublicates)
    
    # Remove inf
    for column in ds.keys():
        ds[column][ds[column]==np.inf] = np.nan
    
    
    def add_variable_metadata(ds):
        """Uses the variable DB (variables.csv) to add metadata to the xarray dataset."""
        df = pd.read_csv("./variables.csv", index_col=0, comment="#")
    
        for v in df.index:
            if v == 'time': continue # coordinate variable, not normal var
            if v not in list(ds.variables): continue
            for c in ['standard_name', 'long_name', 'units']:
                if isinstance(df[c][v], np.float) and np.isnan(df[c][v]): continue
                ds[v].attrs[c] = df[c][v]
                
        return ds
    ds = add_variable_metadata(ds)
    #ds_raw = ds.copy()
    T_0 = 273.15
    
    # Calculate pressure transducer fluid density
    
    if 'z_pt' in ds:
        if ds.attrs['pt_antifreeze'] == 50:
            rho_af = 1092
        elif ds.attrs['pt_antifreeze'] == 100:
            rho_af = 1145
        else:
            rho_af = np.nan
            print("ERROR: Incorrect metadata: 'pt_antifreeze =' ", ds.attrs['pt_antifreeze'])
            print("Antifreeze mix only supported at 50 % or 100%")
            # assert(False)
        
    for v in ['gps_geounit','min_y']:
        if v in list(ds.variables): ds = ds.drop_vars(v)
    
    
    # convert radiation from engineering to physical units
    if 'dsr' in ds:
            
        ds['dsr'] = (ds['dsr']*10) / ds.attrs['dsr_eng_coef'] * 100
        ds['usr'] = (ds['usr']*10) / ds.attrs['usr_eng_coef'] * 100
        ds['dlr'] = ((ds['dlr']*1000) / ds.attrs['dlr_eng_coef']) + 5.67*10**(-8)*(ds['t_rad'] + T_0)**4 
        ds['ulr'] = ((ds['ulr']*1000) / ds.attrs['ulr_eng_coef']) + 5.67*10**(-8)*(ds['t_rad'] + T_0)**4
        
        ds['tilt_x'] = ds['tilt_x'].astype(float) / 100
        ds['tilt_y'] = ds['tilt_y'].astype(float) / 100
    
    # Adjust sonic ranger readings for sensitivity to air temperature
    if 'z_boom' in ds:
        ds['z_boom'] = ds['z_boom'] * ((ds['t_1'] + T_0)/T_0)**0.5 
    if 'z_stake' in ds:
        ds['z_stake'] = ds['z_stake'] * ((ds['t_1'] + T_0)/T_0)**0.5
    
    # Adjust pressure transducer due to fluid properties
    if 'z_pt' in ds:
        #print('z_pt_corr is produced in' + str(infile) )
        #ds['z_pt'] = ds['z_pt'] * ds.attrs['pt_z_coef'] * ds.attrs['pt_z_factor'] * 998.0 / rho_af
    
        # Calculate pressure transducer depth
        ds['z_pt_corr'] = ds['z_pt'] * np.nan # new 'z_pt_corr' copied from 'z_pt'
        ds['z_pt_corr'].attrs['long_name'] = ds['z_pt'].long_name + " corrected"
        ds['z_pt_corr'] = ds['z_pt'] * ds.attrs['pt_z_coef'] * ds.attrs['pt_z_factor'] * 998.0 / rho_af \
            + 100 * (ds.attrs['pt_z_p_coef'] - ds['p']) / (rho_af * 9.81)
    
    
    # Decode GPS
    if 'gps_lat' in ds:
        if ds['gps_lat'].dtype.kind == 'O': # not a float. Probably has "NH"
            #assert('NH' in ds['gps_lat'].dropna(dim='time').values[0])
            for v in ['gps_lat','gps_lon','gps_time']:
                a = ds[v].attrs # store
                str2nums = [re.findall(r"[-+]?\d*\.\d+|\d+", _) if isinstance(_, str) else [np.nan] for _ in ds[v].values]
                ds[v][:] = pd.DataFrame(str2nums).astype(float).T.values[0]
                ds[v] = ds[v].astype(float)
                ds[v].attrs = a # restore
    
        if np.any((ds['gps_lat'] <= 90) & (ds['gps_lat'] > 0)):  # Some stations only recorded minutes, not degrees
            xyz = np.array(re.findall("[-+]?[\d]*[.][\d]+", ds.attrs['geometry'])).astype(float)
            x=xyz[0]; y=xyz[1]; z=xyz[2] if len(xyz) == 3 else 0
            p = shapely.geometry.Point(x,y,z)
            # from IPython import embed; embed()
            # assert(False) # should p be ints rather than floats here?
            # ds['gps_lat'] = ds['gps_lat'].where(
            ds['gps_lat'] = ds['gps_lat'] + 100*p.y
        if np.any((ds['gps_lon'] <= 90) & (ds['gps_lon'] > 0)):
            ds['gps_lon'] = ds['gps_lon'] + 100*p.x
    
        for v in ['gps_lat','gps_lon']:
            a = ds[v].attrs # store
            ds[v] = np.floor(ds[v] / 100) + (ds[v] / 100 - np.floor(ds[v] / 100)) * 100 / 60
            ds[v].attrs = a # restore
    
    
    # Correct winddir due to boom_azimuth
    
    # ds['ws'].
    
    # tilt-o-meter voltage to degrees
    # if transmitted ne 'yes' then begin
    #    tiltX = smooth(tiltX,7,/EDGE_MIRROR,MISSING=-999) & tiltY = smooth(tiltY,7,/EDGE_MIRROR, MISSING=-999)
    # endif
    
    # Should just be
    # if ds.attrs['PROMICE_format'] != 'TX': dstxy = dstxy.rolling(time=7, win_type='boxcar', center=True).mean()
    # but the /EDGE_MIRROR makes it a bit more complicated...
    
    if 'tilt_x' in ds:
        win_size=7
        s = np.int(win_size/2)
        tdf = ds['tilt_x'].to_dataframe()
        ds['tilt_x'] = (('time'), tdf.iloc[:s][::-1].append(tdf).append(tdf.iloc[-s:][::-1]).rolling(win_size, win_type='boxcar',     center=True).mean()[s:-s].values.flatten())
        tdf = ds['tilt_y'].to_dataframe()
        ds['tilt_y'] = (('time'), tdf.iloc[:s][::-1].append(tdf).append(tdf.iloc[-s:][::-1]).rolling(win_size, win_type='boxcar',    center=True).mean()[s:-s].values.flatten())
    
        # # notOKtiltX = where(tiltX lt -100, complement=OKtiltX) & notOKtiltY = where(tiltY lt -100, complement=OKtiltY)
        notOKtiltX = (ds['tilt_x'] < -100)
        OKtiltX = (ds['tilt_x'] >= -100)
        notOKtiltY = (ds['tilt_y'] < -100)
        OKtiltY = (ds['tilt_y'] >= -100)
    
        # tiltX = tiltX/10.
        #ds['tilt_x'] = ds['tilt_x'] / 10
        #ds['tilt_y'] = ds['tilt_y'] / 10
    
        # tiltnonzero = where(tiltX ne 0 and tiltX gt -40 and tiltX lt 40)
        # if n_elements(tiltnonzero) ne 1 then tiltX[tiltnonzero] = tiltX[tiltnonzero]/abs(tiltX[tiltnonzero])*(-0.49*(abs(tiltX[tiltnonzero]))^4 +   3.6*(abs(tiltX[tiltnonzero]))^3 - 10.4*(abs(tiltX[tiltnonzero]))^2 +21.1*(abs(tiltX[tiltnonzero])))
    
        # tiltY = tiltY/10.
        # tiltnonzero = where(tiltY ne 0 and tiltY gt -40 and tiltY lt 40)
        # if n_elements(tiltnonzero) ne 1 then tiltY[tiltnonzero] = tiltY[tiltnonzero]/abs(tiltY[tiltnonzero])*(-0.49*(abs(tiltY[tiltnonzero]))^4 + 3.6*(abs(tiltY[tiltnonzero]))^3 - 10.4*(abs(tiltY[tiltnonzero]))^2 +21.1*(abs(tiltY[tiltnonzero])))
    
        dstx = ds['tilt_x']
        nz = (dstx != 0) & (np.abs(dstx) < 40)
        dstx = dstx.where(~nz, other = dstx / np.abs(dstx) * (-0.49 * (np.abs(dstx))**4 + 3.6 * (np.abs(dstx))**3 - 10.4 * (np.abs(dstx))**2 + 21.1 * (np.abs(dstx))))
        ds['tilt_x'] = dstx
    
        dsty = ds['tilt_y']
        nz = (dsty != 0) & (np.abs(dsty) < 40)
        dsty = dsty.where(~nz, other = dsty / np.abs(dsty) * (-0.49 * (np.abs(dsty))**4 + 3.6 * (np.abs(dsty))**3 - 10.4 * (np.abs(dsty))**2 + 21.1 * (np.abs(dsty))))
        ds['tilt_y'] = dsty
    
        # if n_elements(OKtiltX) gt 1 then tiltX[notOKtiltX] = interpol(tiltX[OKtiltX],OKtiltX,notOKtiltX) ; Interpolate over gaps for radiation correction; set to -999 again below.
        # if n_elements(OKtiltY) gt 1 then tiltY[notOKtiltY] = interpol(tiltY[OKtiltY],OKtiltY,notOKtiltY) ; Interpolate over gaps for radiation correction; set to -999 again below.
    
        ds['tilt_x'] = ds['tilt_x'].where(~notOKtiltX)
        ds['tilt_y'] = ds['tilt_y'].where(~notOKtiltY)
        ds['tilt_x'] = ds['tilt_x'].interpolate_na(dim='time')
        ds['tilt_y'] = ds['tilt_y'].interpolate_na(dim='time')
    
    # ds['tilt_x'] = ds['tilt_x'].ffill(dim='time')
    # ds['tilt_y'] = ds['tilt_y'].ffill(dim='time')
    
    
    deg2rad = np.pi / 180
    ds['wdir'] = ds['wdir'].where(ds['wspd'] != 0)
    ds['wspd_x'] = ds['wspd'] * np.sin(ds['wdir'] * deg2rad)
    ds['wspd_y'] = ds['wspd'] * np.cos(ds['wdir'] * deg2rad)
         
    #ds_phys = ds.copy()
    below = ds['t_1'].values < 0
    T = ds['t_1'].values + 273.15
    
    ew = 10**(-7.90298*(373.16/T-1)+5.02808*np.log10(373.16/T)-(1.3816*10**(-7))*(10**(11.344*(1-T/373.16))-1)+(8.1328*10**(-3))*(10**(-349149*(373.16/T-1))-1)+np.log10(1013.246))
    
    #ew = 10**(-7.90298*(373.16/T-1)+5.02808*np.log10(373.16/T)-(1.3816*10**(-7))*(10**(11.344*(1-T/373.16))-1)+(8.1328*10**(-3))*(10**(-349149*(373.16/T-1))-1)+np.log10(1013.246))
    
    ei = 10**(-9.09718*(273.16/T-1)-3.56654*np.log10(273.16/T)+0.876793*(1-T/273.16)+np.log10(6.1071))
    
    rh_ice = ds['rh'].values*ew/ei
    
    rh = ds['rh'].copy()
    rh[below] = rh_ice[below]
    #rh[rh>100] = 100
    rh[rh<0] = 0
    ds['rh_corr'] = rh.copy()
    ds['rh_corr'].attrs['long_name'] = ds['rh'].long_name + " corrected"          
    
    T_0 = 273.15
    epsilon = 0.97
    sigma = 5.67*10**(-8)
    Tsurf = ((ds['ulr']-(1-epsilon)*ds['dlr'])/(epsilon*sigma))**0.25 -T_0
    ds = ds.assign({'t_surf':Tsurf})
    
    if 'dsr' in ds:
        # Calculate cloud cover
        T_0 = 273.15
        eps_overcast = 1
        eps_clear = 9.36508e-6
        LR_overcast = eps_overcast*5.67*10**(-8)*(ds['t_1']+T_0)**4 # assumption
        LR_clear = eps_clear*5.67*10**(-8)*(ds['t_1']+T_0)**6 # Swinbank (1963)
        CloudCov = (ds['dlr'].values-LR_clear)/(LR_overcast-LR_clear)
        
        overcast = CloudCov > 1
        Clear = CloudCov < 0
        CloudCov[overcast] = 1
        CloudCov[Clear] = 0
        DifFrac = 0.2+0.8*CloudCov
        
        
        # Calculating the tilt angle and direction of senson and rotating to a north-south aligned coordinate system
        deg2rad = np.pi / 180
        tiltX_rad = ds['tilt_x'].values*deg2rad
        tiltY_rad = ds['tilt_y'].values*deg2rad
        
        X = np.sin(tiltX_rad)*np.cos(tiltX_rad)*(np.sin(tiltY_rad))**2 + np.sin(tiltX_rad)*(np.cos(tiltY_rad))**2 # Cartesian coordinate
        Y = np.sin(tiltY_rad)*np.cos(tiltY_rad)*(np.sin(tiltX_rad))**2 + np.sin(tiltY_rad)*(np.cos(tiltX_rad))**2 # Cartesian coordinate
        Z = np.cos(tiltX_rad)*np.cos(tiltY_rad) + (np.sin(tiltX_rad))**2*(np.sin(tiltY_rad))**2 # Cartesian coordinate
        phi_sensor_rad = -np.pi/2-np.arctan(Y/X) # spherical coordinate
        
        phi_sensor_rad[X > 0] = phi_sensor_rad[X > 0]+np.pi
        phi_sensor_rad[(X == 0) & (Y < 0)] = np.pi
        phi_sensor_rad[(X == 0) & (Y >= 0)] = 0
        phi_sensor_rad[phi_sensor_rad < 0] = phi_sensor_rad[phi_sensor_rad < 0]+2*np.pi
        
        phi_sensor_deg = phi_sensor_rad*180/np.pi # radians to degrees
        theta_sensor_rad = np.arccos(Z/(X**2+Y**2+Z**2)**0.5) # spherical coordinate (or actually total tilt of the sensor, i.e. 0 when horizontal)
        theta_sensor_deg = theta_sensor_rad*180/np.pi # radians to degrees
        
        
        
        # Calculating zenith and hour angle of the sun
        lat = float(ds.geometry[13:19]) #ds['gps_lat'].median().values
        lon = float(ds.geometry[6:12]) #ds['gps_lon'].median().values
        dates = ds.time.to_dataframe()
        dates.index = pd.to_datetime(dates['time'])
        dayofyear =dates.index.dayofyear.values
        hour = dates.index.hour.values
        minute = dates.index.minute.values
        
        d0_rad = 2*np.pi*(dayofyear+(hour+minute/60)/24-1)/365
        Declination_rad = np.arcsin(0.006918-0.399912*np.cos(d0_rad)+0.070257*np.sin(d0_rad)-0.006758*np.cos(2*d0_rad)+0.000907*np.sin(2*d0_rad)-0.002697*np.cos(3*d0_rad)+0.00148*np.sin(3*d0_rad))
        
        HourAngle_rad = 2*np.pi*(((hour+minute/60.)/24-0.5))# - lon/360) #- 15.*timezone/360.) ; NB: Make sure time is in UTC and longitude is positive when west! Hour angle should be 0 at noon.
        DirectionSun_deg = HourAngle_rad*180/np.pi-180 # This is 180 deg at noon (NH), as opposed to HourAngle.
        DirectionSun_deg[DirectionSun_deg < 0] = DirectionSun_deg[DirectionSun_deg < 0]+360
        DirectionSun_deg[DirectionSun_deg < 0] = DirectionSun_deg[DirectionSun_deg < 0]+360
        
        ZenithAngle_rad = np.arccos(np.cos(lat*np.pi/180)*np.cos(Declination_rad)*np.cos(HourAngle_rad) + np.sin(lat*np.pi/180)*np.sin(Declination_rad))
        ZenithAngle_deg = ZenithAngle_rad*180/np.pi
        sundown = ZenithAngle_deg >= 90
        SRtoa = 1372*np.cos(ZenithAngle_rad) # SRin at the top of the atmosphere
        SRtoa[sundown] = 0
        
        
        
        
        # correction factor for direct beam radiation
        CorFac = np.sin(Declination_rad) * np.sin(lat*np.pi/180.) * np.cos(theta_sensor_rad) \
                 -np.sin(Declination_rad) * np.cos(lat*np.pi/180.) * np.sin(theta_sensor_rad) * np.cos(phi_sensor_rad+np.pi) \
                +np.cos(Declination_rad) * np.cos(lat*np.pi/180.) * np.cos(theta_sensor_rad) * np.cos(HourAngle_rad) \
                +np.cos(Declination_rad) * np.sin(lat*np.pi/180.) * np.sin(theta_sensor_rad) * np.cos(phi_sensor_rad+np.pi) * np.cos(HourAngle_rad) \
                +np.cos(Declination_rad) * np.sin(theta_sensor_rad)*np.sin(phi_sensor_rad+np.pi)*np.sin(HourAngle_rad)
        
        CorFac = np.cos(ZenithAngle_rad)/CorFac
        no_correction = (CorFac <= 0) | ( ZenithAngle_deg > 90) # sun out of field of view upper sensor
        CorFac[no_correction] = 1
         # Calculating SRin over a horizontal surface corrected for station/sensor tilt
        CorFac_all = CorFac/(1-DifFrac+CorFac*DifFrac)
        SRin_cor = ds['dsr']*CorFac_all
        #srin_tilt_cor = SRin_cor.copy() # Debuggin
        # Calculating albedo based on albedo values when sun is in sight of the upper sensor
        AngleDif_deg = 180/np.pi*np.arccos(np.sin(ZenithAngle_rad)*np.cos(HourAngle_rad+np.pi)*np.sin(theta_sensor_rad)*np.cos(phi_sensor_rad) \
                                     +np.sin(ZenithAngle_rad)*np.sin(HourAngle_rad+np.pi)*np.sin(theta_sensor_rad)*np.sin(phi_sensor_rad) \
                                     +np.cos(ZenithAngle_rad)*np.cos(theta_sensor_rad)) # angle between sun and sensor
        
        
        albedo = ds['usr']/SRin_cor
        
        
        OKalbedos = (AngleDif_deg < 70) & (ZenithAngle_deg < 70) & (albedo < 1) & (albedo > 0)
        
        notOKalbedos = (AngleDif_deg >= 70) | (ZenithAngle_deg >= 70) | (albedo >= 1) | (albedo <= 0)
        
        albedo[notOKalbedos] = np.nan
        albedo = albedo.ffill('time')
        #albedo = interp1(datenumber,albedo,datenumber,'pchip') # interpolate over gaps - gives problems for discontinuous data sets, but is not the end of the world
        
        # Correcting SR using SRin when sun is in field of view of lower sensor assuming sensor measures only diffuse radiation
        sunonlowerdome = (AngleDif_deg >= 90) & (ZenithAngle_deg <= 90)
        SRin_cor[sunonlowerdome] = ds['dsr'][sunonlowerdome].values/DifFrac[sunonlowerdome]
        
        
        SRout_cor = ds['usr']
        SRout_cor[sunonlowerdome] = albedo[sunonlowerdome]*ds['dsr'][sunonlowerdome].values/DifFrac[sunonlowerdome]
        #srin_cor_dome = SRin_cor.copy() # debugging
        # Setting SRin and SRout to zero for solar zenith angles larger than 95 deg or either SRin or SRout are (less than) zero
        no_SR = (ZenithAngle_deg > 95) | (SRin_cor <= 0) | (SRout_cor <= 0)
        
        SRin_cor[no_SR] = 0
        SRout_cor[no_SR] = 0
        
        # Correcting SRin using more reliable SRout when sun not in sight of upper sensor
        
        #SRin_cor[no_correction] = SRout_cor[no_correction]/albedo[no_correction]
        SRin_cor[~notOKalbedos] = SRout_cor[~notOKalbedos]/albedo[~notOKalbedos]
        #SRin_cor_alb = SRin_cor.copy() # Debugging
        #SRin_cor = SRout_cor/albedo # What is done in the IDL code
        #albedo[notOKalbedos] = -999
        # Removing spikes by interpolation based on a simple top-of-the-atmosphere limitation
        
        SRin_cor_mark = SRin_cor.copy()
        TOA_crit_nopass = SRin_cor > 0.9*SRtoa+10
        
        SRin_cor[TOA_crit_nopass] = np.nan
        SRout_cor[TOA_crit_nopass] = np.nan
        
        
        SRin_cor_final = SRin_cor.copy()
         #The spike removal can disquise problems
        # Assign columns to ds file
        
        ds = ds.assign({'albedo':albedo, 'dsr_corr':SRin_cor, 'usr_corr':SRout_cor, 'cloud_cov':CloudCov})
        ds['I'] = ('time', SRtoa)
        ds['solar_zenith_angle'] = ('time', ZenithAngle_deg )
        
        
        
        #ds_test = ds.assign({'albedo':albedo,'srin_tilt_cor':srin_tilt_cor,'srin_cor_dome':srin_cor_dome, 'SRin_cor_alb':SRin_cor_alb ,'SRin_cor_final':SRin_cor_final,'SRin_cor_mark':SRin_cor_mark , 'dsr_corr':SRin_cor, 'usr_corr':SRout_cor, 'cloud_cov':CloudCov}) # debugging
        #ds['dsr_corr']=SRin_cor
        ds['dsr_corr'].attrs['long_name'] = ds['dsr'].long_name + " corrected"   
        #ds['usr_corr']=SRout_cor.copy()
        ds['usr_corr'].attrs['long_name'] = ds['usr'].long_name + " corrected"   
        
        #ds['cloud_cover']=CloudCov.copy()
        
        
    
    
    outpath = 'data_v1.0/L1/'+station+'/'
    outfile = infile[-14:-4]
    ds = ds.sel(time=ds.time.notnull())
    #ds_test = ds_test.sel(time=ds.time.notnull()) # debugging
    
    outpathfile = outpath + outfile + ".nc"
    #outpathfile_test = outpath + outfile + "_test.nc" #debug
    if os.path.exists(outpathfile): os.remove(outpathfile)
    ds.to_netcdf(outpathfile, mode='w', format='NETCDF4', compute=True)
    #ds_test.to_netcdf(outpathfile_test, mode='w', format='NETCDF4', compute=True) #debug
    
    
   

station = 'zac_u'
filelist = glob.glob(workingdir+'data_v1.0/L0M/'+station+'/'+station+'**.csv')

for infile in filelist:
    print(str(infile))
    #infile = filelist[0]
    ds = nead.read(infile)
    ds = ds.set_coords(['time'])
    ds = ds.set_index({'index':'time'})
    ds = ds.rename({'index':'time'})
    ds['time'] = pd.to_datetime(ds.time.values)
    ds['n'] = (('time'), np.arange(ds.time.size)+1)
    
    # Remove duplicate dates
    _, index_dublicates = np.unique(ds['time'], return_index=True)
    ds = ds.isel(time=index_dublicates)
    
    # Remove inf
    for column in ds.keys():
        ds[column][ds[column]==np.inf] = np.nan
    
    
    def add_variable_metadata(ds):
        """Uses the variable DB (variables.csv) to add metadata to the xarray dataset."""
        df = pd.read_csv("./variables.csv", index_col=0, comment="#")
    
        for v in df.index:
            if v == 'time': continue # coordinate variable, not normal var
            if v not in list(ds.variables): continue
            for c in ['standard_name', 'long_name', 'units']:
                if isinstance(df[c][v], np.float) and np.isnan(df[c][v]): continue
                ds[v].attrs[c] = df[c][v]
                
        return ds
    ds = add_variable_metadata(ds)
    #ds_raw = ds.copy()
    T_0 = 273.15
    
    # Calculate pressure transducer fluid density
    
    if 'z_pt' in ds:
        if ds.attrs['pt_antifreeze'] == 50:
            rho_af = 1092
        elif ds.attrs['pt_antifreeze'] == 100:
            rho_af = 1145
        else:
            rho_af = np.nan
            print("ERROR: Incorrect metadata: 'pt_antifreeze =' ", ds.attrs['pt_antifreeze'])
            print("Antifreeze mix only supported at 50 % or 100%")
            # assert(False)
        
    for v in ['gps_geounit','min_y']:
        if v in list(ds.variables): ds = ds.drop_vars(v)
    
    
    # convert radiation from engineering to physical units
    if 'dsr' in ds:
            
        ds['dsr'] = (ds['dsr']*10) / ds.attrs['dsr_eng_coef'] * 100
        ds['usr'] = (ds['usr']*10) / ds.attrs['usr_eng_coef'] * 100
        ds['dlr'] = ((ds['dlr']*1000) / ds.attrs['dlr_eng_coef']) + 5.67*10**(-8)*(ds['t_rad'] + T_0)**4 
        ds['ulr'] = ((ds['ulr']*1000) / ds.attrs['ulr_eng_coef']) + 5.67*10**(-8)*(ds['t_rad'] + T_0)**4
        
        ds['tilt_x'] = ds['tilt_x'].astype(float) / 100
        ds['tilt_y'] = ds['tilt_y'].astype(float) / 100
    
    # Adjust sonic ranger readings for sensitivity to air temperature
    if 'z_boom' in ds:
        ds['z_boom'] = ds['z_boom'] * ((ds['t_1'] + T_0)/T_0)**0.5 
    if 'z_stake' in ds:
        ds['z_stake'] = ds['z_stake'] * ((ds['t_1'] + T_0)/T_0)**0.5
    
    # Adjust pressure transducer due to fluid properties
    if 'z_pt' in ds:
        #print('z_pt_corr is produced in' + str(infile) )
        #ds['z_pt'] = ds['z_pt'] * ds.attrs['pt_z_coef'] * ds.attrs['pt_z_factor'] * 998.0 / rho_af
    
        # Calculate pressure transducer depth
        ds['z_pt_corr'] = ds['z_pt'] * np.nan # new 'z_pt_corr' copied from 'z_pt'
        ds['z_pt_corr'].attrs['long_name'] = ds['z_pt'].long_name + " corrected"
        ds['z_pt_corr'] = ds['z_pt'] * ds.attrs['pt_z_coef'] * ds.attrs['pt_z_factor'] * 998.0 / rho_af \
            + 100 * (ds.attrs['pt_z_p_coef'] - ds['p']) / (rho_af * 9.81)
    
    
    # Decode GPS
    if 'gps_lat' in ds:
        if ds['gps_lat'].dtype.kind == 'O': # not a float. Probably has "NH"
            #assert('NH' in ds['gps_lat'].dropna(dim='time').values[0])
            for v in ['gps_lat','gps_lon','gps_time']:
                a = ds[v].attrs # store
                str2nums = [re.findall(r"[-+]?\d*\.\d+|\d+", _) if isinstance(_, str) else [np.nan] for _ in ds[v].values]
                ds[v][:] = pd.DataFrame(str2nums).astype(float).T.values[0]
                ds[v] = ds[v].astype(float)
                ds[v].attrs = a # restore
    
        if np.any((ds['gps_lat'] <= 90) & (ds['gps_lat'] > 0)):  # Some stations only recorded minutes, not degrees
            xyz = np.array(re.findall("[-+]?[\d]*[.][\d]+", ds.attrs['geometry'])).astype(float)
            x=xyz[0]; y=xyz[1]; z=xyz[2] if len(xyz) == 3 else 0
            p = shapely.geometry.Point(x,y,z)
            # from IPython import embed; embed()
            # assert(False) # should p be ints rather than floats here?
            # ds['gps_lat'] = ds['gps_lat'].where(
            ds['gps_lat'] = ds['gps_lat'] + 100*p.y
        if np.any((ds['gps_lon'] <= 90) & (ds['gps_lon'] > 0)):
            ds['gps_lon'] = ds['gps_lon'] + 100*p.x
    
        for v in ['gps_lat','gps_lon']:
            a = ds[v].attrs # store
            ds[v] = np.floor(ds[v] / 100) + (ds[v] / 100 - np.floor(ds[v] / 100)) * 100 / 60
            ds[v].attrs = a # restore
    
    
    # Correct winddir due to boom_azimuth
    
    # ds['ws'].
    
    # tilt-o-meter voltage to degrees
    # if transmitted ne 'yes' then begin
    #    tiltX = smooth(tiltX,7,/EDGE_MIRROR,MISSING=-999) & tiltY = smooth(tiltY,7,/EDGE_MIRROR, MISSING=-999)
    # endif
    
    # Should just be
    # if ds.attrs['PROMICE_format'] != 'TX': dstxy = dstxy.rolling(time=7, win_type='boxcar', center=True).mean()
    # but the /EDGE_MIRROR makes it a bit more complicated...
    
    if 'tilt_x' in ds:
        win_size=7
        s = np.int(win_size/2)
        tdf = ds['tilt_x'].to_dataframe()
        ds['tilt_x'] = (('time'), tdf.iloc[:s][::-1].append(tdf).append(tdf.iloc[-s:][::-1]).rolling(win_size, win_type='boxcar',     center=True).mean()[s:-s].values.flatten())
        tdf = ds['tilt_y'].to_dataframe()
        ds['tilt_y'] = (('time'), tdf.iloc[:s][::-1].append(tdf).append(tdf.iloc[-s:][::-1]).rolling(win_size, win_type='boxcar',    center=True).mean()[s:-s].values.flatten())
    
        # # notOKtiltX = where(tiltX lt -100, complement=OKtiltX) & notOKtiltY = where(tiltY lt -100, complement=OKtiltY)
        notOKtiltX = (ds['tilt_x'] < -100)
        OKtiltX = (ds['tilt_x'] >= -100)
        notOKtiltY = (ds['tilt_y'] < -100)
        OKtiltY = (ds['tilt_y'] >= -100)
    
        # tiltX = tiltX/10.
        #ds['tilt_x'] = ds['tilt_x'] / 10
        #ds['tilt_y'] = ds['tilt_y'] / 10
    
        # tiltnonzero = where(tiltX ne 0 and tiltX gt -40 and tiltX lt 40)
        # if n_elements(tiltnonzero) ne 1 then tiltX[tiltnonzero] = tiltX[tiltnonzero]/abs(tiltX[tiltnonzero])*(-0.49*(abs(tiltX[tiltnonzero]))^4 +   3.6*(abs(tiltX[tiltnonzero]))^3 - 10.4*(abs(tiltX[tiltnonzero]))^2 +21.1*(abs(tiltX[tiltnonzero])))
    
        # tiltY = tiltY/10.
        # tiltnonzero = where(tiltY ne 0 and tiltY gt -40 and tiltY lt 40)
        # if n_elements(tiltnonzero) ne 1 then tiltY[tiltnonzero] = tiltY[tiltnonzero]/abs(tiltY[tiltnonzero])*(-0.49*(abs(tiltY[tiltnonzero]))^4 + 3.6*(abs(tiltY[tiltnonzero]))^3 - 10.4*(abs(tiltY[tiltnonzero]))^2 +21.1*(abs(tiltY[tiltnonzero])))
    
        dstx = ds['tilt_x']
        nz = (dstx != 0) & (np.abs(dstx) < 40)
        dstx = dstx.where(~nz, other = dstx / np.abs(dstx) * (-0.49 * (np.abs(dstx))**4 + 3.6 * (np.abs(dstx))**3 - 10.4 * (np.abs(dstx))**2 + 21.1 * (np.abs(dstx))))
        ds['tilt_x'] = dstx
    
        dsty = ds['tilt_y']
        nz = (dsty != 0) & (np.abs(dsty) < 40)
        dsty = dsty.where(~nz, other = dsty / np.abs(dsty) * (-0.49 * (np.abs(dsty))**4 + 3.6 * (np.abs(dsty))**3 - 10.4 * (np.abs(dsty))**2 + 21.1 * (np.abs(dsty))))
        ds['tilt_y'] = dsty
    
        # if n_elements(OKtiltX) gt 1 then tiltX[notOKtiltX] = interpol(tiltX[OKtiltX],OKtiltX,notOKtiltX) ; Interpolate over gaps for radiation correction; set to -999 again below.
        # if n_elements(OKtiltY) gt 1 then tiltY[notOKtiltY] = interpol(tiltY[OKtiltY],OKtiltY,notOKtiltY) ; Interpolate over gaps for radiation correction; set to -999 again below.
    
        ds['tilt_x'] = ds['tilt_x'].where(~notOKtiltX)
        ds['tilt_y'] = ds['tilt_y'].where(~notOKtiltY)
        ds['tilt_x'] = ds['tilt_x'].interpolate_na(dim='time')
        ds['tilt_y'] = ds['tilt_y'].interpolate_na(dim='time')
    
    # ds['tilt_x'] = ds['tilt_x'].ffill(dim='time')
    # ds['tilt_y'] = ds['tilt_y'].ffill(dim='time')
    
    
    deg2rad = np.pi / 180
    ds['wdir'] = ds['wdir'].where(ds['wspd'] != 0)
    ds['wspd_x'] = ds['wspd'] * np.sin(ds['wdir'] * deg2rad)
    ds['wspd_y'] = ds['wspd'] * np.cos(ds['wdir'] * deg2rad)
         
    #ds_phys = ds.copy()
    below = ds['t_1'].values < 0
    T = ds['t_1'].values + 273.15
    
    ew = 10**(-7.90298*(373.16/T-1)+5.02808*np.log10(373.16/T)-(1.3816*10**(-7))*(10**(11.344*(1-T/373.16))-1)+(8.1328*10**(-3))*(10**(-349149*(373.16/T-1))-1)+np.log10(1013.246))
    
    #ew = 10**(-7.90298*(373.16/T-1)+5.02808*np.log10(373.16/T)-(1.3816*10**(-7))*(10**(11.344*(1-T/373.16))-1)+(8.1328*10**(-3))*(10**(-349149*(373.16/T-1))-1)+np.log10(1013.246))
    
    ei = 10**(-9.09718*(273.16/T-1)-3.56654*np.log10(273.16/T)+0.876793*(1-T/273.16)+np.log10(6.1071))
    
    rh_ice = ds['rh'].values*ew/ei
    
    rh = ds['rh'].copy()
    rh[below] = rh_ice[below]
    #rh[rh>100] = 100
    rh[rh<0] = 0
    ds['rh_corr'] = rh.copy()
    ds['rh_corr'].attrs['long_name'] = ds['rh'].long_name + " corrected"          
    
    T_0 = 273.15
    epsilon = 0.97
    sigma = 5.67*10**(-8)
    Tsurf = ((ds['ulr']-(1-epsilon)*ds['dlr'])/(epsilon*sigma))**0.25 -T_0
    ds = ds.assign({'t_surf':Tsurf})
    
    if 'dsr' in ds:
        # Calculate cloud cover
        T_0 = 273.15
        eps_overcast = 1
        eps_clear = 9.36508e-6
        LR_overcast = eps_overcast*5.67*10**(-8)*(ds['t_1']+T_0)**4 # assumption
        LR_clear = eps_clear*5.67*10**(-8)*(ds['t_1']+T_0)**6 # Swinbank (1963)
        CloudCov = (ds['dlr'].values-LR_clear)/(LR_overcast-LR_clear)
        
        overcast = CloudCov > 1
        Clear = CloudCov < 0
        CloudCov[overcast] = 1
        CloudCov[Clear] = 0
        DifFrac = 0.2+0.8*CloudCov
        
        
        # Calculating the tilt angle and direction of senson and rotating to a north-south aligned coordinate system
        deg2rad = np.pi / 180
        tiltX_rad = ds['tilt_x'].values*deg2rad
        tiltY_rad = ds['tilt_y'].values*deg2rad
        
        X = np.sin(tiltX_rad)*np.cos(tiltX_rad)*(np.sin(tiltY_rad))**2 + np.sin(tiltX_rad)*(np.cos(tiltY_rad))**2 # Cartesian coordinate
        Y = np.sin(tiltY_rad)*np.cos(tiltY_rad)*(np.sin(tiltX_rad))**2 + np.sin(tiltY_rad)*(np.cos(tiltX_rad))**2 # Cartesian coordinate
        Z = np.cos(tiltX_rad)*np.cos(tiltY_rad) + (np.sin(tiltX_rad))**2*(np.sin(tiltY_rad))**2 # Cartesian coordinate
        phi_sensor_rad = -np.pi/2-np.arctan(Y/X) # spherical coordinate
        
        phi_sensor_rad[X > 0] = phi_sensor_rad[X > 0]+np.pi
        phi_sensor_rad[(X == 0) & (Y < 0)] = np.pi
        phi_sensor_rad[(X == 0) & (Y >= 0)] = 0
        phi_sensor_rad[phi_sensor_rad < 0] = phi_sensor_rad[phi_sensor_rad < 0]+2*np.pi
        
        phi_sensor_deg = phi_sensor_rad*180/np.pi # radians to degrees
        theta_sensor_rad = np.arccos(Z/(X**2+Y**2+Z**2)**0.5) # spherical coordinate (or actually total tilt of the sensor, i.e. 0 when horizontal)
        theta_sensor_deg = theta_sensor_rad*180/np.pi # radians to degrees
        
        
        
        # Calculating zenith and hour angle of the sun
        lat = float(ds.geometry[13:19]) #ds['gps_lat'].median().values
        lon = float(ds.geometry[6:12]) #ds['gps_lon'].median().values
        dates = ds.time.to_dataframe()
        dates.index = pd.to_datetime(dates['time'])
        dayofyear =dates.index.dayofyear.values
        hour = dates.index.hour.values
        minute = dates.index.minute.values
        
        d0_rad = 2*np.pi*(dayofyear+(hour+minute/60)/24-1)/365
        Declination_rad = np.arcsin(0.006918-0.399912*np.cos(d0_rad)+0.070257*np.sin(d0_rad)-0.006758*np.cos(2*d0_rad)+0.000907*np.sin(2*d0_rad)-0.002697*np.cos(3*d0_rad)+0.00148*np.sin(3*d0_rad))
        
        HourAngle_rad = 2*np.pi*(((hour+minute/60.)/24-0.5))# - lon/360) #- 15.*timezone/360.) ; NB: Make sure time is in UTC and longitude is positive when west! Hour angle should be 0 at noon.
        DirectionSun_deg = HourAngle_rad*180/np.pi-180 # This is 180 deg at noon (NH), as opposed to HourAngle.
        DirectionSun_deg[DirectionSun_deg < 0] = DirectionSun_deg[DirectionSun_deg < 0]+360
        DirectionSun_deg[DirectionSun_deg < 0] = DirectionSun_deg[DirectionSun_deg < 0]+360
        
        ZenithAngle_rad = np.arccos(np.cos(lat*np.pi/180)*np.cos(Declination_rad)*np.cos(HourAngle_rad) + np.sin(lat*np.pi/180)*np.sin(Declination_rad))
        ZenithAngle_deg = ZenithAngle_rad*180/np.pi
        sundown = ZenithAngle_deg >= 90
        SRtoa = 1372*np.cos(ZenithAngle_rad) # SRin at the top of the atmosphere
        SRtoa[sundown] = 0
        
        
        
        
        # correction factor for direct beam radiation
        CorFac = np.sin(Declination_rad) * np.sin(lat*np.pi/180.) * np.cos(theta_sensor_rad) \
                 -np.sin(Declination_rad) * np.cos(lat*np.pi/180.) * np.sin(theta_sensor_rad) * np.cos(phi_sensor_rad+np.pi) \
                +np.cos(Declination_rad) * np.cos(lat*np.pi/180.) * np.cos(theta_sensor_rad) * np.cos(HourAngle_rad) \
                +np.cos(Declination_rad) * np.sin(lat*np.pi/180.) * np.sin(theta_sensor_rad) * np.cos(phi_sensor_rad+np.pi) * np.cos(HourAngle_rad) \
                +np.cos(Declination_rad) * np.sin(theta_sensor_rad)*np.sin(phi_sensor_rad+np.pi)*np.sin(HourAngle_rad)
        
        CorFac = np.cos(ZenithAngle_rad)/CorFac
        no_correction = (CorFac <= 0) | ( ZenithAngle_deg > 90) # sun out of field of view upper sensor
        CorFac[no_correction] = 1
         # Calculating SRin over a horizontal surface corrected for station/sensor tilt
        CorFac_all = CorFac/(1-DifFrac+CorFac*DifFrac)
        SRin_cor = ds['dsr']*CorFac_all
        #srin_tilt_cor = SRin_cor.copy() # Debuggin
        # Calculating albedo based on albedo values when sun is in sight of the upper sensor
        AngleDif_deg = 180/np.pi*np.arccos(np.sin(ZenithAngle_rad)*np.cos(HourAngle_rad+np.pi)*np.sin(theta_sensor_rad)*np.cos(phi_sensor_rad) \
                                     +np.sin(ZenithAngle_rad)*np.sin(HourAngle_rad+np.pi)*np.sin(theta_sensor_rad)*np.sin(phi_sensor_rad) \
                                     +np.cos(ZenithAngle_rad)*np.cos(theta_sensor_rad)) # angle between sun and sensor
        
        
        albedo = ds['usr']/SRin_cor
        
        
        OKalbedos = (AngleDif_deg < 70) & (ZenithAngle_deg < 70) & (albedo < 1) & (albedo > 0)
        
        notOKalbedos = (AngleDif_deg >= 70) | (ZenithAngle_deg >= 70) | (albedo >= 1) | (albedo <= 0)
        
        albedo[notOKalbedos] = np.nan
        albedo = albedo.ffill('time')
        #albedo = interp1(datenumber,albedo,datenumber,'pchip') # interpolate over gaps - gives problems for discontinuous data sets, but is not the end of the world
        
        # Correcting SR using SRin when sun is in field of view of lower sensor assuming sensor measures only diffuse radiation
        sunonlowerdome = (AngleDif_deg >= 90) & (ZenithAngle_deg <= 90)
        SRin_cor[sunonlowerdome] = ds['dsr'][sunonlowerdome].values/DifFrac[sunonlowerdome]
        
        
        SRout_cor = ds['usr']
        SRout_cor[sunonlowerdome] = albedo[sunonlowerdome]*ds['dsr'][sunonlowerdome].values/DifFrac[sunonlowerdome]
        #srin_cor_dome = SRin_cor.copy() # debugging
        # Setting SRin and SRout to zero for solar zenith angles larger than 95 deg or either SRin or SRout are (less than) zero
        no_SR = (ZenithAngle_deg > 95) | (SRin_cor <= 0) | (SRout_cor <= 0)
        
        SRin_cor[no_SR] = 0
        SRout_cor[no_SR] = 0
        
        # Correcting SRin using more reliable SRout when sun not in sight of upper sensor
        
        #SRin_cor[no_correction] = SRout_cor[no_correction]/albedo[no_correction]
        SRin_cor[~notOKalbedos] = SRout_cor[~notOKalbedos]/albedo[~notOKalbedos]
        #SRin_cor_alb = SRin_cor.copy() # Debugging
        #SRin_cor = SRout_cor/albedo # What is done in the IDL code
        #albedo[notOKalbedos] = -999
        # Removing spikes by interpolation based on a simple top-of-the-atmosphere limitation
        
        SRin_cor_mark = SRin_cor.copy()
        TOA_crit_nopass = SRin_cor > 0.9*SRtoa+10
        
        SRin_cor[TOA_crit_nopass] = np.nan
        SRout_cor[TOA_crit_nopass] = np.nan
        
        
        SRin_cor_final = SRin_cor.copy()
         #The spike removal can disquise problems
        # Assign columns to ds file
        
        ds = ds.assign({'albedo':albedo, 'dsr_corr':SRin_cor, 'usr_corr':SRout_cor, 'cloud_cov':CloudCov})
        ds['I'] = ('time', SRtoa)
        ds['solar_zenith_angle'] = ('time', ZenithAngle_deg )
        
        
        
        #ds_test = ds.assign({'albedo':albedo,'srin_tilt_cor':srin_tilt_cor,'srin_cor_dome':srin_cor_dome, 'SRin_cor_alb':SRin_cor_alb ,'SRin_cor_final':SRin_cor_final,'SRin_cor_mark':SRin_cor_mark , 'dsr_corr':SRin_cor, 'usr_corr':SRout_cor, 'cloud_cov':CloudCov}) # debugging
        #ds['dsr_corr']=SRin_cor
        ds['dsr_corr'].attrs['long_name'] = ds['dsr'].long_name + " corrected"   
        #ds['usr_corr']=SRout_cor.copy()
        ds['usr_corr'].attrs['long_name'] = ds['usr'].long_name + " corrected"   
        
        #ds['cloud_cover']=CloudCov.copy()
        
        
    
    
    outpath = 'data_v1.0/L1/'+station+'/'
    outfile = infile[-14:-4]
    ds = ds.sel(time=ds.time.notnull())
    #ds_test = ds_test.sel(time=ds.time.notnull()) # debugging
    
    outpathfile = outpath + outfile + ".nc"
    #outpathfile_test = outpath + outfile + "_test.nc" #debug
    if os.path.exists(outpathfile): os.remove(outpathfile)
    ds.to_netcdf(outpathfile, mode='w', format='NETCDF4', compute=True)
    #ds_test.to_netcdf(outpathfile_test, mode='w', format='NETCDF4', compute=True) #debug
    
    
   

station = 'zac_a'
filelist = glob.glob(workingdir+'data_v1.0/L0M/'+station+'/'+station+'**.csv')

for infile in filelist:
    print(str(infile))
    #infile = filelist[0]
    ds = nead.read(infile)
    ds = ds.set_coords(['time'])
    ds = ds.set_index({'index':'time'})
    ds = ds.rename({'index':'time'})
    ds['time'] = pd.to_datetime(ds.time.values)
    ds['n'] = (('time'), np.arange(ds.time.size)+1)
    
    # Remove duplicate dates
    _, index_dublicates = np.unique(ds['time'], return_index=True)
    ds = ds.isel(time=index_dublicates)
    
    # Remove inf
    for column in ds.keys():
        ds[column][ds[column]==np.inf] = np.nan
    
    
    def add_variable_metadata(ds):
        """Uses the variable DB (variables.csv) to add metadata to the xarray dataset."""
        df = pd.read_csv("./variables.csv", index_col=0, comment="#")
    
        for v in df.index:
            if v == 'time': continue # coordinate variable, not normal var
            if v not in list(ds.variables): continue
            for c in ['standard_name', 'long_name', 'units']:
                if isinstance(df[c][v], np.float) and np.isnan(df[c][v]): continue
                ds[v].attrs[c] = df[c][v]
                
        return ds
    ds = add_variable_metadata(ds)
    #ds_raw = ds.copy()
    T_0 = 273.15
    
    # Calculate pressure transducer fluid density
    
    if 'z_pt' in ds:
        if ds.attrs['pt_antifreeze'] == 50:
            rho_af = 1092
        elif ds.attrs['pt_antifreeze'] == 100:
            rho_af = 1145
        else:
            rho_af = np.nan
            print("ERROR: Incorrect metadata: 'pt_antifreeze =' ", ds.attrs['pt_antifreeze'])
            print("Antifreeze mix only supported at 50 % or 100%")
            # assert(False)
        
    for v in ['gps_geounit','min_y']:
        if v in list(ds.variables): ds = ds.drop_vars(v)
    
    
    # convert radiation from engineering to physical units
    if 'dsr' in ds:
            
        ds['dsr'] = (ds['dsr']*10) / ds.attrs['dsr_eng_coef'] * 100
        ds['usr'] = (ds['usr']*10) / ds.attrs['usr_eng_coef'] * 100
        ds['dlr'] = ((ds['dlr']*1000) / ds.attrs['dlr_eng_coef']) + 5.67*10**(-8)*(ds['t_rad'] + T_0)**4 
        ds['ulr'] = ((ds['ulr']*1000) / ds.attrs['ulr_eng_coef']) + 5.67*10**(-8)*(ds['t_rad'] + T_0)**4
        
        ds['tilt_x'] = ds['tilt_x'].astype(float) / 100
        ds['tilt_y'] = ds['tilt_y'].astype(float) / 100
    
    # Adjust sonic ranger readings for sensitivity to air temperature
    if 'z_boom' in ds:
        ds['z_boom'] = ds['z_boom'] * ((ds['t_1'] + T_0)/T_0)**0.5 
    if 'z_stake' in ds:
        ds['z_stake'] = ds['z_stake'] * ((ds['t_1'] + T_0)/T_0)**0.5
    
    # Adjust pressure transducer due to fluid properties
    if 'z_pt' in ds:
        #print('z_pt_corr is produced in' + str(infile) )
        #ds['z_pt'] = ds['z_pt'] * ds.attrs['pt_z_coef'] * ds.attrs['pt_z_factor'] * 998.0 / rho_af
    
        # Calculate pressure transducer depth
        ds['z_pt_corr'] = ds['z_pt'] * np.nan # new 'z_pt_corr' copied from 'z_pt'
        ds['z_pt_corr'].attrs['long_name'] = ds['z_pt'].long_name + " corrected"
        ds['z_pt_corr'] = ds['z_pt'] * ds.attrs['pt_z_coef'] * ds.attrs['pt_z_factor'] * 998.0 / rho_af \
            + 100 * (ds.attrs['pt_z_p_coef'] - ds['p']) / (rho_af * 9.81)
    
    
    # Decode GPS
    if 'gps_lat' in ds:
        if ds['gps_lat'].dtype.kind == 'O': # not a float. Probably has "NH"
            #assert('NH' in ds['gps_lat'].dropna(dim='time').values[0])
            for v in ['gps_lat','gps_lon','gps_time']:
                a = ds[v].attrs # store
                str2nums = [re.findall(r"[-+]?\d*\.\d+|\d+", _) if isinstance(_, str) else [np.nan] for _ in ds[v].values]
                ds[v][:] = pd.DataFrame(str2nums).astype(float).T.values[0]
                ds[v] = ds[v].astype(float)
                ds[v].attrs = a # restore
    
        if np.any((ds['gps_lat'] <= 90) & (ds['gps_lat'] > 0)):  # Some stations only recorded minutes, not degrees
            xyz = np.array(re.findall("[-+]?[\d]*[.][\d]+", ds.attrs['geometry'])).astype(float)
            x=xyz[0]; y=xyz[1]; z=xyz[2] if len(xyz) == 3 else 0
            p = shapely.geometry.Point(x,y,z)
            # from IPython import embed; embed()
            # assert(False) # should p be ints rather than floats here?
            # ds['gps_lat'] = ds['gps_lat'].where(
            ds['gps_lat'] = ds['gps_lat'] + 100*p.y
        if np.any((ds['gps_lon'] <= 90) & (ds['gps_lon'] > 0)):
            ds['gps_lon'] = ds['gps_lon'] + 100*p.x
    
        for v in ['gps_lat','gps_lon']:
            a = ds[v].attrs # store
            ds[v] = np.floor(ds[v] / 100) + (ds[v] / 100 - np.floor(ds[v] / 100)) * 100 / 60
            ds[v].attrs = a # restore
    
    
    # Correct winddir due to boom_azimuth
    
    # ds['ws'].
    
    # tilt-o-meter voltage to degrees
    # if transmitted ne 'yes' then begin
    #    tiltX = smooth(tiltX,7,/EDGE_MIRROR,MISSING=-999) & tiltY = smooth(tiltY,7,/EDGE_MIRROR, MISSING=-999)
    # endif
    
    # Should just be
    # if ds.attrs['PROMICE_format'] != 'TX': dstxy = dstxy.rolling(time=7, win_type='boxcar', center=True).mean()
    # but the /EDGE_MIRROR makes it a bit more complicated...
    
    if 'tilt_x' in ds:
        win_size=7
        s = np.int(win_size/2)
        tdf = ds['tilt_x'].to_dataframe()
        ds['tilt_x'] = (('time'), tdf.iloc[:s][::-1].append(tdf).append(tdf.iloc[-s:][::-1]).rolling(win_size, win_type='boxcar',     center=True).mean()[s:-s].values.flatten())
        tdf = ds['tilt_y'].to_dataframe()
        ds['tilt_y'] = (('time'), tdf.iloc[:s][::-1].append(tdf).append(tdf.iloc[-s:][::-1]).rolling(win_size, win_type='boxcar',    center=True).mean()[s:-s].values.flatten())
    
        # # notOKtiltX = where(tiltX lt -100, complement=OKtiltX) & notOKtiltY = where(tiltY lt -100, complement=OKtiltY)
        notOKtiltX = (ds['tilt_x'] < -100)
        OKtiltX = (ds['tilt_x'] >= -100)
        notOKtiltY = (ds['tilt_y'] < -100)
        OKtiltY = (ds['tilt_y'] >= -100)
    
        # tiltX = tiltX/10.
        #ds['tilt_x'] = ds['tilt_x'] / 10
        #ds['tilt_y'] = ds['tilt_y'] / 10
    
        # tiltnonzero = where(tiltX ne 0 and tiltX gt -40 and tiltX lt 40)
        # if n_elements(tiltnonzero) ne 1 then tiltX[tiltnonzero] = tiltX[tiltnonzero]/abs(tiltX[tiltnonzero])*(-0.49*(abs(tiltX[tiltnonzero]))^4 +   3.6*(abs(tiltX[tiltnonzero]))^3 - 10.4*(abs(tiltX[tiltnonzero]))^2 +21.1*(abs(tiltX[tiltnonzero])))
    
        # tiltY = tiltY/10.
        # tiltnonzero = where(tiltY ne 0 and tiltY gt -40 and tiltY lt 40)
        # if n_elements(tiltnonzero) ne 1 then tiltY[tiltnonzero] = tiltY[tiltnonzero]/abs(tiltY[tiltnonzero])*(-0.49*(abs(tiltY[tiltnonzero]))^4 + 3.6*(abs(tiltY[tiltnonzero]))^3 - 10.4*(abs(tiltY[tiltnonzero]))^2 +21.1*(abs(tiltY[tiltnonzero])))
    
        dstx = ds['tilt_x']
        nz = (dstx != 0) & (np.abs(dstx) < 40)
        dstx = dstx.where(~nz, other = dstx / np.abs(dstx) * (-0.49 * (np.abs(dstx))**4 + 3.6 * (np.abs(dstx))**3 - 10.4 * (np.abs(dstx))**2 + 21.1 * (np.abs(dstx))))
        ds['tilt_x'] = dstx
    
        dsty = ds['tilt_y']
        nz = (dsty != 0) & (np.abs(dsty) < 40)
        dsty = dsty.where(~nz, other = dsty / np.abs(dsty) * (-0.49 * (np.abs(dsty))**4 + 3.6 * (np.abs(dsty))**3 - 10.4 * (np.abs(dsty))**2 + 21.1 * (np.abs(dsty))))
        ds['tilt_y'] = dsty
    
        # if n_elements(OKtiltX) gt 1 then tiltX[notOKtiltX] = interpol(tiltX[OKtiltX],OKtiltX,notOKtiltX) ; Interpolate over gaps for radiation correction; set to -999 again below.
        # if n_elements(OKtiltY) gt 1 then tiltY[notOKtiltY] = interpol(tiltY[OKtiltY],OKtiltY,notOKtiltY) ; Interpolate over gaps for radiation correction; set to -999 again below.
    
        ds['tilt_x'] = ds['tilt_x'].where(~notOKtiltX)
        ds['tilt_y'] = ds['tilt_y'].where(~notOKtiltY)
        ds['tilt_x'] = ds['tilt_x'].interpolate_na(dim='time')
        ds['tilt_y'] = ds['tilt_y'].interpolate_na(dim='time')
    
    # ds['tilt_x'] = ds['tilt_x'].ffill(dim='time')
    # ds['tilt_y'] = ds['tilt_y'].ffill(dim='time')
    
    
    deg2rad = np.pi / 180
    ds['wdir'] = ds['wdir'].where(ds['wspd'] != 0)
    ds['wspd_x'] = ds['wspd'] * np.sin(ds['wdir'] * deg2rad)
    ds['wspd_y'] = ds['wspd'] * np.cos(ds['wdir'] * deg2rad)
         
    #ds_phys = ds.copy()
    below = ds['t_1'].values < 0
    T = ds['t_1'].values + 273.15
    
    ew = 10**(-7.90298*(373.16/T-1)+5.02808*np.log10(373.16/T)-(1.3816*10**(-7))*(10**(11.344*(1-T/373.16))-1)+(8.1328*10**(-3))*(10**(-349149*(373.16/T-1))-1)+np.log10(1013.246))
    
    #ew = 10**(-7.90298*(373.16/T-1)+5.02808*np.log10(373.16/T)-(1.3816*10**(-7))*(10**(11.344*(1-T/373.16))-1)+(8.1328*10**(-3))*(10**(-349149*(373.16/T-1))-1)+np.log10(1013.246))
    
    ei = 10**(-9.09718*(273.16/T-1)-3.56654*np.log10(273.16/T)+0.876793*(1-T/273.16)+np.log10(6.1071))
    
    rh_ice = ds['rh'].values*ew/ei
    
    rh = ds['rh'].copy()
    rh[below] = rh_ice[below]
    #rh[rh>100] = 100
    rh[rh<0] = 0
    ds['rh_corr'] = rh.copy()
    ds['rh_corr'].attrs['long_name'] = ds['rh'].long_name + " corrected"          
    
    T_0 = 273.15
    epsilon = 0.97
    sigma = 5.67*10**(-8)
    Tsurf = ((ds['ulr']-(1-epsilon)*ds['dlr'])/(epsilon*sigma))**0.25 -T_0
    ds = ds.assign({'t_surf':Tsurf})
    
    if 'dsr' in ds:
        # Calculate cloud cover
        T_0 = 273.15
        eps_overcast = 1
        eps_clear = 9.36508e-6
        LR_overcast = eps_overcast*5.67*10**(-8)*(ds['t_1']+T_0)**4 # assumption
        LR_clear = eps_clear*5.67*10**(-8)*(ds['t_1']+T_0)**6 # Swinbank (1963)
        CloudCov = (ds['dlr'].values-LR_clear)/(LR_overcast-LR_clear)
        
        overcast = CloudCov > 1
        Clear = CloudCov < 0
        CloudCov[overcast] = 1
        CloudCov[Clear] = 0
        DifFrac = 0.2+0.8*CloudCov
        
        
        # Calculating the tilt angle and direction of senson and rotating to a north-south aligned coordinate system
        deg2rad = np.pi / 180
        tiltX_rad = ds['tilt_x'].values*deg2rad
        tiltY_rad = ds['tilt_y'].values*deg2rad
        
        X = np.sin(tiltX_rad)*np.cos(tiltX_rad)*(np.sin(tiltY_rad))**2 + np.sin(tiltX_rad)*(np.cos(tiltY_rad))**2 # Cartesian coordinate
        Y = np.sin(tiltY_rad)*np.cos(tiltY_rad)*(np.sin(tiltX_rad))**2 + np.sin(tiltY_rad)*(np.cos(tiltX_rad))**2 # Cartesian coordinate
        Z = np.cos(tiltX_rad)*np.cos(tiltY_rad) + (np.sin(tiltX_rad))**2*(np.sin(tiltY_rad))**2 # Cartesian coordinate
        phi_sensor_rad = -np.pi/2-np.arctan(Y/X) # spherical coordinate
        
        phi_sensor_rad[X > 0] = phi_sensor_rad[X > 0]+np.pi
        phi_sensor_rad[(X == 0) & (Y < 0)] = np.pi
        phi_sensor_rad[(X == 0) & (Y >= 0)] = 0
        phi_sensor_rad[phi_sensor_rad < 0] = phi_sensor_rad[phi_sensor_rad < 0]+2*np.pi
        
        phi_sensor_deg = phi_sensor_rad*180/np.pi # radians to degrees
        theta_sensor_rad = np.arccos(Z/(X**2+Y**2+Z**2)**0.5) # spherical coordinate (or actually total tilt of the sensor, i.e. 0 when horizontal)
        theta_sensor_deg = theta_sensor_rad*180/np.pi # radians to degrees
        
        
        
        # Calculating zenith and hour angle of the sun
        lat = float(ds.geometry[13:19]) #ds['gps_lat'].median().values
        lon = float(ds.geometry[6:12]) #ds['gps_lon'].median().values
        dates = ds.time.to_dataframe()
        dates.index = pd.to_datetime(dates['time'])
        dayofyear =dates.index.dayofyear.values
        hour = dates.index.hour.values
        minute = dates.index.minute.values
        
        d0_rad = 2*np.pi*(dayofyear+(hour+minute/60)/24-1)/365
        Declination_rad = np.arcsin(0.006918-0.399912*np.cos(d0_rad)+0.070257*np.sin(d0_rad)-0.006758*np.cos(2*d0_rad)+0.000907*np.sin(2*d0_rad)-0.002697*np.cos(3*d0_rad)+0.00148*np.sin(3*d0_rad))
        
        HourAngle_rad = 2*np.pi*(((hour+minute/60.)/24-0.5))# - lon/360) #- 15.*timezone/360.) ; NB: Make sure time is in UTC and longitude is positive when west! Hour angle should be 0 at noon.
        DirectionSun_deg = HourAngle_rad*180/np.pi-180 # This is 180 deg at noon (NH), as opposed to HourAngle.
        DirectionSun_deg[DirectionSun_deg < 0] = DirectionSun_deg[DirectionSun_deg < 0]+360
        DirectionSun_deg[DirectionSun_deg < 0] = DirectionSun_deg[DirectionSun_deg < 0]+360
        
        ZenithAngle_rad = np.arccos(np.cos(lat*np.pi/180)*np.cos(Declination_rad)*np.cos(HourAngle_rad) + np.sin(lat*np.pi/180)*np.sin(Declination_rad))
        ZenithAngle_deg = ZenithAngle_rad*180/np.pi
        sundown = ZenithAngle_deg >= 90
        SRtoa = 1372*np.cos(ZenithAngle_rad) # SRin at the top of the atmosphere
        SRtoa[sundown] = 0
        
        
        
        
        # correction factor for direct beam radiation
        CorFac = np.sin(Declination_rad) * np.sin(lat*np.pi/180.) * np.cos(theta_sensor_rad) \
                 -np.sin(Declination_rad) * np.cos(lat*np.pi/180.) * np.sin(theta_sensor_rad) * np.cos(phi_sensor_rad+np.pi) \
                +np.cos(Declination_rad) * np.cos(lat*np.pi/180.) * np.cos(theta_sensor_rad) * np.cos(HourAngle_rad) \
                +np.cos(Declination_rad) * np.sin(lat*np.pi/180.) * np.sin(theta_sensor_rad) * np.cos(phi_sensor_rad+np.pi) * np.cos(HourAngle_rad) \
                +np.cos(Declination_rad) * np.sin(theta_sensor_rad)*np.sin(phi_sensor_rad+np.pi)*np.sin(HourAngle_rad)
        
        CorFac = np.cos(ZenithAngle_rad)/CorFac
        no_correction = (CorFac <= 0) | ( ZenithAngle_deg > 90) # sun out of field of view upper sensor
        CorFac[no_correction] = 1
         # Calculating SRin over a horizontal surface corrected for station/sensor tilt
        CorFac_all = CorFac/(1-DifFrac+CorFac*DifFrac)
        SRin_cor = ds['dsr']*CorFac_all
        #srin_tilt_cor = SRin_cor.copy() # Debuggin
        # Calculating albedo based on albedo values when sun is in sight of the upper sensor
        AngleDif_deg = 180/np.pi*np.arccos(np.sin(ZenithAngle_rad)*np.cos(HourAngle_rad+np.pi)*np.sin(theta_sensor_rad)*np.cos(phi_sensor_rad) \
                                     +np.sin(ZenithAngle_rad)*np.sin(HourAngle_rad+np.pi)*np.sin(theta_sensor_rad)*np.sin(phi_sensor_rad) \
                                     +np.cos(ZenithAngle_rad)*np.cos(theta_sensor_rad)) # angle between sun and sensor
        
        
        albedo = ds['usr']/SRin_cor
        
        
        OKalbedos = (AngleDif_deg < 70) & (ZenithAngle_deg < 70) & (albedo < 1) & (albedo > 0)
        
        notOKalbedos = (AngleDif_deg >= 70) | (ZenithAngle_deg >= 70) | (albedo >= 1) | (albedo <= 0)
        
        albedo[notOKalbedos] = np.nan
        albedo = albedo.ffill('time')
        #albedo = interp1(datenumber,albedo,datenumber,'pchip') # interpolate over gaps - gives problems for discontinuous data sets, but is not the end of the world
        
        # Correcting SR using SRin when sun is in field of view of lower sensor assuming sensor measures only diffuse radiation
        sunonlowerdome = (AngleDif_deg >= 90) & (ZenithAngle_deg <= 90)
        SRin_cor[sunonlowerdome] = ds['dsr'][sunonlowerdome].values/DifFrac[sunonlowerdome]
        
        
        SRout_cor = ds['usr']
        SRout_cor[sunonlowerdome] = albedo[sunonlowerdome]*ds['dsr'][sunonlowerdome].values/DifFrac[sunonlowerdome]
        #srin_cor_dome = SRin_cor.copy() # debugging
        # Setting SRin and SRout to zero for solar zenith angles larger than 95 deg or either SRin or SRout are (less than) zero
        no_SR = (ZenithAngle_deg > 95) | (SRin_cor <= 0) | (SRout_cor <= 0)
        
        SRin_cor[no_SR] = 0
        SRout_cor[no_SR] = 0
        
        # Correcting SRin using more reliable SRout when sun not in sight of upper sensor
        
        #SRin_cor[no_correction] = SRout_cor[no_correction]/albedo[no_correction]
        SRin_cor[~notOKalbedos] = SRout_cor[~notOKalbedos]/albedo[~notOKalbedos]
        #SRin_cor_alb = SRin_cor.copy() # Debugging
        #SRin_cor = SRout_cor/albedo # What is done in the IDL code
        #albedo[notOKalbedos] = -999
        # Removing spikes by interpolation based on a simple top-of-the-atmosphere limitation
        
        SRin_cor_mark = SRin_cor.copy()
        TOA_crit_nopass = SRin_cor > 0.9*SRtoa+10
        
        SRin_cor[TOA_crit_nopass] = np.nan
        SRout_cor[TOA_crit_nopass] = np.nan
        
        
        SRin_cor_final = SRin_cor.copy()
         #The spike removal can disquise problems
        # Assign columns to ds file
        
        ds = ds.assign({'albedo':albedo, 'dsr_corr':SRin_cor, 'usr_corr':SRout_cor, 'cloud_cov':CloudCov})
        ds['I'] = ('time', SRtoa)
        ds['solar_zenith_angle'] = ('time', ZenithAngle_deg )
        
        
        
        #ds_test = ds.assign({'albedo':albedo,'srin_tilt_cor':srin_tilt_cor,'srin_cor_dome':srin_cor_dome, 'SRin_cor_alb':SRin_cor_alb ,'SRin_cor_final':SRin_cor_final,'SRin_cor_mark':SRin_cor_mark , 'dsr_corr':SRin_cor, 'usr_corr':SRout_cor, 'cloud_cov':CloudCov}) # debugging
        #ds['dsr_corr']=SRin_cor
        ds['dsr_corr'].attrs['long_name'] = ds['dsr'].long_name + " corrected"   
        #ds['usr_corr']=SRout_cor.copy()
        ds['usr_corr'].attrs['long_name'] = ds['usr'].long_name + " corrected"   
        
        #ds['cloud_cover']=CloudCov.copy()
        
        
    
    
    outpath = 'data_v1.0/L1/'+station+'/'
    outfile = infile[-14:-4]
    ds = ds.sel(time=ds.time.notnull())
    #ds_test = ds_test.sel(time=ds.time.notnull()) # debugging
    
    outpathfile = outpath + outfile + ".nc"
    #outpathfile_test = outpath + outfile + "_test.nc" #debug
    if os.path.exists(outpathfile): os.remove(outpathfile)
    ds.to_netcdf(outpathfile, mode='w', format='NETCDF4', compute=True)
    #ds_test.to_netcdf(outpathfile_test, mode='w', format='NETCDF4', compute=True) #debug
    
    
   



import pandas as pd
import xarray as xr
from glob import glob
import matplotlib.pyplot as plt
import os

datadir = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'
outfile = datadir+'zac_l/zac_l-2008-2022.nc'
with xr.open_mfdataset(datadir+'zac_l/zac_l-20??.nc') as ds:
    ds.to_netcdf(outfile)
    ds.to_dataframe().to_csv(datadir+'zac_l/zac_l-2008-2022.txt')

outfile = datadir+'zac_u/zac_u-2008-2022.nc'
with xr.open_mfdataset(datadir+'zac_u/zac_u-20??.nc') as ds:
    ds.to_netcdf(outfile)
    ds.to_dataframe().to_csv(datadir+'zac_u/zac_u-2008-2022.txt')

outfile = datadir+'zac_a/zac_a-2009-2020.nc'
with xr.open_mfdataset(datadir+'zac_a/zac_a-20??.nc') as ds:
    ds.to_netcdf(outfile)
    ds.to_dataframe().to_csv(datadir+'zac_a/zac_a-2009-2020.txt')
