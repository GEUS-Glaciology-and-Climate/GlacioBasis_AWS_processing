import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
station = 'zac_l'
filename = 'zac_l-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_l_path = datapath+filename

station = 'zac_u'
filename = 'zac_u-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_u_path = datapath+filename

station = 'zac_a'
filename = 'zac_a-2009-2020.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_a_path = datapath+filename
datadir = '/home/shl/OneDrive/projects/glaciobasis/data/aws_transmitted/'
station = 'zac_a'
filename = 'AWS_300234061218540.txt'
diagnostic = 'AWS_300234061218540-D.txt'
transmitted = pd.read_csv(datadir+station+'/'+filename,header=0,skiprows=[1,2,3],sep=',',engine='python')
transmitted.index = pd.to_datetime(transmitted[' timestamp'])
transmitted = transmitted.drop(' timestamp', axis = 1)
diag = pd.read_csv(datadir+station+'/'+diagnostic,header=0,skiprows=[1],sep=',',engine='python')
#print(data.keys().tolist())
#print(diag.keys())



transmitted = transmitted[:'2019-August-30'].astype(float)


# The transmitted data is divided into timestamps so 
timestep = transmitted.index[1:]-transmitted.index[0:-1]

timestep_hour = pd.DataFrame(timestep.components.hours)
timestep_hour.index =  transmitted.index[0:-1]
transmitted_hour = transmitted[0:-1][(timestep_hour==1).values]

timestep_day = timestep.components.days
timestep_day.index =  transmitted.index[0:-1]
transmitted_day = transmitted[0:-1][(timestep_day==1).values]




fig, ax = plt.subplots(1,1,figsize = (10,5))
with xr.open_dataset(zac_a_path) as ds:
    df = ds[['t_1']].to_dataframe()

count10min = df.resample('H').count()
temp_hour = df.resample('H').mean()
temp_hour[count10min<6] = np.nan
count_hours = temp_hour.resample('D').count()
count_hours.plot()
count10min.plot()
temp_day = temp_hour.resample('D').mean()
temp_day[count_hours<24 ] = np.nan

temp_day.plot()
temp_hour.plot(ax=ax)

temp_hour.to_csv('data_v1.0/gem_database/2022/preQC/zac_a_hour_temperature.csv', index = True, float_format = '%g')
temp_day.to_csv('data_v1.0/gem_database/2022/preQC/zac_a_day_temperature.csv', index = True, float_format = '%g')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
station = 'zac_l'
filename = 'zac_l-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_l_path = datapath+filename

station = 'zac_u'
filename = 'zac_u-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_u_path = datapath+filename

station = 'zac_a'
filename = 'zac_a-2009-2020.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_a_path = datapath+filename

fig, ax = plt.subplots(1,1,figsize = (10,5))
with xr.open_dataset(zac_a_path) as ds:
#    ds = ds[['dsr','usr','dlr','ulr']].to_dataframe()
#    ds['dsr'] = ds['dsr'].where(ds['dsr'] != -9999.)
#    ds['usr'] = ds['usr'].where(ds['usr'] != -9999.)
    ds = ds[['dsr','usr','dsr_corr','usr_corr','dlr','ulr','albedo','cloud_cov', 't_surf', 'I']].to_dataframe()
    ds['dsr_corr'] = ds['dsr_corr'].where(ds['dsr_corr'] != -9999.)
    ds['usr_corr'] = ds['usr_corr'].where(ds['usr_corr'] != -9999.)
    ds['dlr'] = ds['dlr'].where(ds['dlr'] != -9999.)
    ds['ulr'] = ds['ulr'].where(ds['ulr'] != -9999.)


count10min = ds.resample('H').count()
rad_hour = ds.resample('H').mean()
rad_hour[count10min<6] = np.nan
count_hours = rad_hour.resample('D').count()
rad_day = rad_hour.resample('D').mean()
rad_day[count_hours<24 ] = np.nan

rad_day.plot(ax=ax)

rad_hour.to_csv('data_v1.0/gem_database/2022/preQC/zac_a_hour_radiation.csv', index = True, float_format = '%g')
rad_day.to_csv('data_v1.0/gem_database/2022/preQC/zac_a_day_radiation.csv', index = True, float_format = '%g')


import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
station = 'zac_l'
filename = 'zac_l-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_l_path = datapath+filename

station = 'zac_u'
filename = 'zac_u-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_u_path = datapath+filename

station = 'zac_a'
filename = 'zac_a-2009-2020.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_a_path = datapath+filename

with xr.open_dataset(zac_a_path) as ds:
    df = ds[['rh']].to_dataframe()

count10min = df.resample('H').count()
rh_hour = df.resample('H').mean()
rh_hour[count10min<6] = np.nan
count_hours = rh_hour.resample('D').count()
count_hours.plot()
count10min.plot()
rh_day = rh_hour.resample('D').mean()
rh_day[count_hours<24 ] = np.nan

rh_day.plot()
rh_hour.plot()

rh_hour.to_csv('data_v1.0/gem_database/2022/preQC/zac_a_hour_relative_humidity.csv', index = True, float_format = '%g')
rh_day.to_csv('data_v1.0/gem_database/2022/preQC/zac_a_day_relative_humidity.csv', index = True, float_format = '%g')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
station = 'zac_l'
filename = 'zac_l-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_l_path = datapath+filename

station = 'zac_u'
filename = 'zac_u-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_u_path = datapath+filename

station = 'zac_a'
filename = 'zac_a-2009-2020.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_a_path = datapath+filename

with xr.open_dataset(zac_a_path) as ds:
    df = ds[['wspd']].to_dataframe()

count10min = df.resample('H').count()
wspd_hour = df.resample('H').mean()
wspd_hour[count10min<6] = np.nan
count_hours = wspd_hour.resample('D').count()
count_hours.plot()
count10min.plot()
wspd_day = wspd_hour.resample('D').mean()
wspd_day[count_hours<24 ] = np.nan

wspd_day.plot()
wspd_hour.plot()

wspd_hour.to_csv('data_v1.0/gem_database/2022/preQC/zac_a_hour_wind_speed.csv', index = True, float_format = '%g')
wspd_day.to_csv('data_v1.0/gem_database/2022/preQC/zac_a_day_wind_speed.csv', index = True, float_format = '%g')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
station = 'zac_l'
filename = 'zac_l-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_l_path = datapath+filename

station = 'zac_u'
filename = 'zac_u-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_u_path = datapath+filename

station = 'zac_a'
filename = 'zac_a-2009-2020.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_a_path = datapath+filename

with xr.open_dataset(zac_a_path) as ds:
    df = ds[['p']].to_dataframe()

count10min = df.resample('H').count()
p_hour = df.resample('H').mean()
p_hour[count10min<6] = np.nan
count_hours = p_hour.resample('D').count()
count_hours.plot()
count10min.plot()
p_day = p_hour.resample('D').mean()
p_day[count_hours<24 ] = np.nan

p_day.plot()
p_hour.plot()

p_hour.to_csv('data_v1.0/gem_database/2022/preQC/zac_a_hour_pressure.csv', index = True, float_format = '%g')
p_day.to_csv('data_v1.0/gem_database/2022/preQC/zac_a_day_pressure.csv', index = True, float_format = '%g')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
station = 'zac_l'
filename = 'zac_l-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_l_path = datapath+filename

station = 'zac_u'
filename = 'zac_u-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_u_path = datapath+filename

station = 'zac_a'
filename = 'zac_a-2009-2020.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_a_path = datapath+filename

with xr.open_dataset(zac_a_path) as ds:
    df = ds[['z_boom']].to_dataframe()

count10min = df.resample('H').count()
z_hour = df.resample('H').median()
z_hour[count10min<6] = np.nan
count_hours = z_hour.resample('D').count()
count_hours.plot()
count10min.plot()
z_day = z_hour.resample('D').median()
z_day[count_hours<24 ] = np.nan

z_day.plot()
z_hour.plot()

z_hour.to_csv('data_v1.0/gem_database/2022/preQC/zac_a_hour_boom_height.csv', index = True, float_format = '%g')
z_day.to_csv('data_v1.0/gem_database/2022/preQC/zac_a_day_boom_height.csv', index = True, float_format = '%g')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
station = 'zac_l'
filename = 'zac_l-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_l_path = datapath+filename

station = 'zac_u'
filename = 'zac_u-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_u_path = datapath+filename

station = 'zac_a'
filename = 'zac_a-2009-2020.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_a_path = datapath+filename

with xr.open_dataset(zac_a_path) as ds:
    df = ds[['z_stake']].to_dataframe()

count10min = df.resample('H').count()
z_hour = df.resample('H').median()
z_hour[count10min<6] = np.nan
count_hours = z_hour.resample('D').count()
count_hours.plot()
count10min.plot()
z_day = z_hour.resample('D').median()
z_day[count_hours<24 ] = np.nan

z_day.plot()
z_hour.plot()

z_hour.to_csv('data_v1.0/gem_database/2022/preQC/zac_a_hour_SR50_stake_height.csv', index = True, float_format = '%g')
z_day.to_csv('data_v1.0/gem_database/2022/preQC/zac_a_day_SR50_stake_height.csv', index = True, float_format = '%g')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
station = 'zac_l'
filename = 'zac_l-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_l_path = datapath+filename

station = 'zac_u'
filename = 'zac_u-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_u_path = datapath+filename

station = 'zac_a'
filename = 'zac_a-2009-2020.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_a_path = datapath+filename
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/gem_database/2022/'

temp_hour = pd.read_csv(datapath+'preQC/zac_a_hour_temperature.csv', parse_dates = True, index_col=0)


# Bad data deleted
temp_hour['2015-01-05':'2015-05-01'] = np.nan
temp_hour[:'2009-08-08 21:00'] = np.nan

count_hours = temp_hour.resample('D').count()
temp_day = temp_hour.resample('D').mean()
temp_day[count_hours<24 ] = np.nan

temp_hour.to_csv('data_v1.0/gem_database/2022/zac_a_hour_temperature.csv', index = True, float_format = '%g')
temp_day.to_csv('data_v1.0/gem_database/2022/zac_a_day_temperature.csv', index = True, float_format = '%g')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
station = 'zac_l'
filename = 'zac_l-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_l_path = datapath+filename

station = 'zac_u'
filename = 'zac_u-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_u_path = datapath+filename

station = 'zac_a'
filename = 'zac_a-2009-2020.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_a_path = datapath+filename
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/gem_database/2022/'

rad_hour = pd.read_csv(datapath+'preQC/zac_a_hour_radiation.csv', parse_dates = True, index_col=0)

# Deliting bad data: Out of bounds
maximum = 1000
variable = 'dsr_corr'
rad_hour[variable][rad_hour[variable]>maximum] = np.nan
rad_hour[variable][rad_hour[variable]<0] = np.nan

variable = 'usr_corr'
albedo = rad_hour['usr_corr']/rad_hour['dsr_corr']
rad_hour[variable][albedo>1] = np.nan
variable = 'albedo'
rad_hour[variable][albedo>1] = np.nan

variable = 'ulr'
rad_hour[variable][rad_hour['ulr']<150] = np.nan
variable = 'cloud_cov'
rad_hour[variable][rad_hour['ulr']<150] = np.nan
variable = 't_surf'
rad_hour[variable][rad_hour['ulr']<150] = np.nan

variable = 'dlr'
rad_hour[variable][rad_hour['dlr']<120] = np.nan
variable = 'cloud_cov'
rad_hour[variable][rad_hour['dlr']<120] = np.nan 
variable = 't_surf'
rad_hour[variable][rad_hour['dlr']<120] = np.nan 

# Deleting bad data manually
rad_hour['2015-01-01':'2015-05-01'] = np.nan

# Then calculate daily averages again
count_hours = rad_hour.resample('D').count()
rad_day = rad_hour.resample('D').mean()
rad_day[count_hours<24 ] = np.nan

rad_hour.to_csv('data_v1.0/gem_database/2022/zac_a_hour_radiation.csv', index = True, float_format = '%g')
rad_day.to_csv('data_v1.0/gem_database/2022/zac_a_day_radiation.csv', index = True, float_format = '%g')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/gem_database/2022/'
rh_hour = pd.read_csv(datapath+'preQC/zac_a_hour_relative_humidity.csv', parse_dates = True, index_col=0)


# Outliers
#rh_hour = rh_hour.where(rh_hour['rh']<= 110., np.nan)
rh_hour = rh_hour.where(rh_hour['rh']>= 0., np.nan)
# Deleting bad data manually
rh_hour['2014-12-01':'2015-05-01'] = np.nan
rh_hour['2011':'2014'] = np.nan
rh_hour['2016-04-01':] = np.nan

count_hours = rh_hour.resample('D').count()
rh_day = rh_hour.resample('D').mean()
rh_day[count_hours<24 ] = np.nan

rh_hour.to_csv('data_v1.0/gem_database/2022/zac_a_hour_relative_humidity.csv', index = True, float_format = '%g')
rh_day.to_csv('data_v1.0/gem_database/2022/zac_a_day_relative_humidity.csv', index = True, float_format = '%g')
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/gem_database/2022/'
wspd_hour = pd.read_csv(datapath+'preQC/zac_a_hour_wind_speed.csv', parse_dates = True, index_col=0)


# Bad data
startday = datetime.datetime(2020,8,15)
endday = datetime.datetime(2022,4,21)
wspd_hour['2020-August-15':'2021-July-21'] = np.nan
wspd_hour['2014-12-01':'2015-05-01'] = np.nan

count_hours = wspd_hour.resample('D').count()
wspd_day = wspd_hour.resample('D').mean()
wspd_day[count_hours<24 ] = np.nan

wspd_hour.to_csv('data_v1.0/gem_database/2022/zac_a_hour_wind_speed.csv', index = True, float_format = '%g')
wspd_day.to_csv('data_v1.0/gem_database/2022/zac_a_day_wind_speed.csv', index = True, float_format = '%g')
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/gem_database/2022/'
p_hour = pd.read_csv(datapath+'preQC/zac_a_hour_pressure.csv', parse_dates = True, index_col=0)

#Outliers

p_hour = p_hour.where(p_hour['p']> 800., np.nan)


count_hours = p_hour.resample('D').count()
p_day = p_hour.resample('D').mean()
p_day[count_hours<24 ] = np.nan

p_hour.to_csv('data_v1.0/gem_database/2022/zac_a_hour_pressure.csv', index = True, float_format = '%g')
p_day.to_csv('data_v1.0/gem_database/2022/zac_a_day_pressure.csv', index = True, float_format = '%g')
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/gem_database/2022/'
boom_hour = pd.read_csv(datapath+'preQC/zac_a_hour_boom_height.csv', parse_dates = True, index_col=0)

#Outliers

boom_hour = boom_hour.where(boom_hour['z_boom']> 0.1, np.nan)
boom_hour = boom_hour.where(boom_hour['z_boom']< 4, np.nan)

# Bad data
boom_hour['2012-April-16':'2013-August-28'] = np.nan
boom_hour['2013-December-20':'2014-April-22'] = np.nan
boom_hour['2015-January-2':'2016-April-22'] = np.nan
boom_hour['2018-April-20':'2018-April-24'] = np.nan


count_hours = boom_hour.resample('D').count()
boom_day = boom_hour.resample('D').median()
boom_day[count_hours<24 ] = np.nan

boom_hour.to_csv('data_v1.0/gem_database/2022/zac_a_hour_boom_height.csv', index = True, float_format = '%g')
boom_day.to_csv('data_v1.0/gem_database/2022/zac_a_day_boom_height.csv', index = True, float_format = '%g')


import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
station = 'zac_l'
filename = 'zac_l-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_l_path = datapath+filename

station = 'zac_u'
filename = 'zac_u-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_u_path = datapath+filename

station = 'zac_a'
filename = 'zac_a-2009-2020.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_a_path = datapath+filename

with xr.open_dataset(zac_l_path) as ds:
    df = ds[['t_1']].to_dataframe()

count10min = df.resample('H').count()
temp_hour = df.resample('H').mean()
temp_hour[count10min<6] = np.nan
count_hours = temp_hour.resample('D').count()
count_hours.plot()
count10min.plot()
temp_day = temp_hour.resample('D').mean()
temp_day[count_hours<24 ] = np.nan

temp_day.plot()
temp_hour.plot()

temp_hour.to_csv('data_v1.0/gem_database/2022/preQC/zac_l_hour_temperature.csv', index = True, float_format = '%g')
temp_day.to_csv('data_v1.0/gem_database/2022/preQC/zac_l_day_temperature.csv', index = True, float_format = '%g')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
station = 'zac_l'
filename = 'zac_l-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_l_path = datapath+filename

station = 'zac_u'
filename = 'zac_u-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_u_path = datapath+filename

station = 'zac_a'
filename = 'zac_a-2009-2020.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_a_path = datapath+filename

fig, ax = plt.subplots(1,1,figsize = (10,5))
with xr.open_dataset(zac_l_path) as ds:
    #print(ds)
    ds = ds[['dsr','usr','dsr_corr','usr_corr','dlr','ulr','albedo','cloud_cov','t_surf', 'I']].to_dataframe()

    ds['dsr_corr'] = ds['dsr_corr'].where(ds['dsr_corr'] != -9999.)
    ds['usr_corr'] = ds['usr_corr'].where(ds['usr_corr'] != -9999.)
    ds['dlr'] = ds['dlr'].where(ds['dlr'] != -9999.)
    ds['ulr'] = ds['ulr'].where(ds['ulr'] != -9999.)#.to_dataframe()
    ds['ulr'] = ds['ulr'].where(ds['ulr'] < 10000.)#.to_dataframe()
    ds['dlr'] = ds['dlr'].where(ds['dlr'] < 10000.)#.to_dataframe()

count10min = ds.resample('H').count()
rad_hour = ds.resample('H').mean()
rad_hour[count10min<6] = np.nan
count_hours = rad_hour.resample('D').count()
count_hours.plot()
rad_day = rad_hour.resample('D').mean()
rad_day[count_hours<24 ] = np.nan

rad_day.plot(ax=ax)

rad_hour.to_csv('data_v1.0/gem_database/2022/preQC/zac_l_hour_radiation.csv', index = True, float_format = '%g')
rad_day.to_csv('data_v1.0/gem_database/2022/preQC/zac_l_day_radiation.csv', index = True, float_format = '%g')


import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
station = 'zac_l'
filename = 'zac_l-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_l_path = datapath+filename

station = 'zac_u'
filename = 'zac_u-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_u_path = datapath+filename

station = 'zac_a'
filename = 'zac_a-2009-2020.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_a_path = datapath+filename

with xr.open_dataset(zac_l_path) as ds:
    df = ds[['rh']].to_dataframe()

count10min = df.resample('H').count()
rh_hour = df.resample('H').mean()
rh_hour[count10min<6] = np.nan
count_hours = rh_hour.resample('D').count()
count_hours.plot()
count10min.plot()
rh_day = rh_hour.resample('D').mean()
rh_day[count_hours<24 ] = np.nan

rh_day.plot()
rh_hour.plot()

rh_hour.to_csv('data_v1.0/gem_database/2022/preQC/zac_l_hour_relative_humidity.csv', index = True, float_format = '%g')
rh_day.to_csv('data_v1.0/gem_database/2022/preQC/zac_l_day_relative_humidity.csv', index = True, float_format = '%g')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
station = 'zac_l'
filename = 'zac_l-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_l_path = datapath+filename

station = 'zac_u'
filename = 'zac_u-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_u_path = datapath+filename

station = 'zac_a'
filename = 'zac_a-2009-2020.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_a_path = datapath+filename

with xr.open_dataset(zac_l_path) as ds:
    df = ds[['wspd']].to_dataframe()

count10min = df.resample('H').count()
wspd_hour = df.resample('H').mean()
wspd_hour[count10min<6] = np.nan
count_hours = wspd_hour.resample('D').count()
count_hours.plot()
count10min.plot()
wspd_day = wspd_hour.resample('D').mean()
wspd_day[count_hours<24 ] = np.nan

wspd_day.plot()
wspd_hour.plot()

wspd_hour.to_csv('data_v1.0/gem_database/2022/preQC/zac_l_hour_wind_speed.csv', index = True, float_format = '%g')
wspd_day.to_csv('data_v1.0/gem_database/2022/preQC/zac_l_day_wind_speed.csv', index = True, float_format = '%g')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
station = 'zac_l'
filename = 'zac_l-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_l_path = datapath+filename

station = 'zac_u'
filename = 'zac_u-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_u_path = datapath+filename

station = 'zac_a'
filename = 'zac_a-2009-2020.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_a_path = datapath+filename

with xr.open_dataset(zac_l_path) as ds:
    df = ds[['p']].to_dataframe()

count10min = df.resample('H').count()
p_hour = df.resample('H').mean()
p_hour[count10min<6] = np.nan
count_hours = p_hour.resample('D').count()
count_hours.plot()
count10min.plot()
p_day = p_hour.resample('D').mean()
p_day[count_hours<24 ] = np.nan

p_day.plot()
p_hour.plot()

p_hour.to_csv('data_v1.0/gem_database/2022/preQC/zac_l_hour_pressure.csv', index = True, float_format = '%g')
p_day.to_csv('data_v1.0/gem_database/2022/preQC/zac_l_day_pressure.csv', index = True, float_format = '%g')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
station = 'zac_l'
filename = 'zac_l-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_l_path = datapath+filename

station = 'zac_u'
filename = 'zac_u-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_u_path = datapath+filename

station = 'zac_a'
filename = 'zac_a-2009-2020.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_a_path = datapath+filename

with xr.open_dataset(zac_l_path) as ds:
    df = ds[['z_boom']].to_dataframe()

count10min = df.resample('H').count()
z_hour = df.resample('H').median()
z_hour[count10min<6] = np.nan
count_hours = z_hour.resample('D').count()
count_hours.plot()
count10min.plot()
z_day = z_hour.resample('D').median()
z_day[count_hours<24 ] = np.nan

z_day.plot()
z_hour.plot()

z_hour.to_csv('data_v1.0/gem_database/2022/preQC/zac_l_hour_boom_height.csv', index = True, float_format = '%g')
z_day.to_csv('data_v1.0/gem_database/2022/preQC/zac_l_day_boom_height.csv', index = True, float_format = '%g')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
station = 'zac_l'
filename = 'zac_l-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_l_path = datapath+filename

station = 'zac_u'
filename = 'zac_u-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_u_path = datapath+filename

station = 'zac_a'
filename = 'zac_a-2009-2020.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_a_path = datapath+filename

with xr.open_dataset(zac_l_path) as ds:
    df = ds[['z_stake']].to_dataframe()

count10min = df.resample('H').count()
z_hour = df.resample('H').median()
z_hour[count10min<6] = np.nan
count_hours = z_hour.resample('D').count()
count_hours.plot()
count10min.plot()
z_day = z_hour.resample('D').median()
z_day[count_hours<24 ] = np.nan

z_day.plot()
z_hour.plot()

z_hour.to_csv('data_v1.0/gem_database/2022/preQC/zac_l_hour_SR50_stake_height.csv', index = True, float_format = '%g')
z_day.to_csv('data_v1.0/gem_database/2022/preQC/zac_l_day_SR50_stake_height.csv', index = True, float_format = '%g')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
station = 'zac_l'
filename = 'zac_l-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_l_path = datapath+filename

station = 'zac_u'
filename = 'zac_u-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_u_path = datapath+filename

station = 'zac_a'
filename = 'zac_a-2009-2020.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_a_path = datapath+filename
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/gem_database/2022/'

temp_hour = pd.read_csv(datapath+'preQC/zac_l_hour_temperature.csv', parse_dates = True, index_col=0)
#Deling bad data

count_hours = temp_hour.resample('D').count()
temp_day = temp_hour.resample('D').mean()
temp_day[count_hours<24 ] = np.nan


temp_hour.to_csv('data_v1.0/gem_database/2022/zac_l_hour_temperature.csv', index = True, float_format = '%g')
temp_day.to_csv('data_v1.0/gem_database/2022/zac_l_day_temperature.csv', index = True, float_format = '%g')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
station = 'zac_l'
filename = 'zac_l-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_l_path = datapath+filename

station = 'zac_u'
filename = 'zac_u-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_u_path = datapath+filename

station = 'zac_a'
filename = 'zac_a-2009-2020.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_a_path = datapath+filename
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/gem_database/2022/'

rad_hour = pd.read_csv(datapath+'preQC/zac_l_hour_radiation.csv', parse_dates = True, index_col=0)


# Deliting bad data: Out of bounds
maximum = 1000


variable = 'dsr_corr'
rad_hour[variable][rad_hour[variable]>maximum] = np.nan
rad_hour[variable][rad_hour[variable]<0] = np.nan

variable = 'usr_corr'
albedo = rad_hour['usr_corr']/rad_hour['dsr_corr']
rad_hour[variable][albedo>1] = np.nan
variable = 'albedo'
rad_hour[variable][albedo>1] = np.nan

variable = 'ulr'
rad_hour[variable][rad_hour['ulr']<150] = np.nan
variable = 'cloud_cov'
rad_hour[variable][rad_hour['ulr']<150] = np.nan
variable = 't_surf'
rad_hour[variable][rad_hour['ulr']<150] = np.nan

variable = 'dlr'
rad_hour[variable][rad_hour['dlr']<120] = np.nan
variable = 'cloud_cov'
rad_hour[variable][rad_hour['dlr']<120] = np.nan 
variable = 't_surf'
rad_hour[variable][rad_hour['dlr']<120] = np.nan 

# Deleting bad data manually
rad_hour['usr_corr']['2021-01-01':'2021-07-21'] = np.nan
rad_hour['2020-07-01':'2021-07-22'] = np.nan
rad_hour['dsr_corr']['22-Jan-2020':] = np.nan
rad_hour['usr_corr']['22-Jan-2020':] = np.nan
rad_hour['albedo']['22-Jan-2020':] = np.nan

# Then calculate daily averages again
count_hours = rad_hour.resample('D').count()
rad_day = rad_hour.resample('D').mean()
rad_day[count_hours<24 ] = np.nan

rad_hour.to_csv('data_v1.0/gem_database/2022/zac_l_hour_radiation.csv', index = True, float_format = '%g')
rad_day.to_csv('data_v1.0/gem_database/2022/zac_l_day_radiation.csv', index = True, float_format = '%g')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/gem_database/2022/'
rad_hour = pd.read_csv(datapath+'preQC/zac_l_hour_relative_humidity.csv', parse_dates = True, index_col=0)


# Outliers
rad_hour = rad_hour.where(rad_hour['rh']<= 100., np.nan)
rad_hour = rad_hour.where(rad_hour['rh']>= 0., np.nan)


count_hours = rad_hour.resample('D').count()
rad_day = rad_hour.resample('D').mean()
rad_day[count_hours<24 ] = np.nan

rad_hour.to_csv('data_v1.0/gem_database/2022/zac_l_hour_relative_humidity.csv', index = True, float_format = '%g')
rad_day.to_csv('data_v1.0/gem_database/2022/zac_l_day_relative_humidity.csv', index = True, float_format = '%g')
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/gem_database/2022/'
wspd_hour = pd.read_csv(datapath+'preQC/zac_l_hour_wind_speed.csv', parse_dates = True, index_col=0)



count_hours = wspd_hour.resample('D').count()
wspd_day = wspd_hour.resample('D').mean()
wspd_day[count_hours<24 ] = np.nan

wspd_hour.to_csv('data_v1.0/gem_database/2022/zac_l_hour_wind_speed.csv', index = True, float_format = '%g')
wspd_day.to_csv('data_v1.0/gem_database/2022/zac_l_day_wind_speed.csv', index = True, float_format = '%g')
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/gem_database/2022/'
p_hour = pd.read_csv(datapath+'preQC/zac_l_hour_pressure.csv', parse_dates = True, index_col=0)

#Outliers

p_hour = p_hour.where(p_hour['p']> 870., np.nan)


p_hour['2016-February-26':'2016-March-1'] = np.nan
p_hour['2017-January-5':'2017-February-22'] = np.nan
p_hour['2018-February-21':'2018-February-28'] = np.nan


count_hours = p_hour.resample('D').count()
p_day = p_hour.resample('D').mean()
p_day[count_hours<24 ] = np.nan

p_hour.to_csv('data_v1.0/gem_database/2022/zac_l_hour_pressure.csv', index = True, float_format = '%g')
p_day.to_csv('data_v1.0/gem_database/2022/zac_l_day_pressure.csv', index = True, float_format = '%g')
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/gem_database/2022/'
boom_hour = pd.read_csv(datapath+'preQC/zac_l_hour_boom_height.csv', parse_dates = True, index_col=0)

#Outliers

boom_hour = boom_hour.where(boom_hour['z_boom']> 0.1, np.nan)
boom_hour = boom_hour.where(boom_hour['z_boom']< 2.75, np.nan)

# Bad data
boom_hour['2011-January-25':'2013-May-3'] = np.nan
boom_hour['2019':'2020'] = np.nan

count_hours = boom_hour.resample('D').count()
boom_day = boom_hour.resample('D').median()
boom_day[count_hours<24 ] = np.nan

boom_hour.to_csv('data_v1.0/gem_database/2022/zac_l_hour_boom_height.csv', index = True, float_format = '%g')
boom_day.to_csv('data_v1.0/gem_database/2022/zac_l_day_boom_height.csv', index = True, float_format = '%g')


import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
station = 'zac_l'
filename = 'zac_l-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_l_path = datapath+filename

station = 'zac_u'
filename = 'zac_u-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_u_path = datapath+filename

station = 'zac_a'
filename = 'zac_a-2009-2020.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_a_path = datapath+filename

with xr.open_dataset(zac_u_path) as ds:
    df = ds[['t_1']].to_dataframe()

count10min = df.resample('H').count()
temp_hour = df.resample('H').mean()
temp_hour[count10min<6] = np.nan
count_hours = temp_hour.resample('D').count()
count_hours.plot()
count10min.plot()
temp_day = temp_hour.resample('D').mean()
temp_day[count_hours<24 ] = np.nan

temp_day.plot()
temp_hour.plot()

temp_hour.to_csv('data_v1.0/gem_database/2022/preQC/zac_u_hour_temperature.csv', index = True, float_format = '%g')
temp_day.to_csv('data_v1.0/gem_database/2022/preQC/zac_u_day_temperature.csv', index = True, float_format = '%g')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
station = 'zac_l'
filename = 'zac_l-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_l_path = datapath+filename

station = 'zac_u'
filename = 'zac_u-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_u_path = datapath+filename

station = 'zac_a'
filename = 'zac_a-2009-2020.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_a_path = datapath+filename

#fig, ax = plt.subplots(1,1,figsize = (10,5))
with xr.open_dataset(zac_u_path) as ds:
    #ds = ds[['dsr','usr','dlr','ulr']].to_dataframe()
    #ds['dsr'] = ds['dsr'].where(ds['dsr'] != -9999.)
    ds = ds[['dsr','usr','dsr_corr','usr_corr','dlr','ulr','albedo','cloud_cov','t_surf', 'I']].to_dataframe()

    ds['dsr_corr'] = ds['dsr_corr'].where(ds['dsr_corr'] != -9999.)
    ds['usr_corr'] = ds['usr_corr'].where(ds['usr_corr'] != -9999.)
    #ds['usr'] = ds['usr'].where(ds['usr'] != -9999.)
    ds['dlr'] = ds['dlr'].where(ds['dlr'] != -9999.)
    ds['ulr'] = ds['ulr'].where(ds['ulr'] != -9999.)#.to_dataframe()
    ds['ulr'] = ds['ulr'].where(ds['ulr'] < 10000.)#.to_dataframe()
    ds['dlr'] = ds['dlr'].where(ds['dlr'] < 10000.)#.to_dataframe()


count10min = ds.resample('H').count()
rad_hour = ds.resample('H').mean()
rad_hour[count10min<6] = np.nan
count_hours = rad_hour.resample('D').count()
rad_day = rad_hour.resample('D').mean()
rad_day[count_hours<24 ] = np.nan
#count_hours['April-2014'].plot()
#count10min['April-2014'].plot()
#rad_hour['April-14-2014':'April-15-2014'].plot(ax=ax)

rad_hour.to_csv('data_v1.0/gem_database/2022/preQC/zac_u_hour_radiation.csv', index = True, float_format = '%g')
rad_day.to_csv('data_v1.0/gem_database/2022/preQC/zac_u_day_radiation.csv', index = True, float_format = '%g')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
station = 'zac_l'
filename = 'zac_l-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_l_path = datapath+filename

station = 'zac_u'
filename = 'zac_u-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_u_path = datapath+filename

station = 'zac_a'
filename = 'zac_a-2009-2020.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_a_path = datapath+filename

with xr.open_dataset(zac_u_path) as ds:
    df = ds[['rh']].to_dataframe()

count10min = df.resample('H').count()
rh_hour = df.resample('H').mean()
rh_hour[count10min<6] = np.nan
count_hours = rh_hour.resample('D').count()
count_hours.plot()
count10min.plot()
rh_day = rh_hour.resample('D').mean()
rh_day[count_hours<24 ] = np.nan

rh_day.plot()
rh_hour.plot()

rh_hour.to_csv('data_v1.0/gem_database/2022/preQC/zac_u_hour_relative_humidity.csv', index = True, float_format = '%g')
rh_day.to_csv('data_v1.0/gem_database/2022/preQC/zac_u_day_relative_humidity.csv', index = True, float_format = '%g')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
station = 'zac_l'
filename = 'zac_l-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_l_path = datapath+filename

station = 'zac_u'
filename = 'zac_u-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_u_path = datapath+filename

station = 'zac_a'
filename = 'zac_a-2009-2020.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_a_path = datapath+filename

with xr.open_dataset(zac_u_path) as ds:
    df = ds[['wspd']].to_dataframe()

count10min = df.resample('H').count()
wspd_hour = df.resample('H').mean()
wspd_hour[count10min<6] = np.nan
count_hours = wspd_hour.resample('D').count()
count_hours.plot()
count10min.plot()
wspd_day = wspd_hour.resample('D').mean()
wspd_day[count_hours<24 ] = np.nan

wspd_day.plot()
wspd_hour.plot()

wspd_hour.to_csv('data_v1.0/gem_database/2022/preQC/zac_u_hour_wind_speed.csv', index = True, float_format = '%g')
wspd_day.to_csv('data_v1.0/gem_database/2022/preQC/zac_u_day_wind_speed.csv', index = True, float_format = '%g')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
station = 'zac_l'
filename = 'zac_l-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_l_path = datapath+filename

station = 'zac_u'
filename = 'zac_u-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_u_path = datapath+filename

station = 'zac_a'
filename = 'zac_a-2009-2020.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_a_path = datapath+filename

with xr.open_dataset(zac_u_path) as ds:
    df = ds[['p']].to_dataframe()

count10min = df.resample('H').count()
p_hour = df.resample('H').mean()
p_hour[count10min<6] = np.nan
count_hours = p_hour.resample('D').count()
count_hours.plot()
count10min.plot()
p_day = p_hour.resample('D').mean()
p_day[count_hours<24 ] = np.nan

p_day.plot()
p_hour.plot()

p_hour.to_csv('data_v1.0/gem_database/2022/preQC/zac_u_hour_pressure.csv', index = True, float_format = '%g')
p_day.to_csv('data_v1.0/gem_database/2022/preQC/zac_u_day_pressure.csv', index = True, float_format = '%g')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
station = 'zac_l'
filename = 'zac_l-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_l_path = datapath+filename

station = 'zac_u'
filename = 'zac_u-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_u_path = datapath+filename

station = 'zac_a'
filename = 'zac_a-2009-2020.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_a_path = datapath+filename

with xr.open_dataset(zac_u_path) as ds:
    df = ds[['z_boom']].to_dataframe()

count10min = df.resample('H').count()
z_hour = df.resample('H').median()
z_hour[count10min<6] = np.nan
count_hours = z_hour.resample('D').count()
count_hours.plot()
count10min.plot()
z_day = z_hour.resample('D').median()
z_day[count_hours<24 ] = np.nan

z_day.plot()
z_hour.plot()

z_hour.to_csv('data_v1.0/gem_database/2022/preQC/zac_u_hour_boom_height.csv', index = True, float_format = '%g')
z_day.to_csv('data_v1.0/gem_database/2022/preQC/zac_u_day_boom_height.csv', index = True, float_format = '%g')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
station = 'zac_l'
filename = 'zac_l-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_l_path = datapath+filename

station = 'zac_u'
filename = 'zac_u-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_u_path = datapath+filename

station = 'zac_a'
filename = 'zac_a-2009-2020.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_a_path = datapath+filename

with xr.open_dataset(zac_u_path) as ds:
    df = ds[['z_stake']].to_dataframe()

count10min = df.resample('H').count()
z_hour = df.resample('H').median()
z_hour[count10min<6] = np.nan
count_hours = z_hour.resample('D').count()
count_hours.plot()
count10min.plot()
z_day = z_hour.resample('D').median()
z_day[count_hours<24 ] = np.nan

z_day.plot()
z_hour.plot()

z_hour.to_csv('data_v1.0/gem_database/2022/preQC/zac_u_hour_SR50_stake_height.csv', index = True, float_format = '%g')
z_day.to_csv('data_v1.0/gem_database/2022/preQC/zac_u_day_SR50_stake_height.csv', index = True, float_format = '%g')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
station = 'zac_l'
filename = 'zac_l-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_l_path = datapath+filename

station = 'zac_u'
filename = 'zac_u-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_u_path = datapath+filename

station = 'zac_a'
filename = 'zac_a-2009-2020.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_a_path = datapath+filename

datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/gem_database/2022/'
temp_hour = pd.read_csv(datapath+'preQC/zac_u_hour_temperature.csv', parse_dates = True, index_col=0)

# Bad data deleted
temp_hour['2020-09-15':'2021-08-01'] = np.nan
temp_hour['2014-10-30':'2015-12-31'] = np.nan


count_hours = temp_hour.resample('D').count()
temp_day = temp_hour.resample('D').mean()
temp_day[count_hours<24 ] = np.nan

temp_hour.to_csv('data_v1.0/gem_database/2022/zac_u_hour_temperature.csv', index = True, float_format = '%g')
temp_day.to_csv('data_v1.0/gem_database/2022/zac_u_day_temperature.csv', index = True, float_format = '%g')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
station = 'zac_l'
filename = 'zac_l-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_l_path = datapath+filename

station = 'zac_u'
filename = 'zac_u-2008-2022.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_u_path = datapath+filename

station = 'zac_a'
filename = 'zac_a-2009-2020.nc'
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/L1/'+station+'/'
zac_a_path = datapath+filename
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/gem_database/2022/'

rad_hour = pd.read_csv(datapath+'preQC/zac_u_hour_radiation.csv', parse_dates = True, index_col=0)

rad_hour.plot()

# Deliting bad data: Out of bounds
maximum = 1000
variable = 'dsr_corr'
rad_hour[variable][rad_hour[variable]>maximum] = np.nan
rad_hour[variable][rad_hour[variable]<0] = np.nan

variable = 'usr_corr'
albedo = rad_hour['usr_corr']/rad_hour['dsr_corr']
rad_hour[variable][albedo>1] = np.nan
variable = 'albedo'
rad_hour[variable][albedo>1] = np.nan

variable = 'ulr'
rad_hour[variable][rad_hour['ulr']<150] = np.nan
variable = 'cloud_cov'
rad_hour[variable][rad_hour['ulr']<150] = np.nan
variable = 't_surf'
rad_hour[variable][rad_hour['ulr']<150] = np.nan

variable = 'dlr'
rad_hour[variable][rad_hour['dlr']<120] = np.nan
variable = 'cloud_cov'
rad_hour[variable][rad_hour['dlr']<120] = np.nan 
variable = 't_surf'
rad_hour[variable][rad_hour['dlr']<120] = np.nan 


# Deleting bad data manually
rad_hour['2020-08-15':'2021-08-01'] = np.nan # Station was tilted
rad_hour['2015-01-01':'2015-12-31'] = np.nan

rad_hour= rad_hour['2012-05-05':]

rad_hour.plot()
# Then calculate daily averages again
count_hours = rad_hour.resample('D').count()

rad_day = rad_hour.resample('D').mean()
rad_day[count_hours<24 ] = np.nan
rad_hour.to_csv('data_v1.0/gem_database/2022/zac_u_hour_radiation.csv', index = True, float_format = '%g')
rad_day.to_csv('data_v1.0/gem_database/2022/zac_u_day_radiation.csv', index = True, float_format = '%g')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/gem_database/2022/'
rh_hour = pd.read_csv(datapath+'preQC/zac_u_hour_relative_humidity.csv', parse_dates = True, index_col=0)


# Outliers
rh_hour = rh_hour.where(rh_hour['rh']<= 100., np.nan)
rh_hour = rh_hour.where(rh_hour['rh']>= 0., np.nan)


# Bad data
rh_hour['2020-August-15':'2021-July-21'] = np.nan
rh_hour['2014-10-30':'2015-12-31'] = np.nan


count_hours = rh_hour.resample('D').count()
rh_day = rh_hour.resample('D').mean()
rh_day[count_hours<24 ] = np.nan

rh_hour.to_csv('data_v1.0/gem_database/2022/zac_u_hour_relative_humidity.csv', index = True, float_format = '%g')
rh_day.to_csv('data_v1.0/gem_database/2022/zac_u_day_relative_humidity.csv', index = True, float_format = '%g')
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/gem_database/2022/'
wspd_hour = pd.read_csv(datapath+'preQC/zac_u_hour_wind_speed.csv', parse_dates = True, index_col=0)


# Bad data
startday = datetime.datetime(2020,8,15)
endday = datetime.datetime(2022,4,21)
wspd_hour['2020-August-15':'2022-July-21'] = np.nan
#wspd_hour['2014-10-30':'2015-12-31'] = np.nan

count_hours = wspd_hour.resample('D').count()
wspd_day = wspd_hour.resample('D').mean()
wspd_day[count_hours<24 ] = np.nan

wspd_hour.to_csv('data_v1.0/gem_database/2022/zac_u_hour_wind_speed.csv', index = True, float_format = '%g')
wspd_day.to_csv('data_v1.0/gem_database/2022/zac_u_day_wind_speed.csv', index = True, float_format = '%g')
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/gem_database/2022/'
p_hour = pd.read_csv(datapath+'preQC/zac_u_hour_pressure.csv', parse_dates = True, index_col=0)

#Outliers

p_hour = p_hour.where(p_hour['p']> 850., np.nan)

# Bad data
p_hour['2014-10-30':'2015-12-31'] = np.nan
p_hour['2016-April-4':'2016-April-18'] = np.nan
p_hour['2017-January-23':'2017-March-2'] = np.nan


count_hours = p_hour.resample('D').count()
p_day = p_hour.resample('D').mean()
p_day[count_hours<24 ] = np.nan

p_hour.to_csv('data_v1.0/gem_database/2022/zac_u_hour_pressure.csv', index = True, float_format = '%g')
p_day.to_csv('data_v1.0/gem_database/2022/zac_u_day_pressure.csv', index = True, float_format = '%g')
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
datapath = '/home/shl/OneDrive/projects/aws_processing_v1.0/data_v1.0/gem_database/2022/'
boom_hour = pd.read_csv(datapath+'preQC/zac_u_hour_boom_height.csv', parse_dates = True, index_col=0)

#Outliers

boom_hour = boom_hour.where(boom_hour['z_boom']> 0.1, np.nan)
boom_hour = boom_hour.where(boom_hour['z_boom']< 2.75, np.nan)

# Bad data
boom_hour['2012-January-1':'2016-April-20'] = np.nan
boom_hour['2019-June-20':'2021-July-26'] = np.nan
#boom_hour['2014-10-30':'2015-12-31'] = np.nan



count_hours = boom_hour.resample('D').count()
boom_day = boom_hour.resample('D').median()
boom_day[count_hours<24 ] = np.nan

boom_hour.to_csv('data_v1.0/gem_database/2022/zac_u_hour_boom_height.csv', index = True, float_format = '%g')
boom_day.to_csv('data_v1.0/gem_database/2022/zac_u_day_boom_height.csv', index = True, float_format = '%g')
