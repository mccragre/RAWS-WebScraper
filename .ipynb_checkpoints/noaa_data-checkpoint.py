import requests
from datetime import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from geopy.distance import distance

data_endpoint = f'https://www.ncdc.noaa.gov/cdo-web/api/v2/data?'
stat_endpoint = f'https://www.ncdc.noaa.gov/cdo-web/api/v2/stations?'
location_id = f'FIPS:06'
stat_id = f'FIPS:06'
Token = f'zDTBSGxqsIUFOQzYcYTcdGEZfTbNuwYk'

idx = pd.IndexSlice

def get_mo_data_noaa(dt, start_year, stop_year):
    df= pd.DataFrame()
    for year in range(start_year, stop_year+1):

        year = str(year)
        print('working on year '+year)

        datatype = dt

        #we now run a small, length =1, call to figure out our offsets:
        r_t = requests.get(f'{data_endpoint}&datasetid=GSOM&datatypeid={datatype}&locationid={location_id}&startdate={year}-01-01&enddate={year}-12-31&limit=1', headers={'token':Token})
        d_t = json.loads(r_t.text)


        for offset in np.arange(1,d_t['metadata']['resultset']['count'],1000):
            offset = str(offset)
            print(offset)

            #make the api call
            r = requests.get(f'{data_endpoint}&datasetid=GSOM&datatypeid={datatype}&limit=1000&locationid={location_id}&startdate={year}-01-01&enddate={year}-12-31&offset={offset}', headers={'token':Token})
            #load the api response as a json
            d = json.loads(r.text)
            #get all items in the response which are average temperature readings
            avg_value = [item for item in d['results'] if item['datatype']== datatype]

            #get the stations from all data
            #station_temp += [item['station'] for item in avg_temps]
            #station_prcp += [item['station'] for item in avg_prcp if item['value']]

            #populate date and average temperature fields 
            #(cast string date to datetime and convert temperature from tenths of Celsius to Fahrenheit)
            df_year = pd.DataFrame(avg_value)

            #Concatinate all years into single dataframe 
            df= pd.concat([df, df_year])

    df['date'] = df['date'].apply(lambda x: datetime.strptime(str(x),"%Y-%m-%dT%H:%M:%S"))

    idx = pd.MultiIndex.from_frame(df[['date','station']])

    frame = pd.DataFrame(df['value'].values, index = idx)

    frame = frame.rename(columns={0: datatype})

    return frame

def get_stat_noaa(start_year, stop_year):
    df= pd.DataFrame()


    st_year = str(start_year)
    end_year = str(stop_year)

    #we now run a trial call to figure out our offsets:
    r_t = requests.get(f'{stat_endpoint}&locationid={stat_id}&limit=1&startdate={st_year}-01-01&enddate={end_year}-12-31', headers={'token':Token})
    d_t = json.loads(r_t.text)


    for offset in np.arange(1,d_t['metadata']['resultset']['count'],1000):
        offset = str(offset)
        print(offset)

        #make the api call
        r = requests.get(f'{stat_endpoint}&locationid={location_id}&limit=1000&startdate={st_year}-01-01&enddate={end_year}-12-31&offset={offset}', headers={'token':Token})
        #load the api response as a json
        d = json.loads(r.text)
        #get all items in the response which are average temperature readings
        avg_value = [item for item in d['results']]

        #populate date and average temperature fields 
        #(cast string date to datetime and convert temperature from tenths of Celsius to Fahrenheit)
        df_year = pd.DataFrame(avg_value)

        #Concatinate all years into single dataframe 
        df= pd.concat([df, df_year])
        
    df = df.rename(columns={'id': 'station'})
    
    #Make a column called 'coords' with a tuple containing latitude and longitude
    
    stat_coords = list(zip(df.latitude.values, df.longitude.values))
    df['coords'] = stat_coords
    
    return df

def get_wf_data(file, small_cut):
    #Import file from local machine
    df = pd.read_csv(file)
    #Relevant data
    df = df[['AcresBurned', 'ArchiveYear', 'Counties', 'Extinguished', 'Latitude', 'Longitude','Name', 'Started']]
    #Filter fires smaller than cutoff
    small_fires = df.loc[df['AcresBurned']<small_cut]
    df = df.drop(small_fires.index)
    #Filter fires outside of CA
    non_ca =df.loc[(df['Latitude']<32.5121) | (df['Latitude']>42.0126) | (df['Longitude']>-114.1315) | 
                   (df['Longitude']<-124.6509) ].index
    df= df.drop(non_ca)
    
    #Make a column called 'coords' with a tuple containing latitude and longitude

    wf_coords = list(zip(df.Latitude.values, df.Longitude.values))
    df['coords'] = wf_coords

    
    return df

def get_close(wf_frame, stat_frame, cutoff):
    stat_list = np.zeros(stat_frame.coords.shape)
    stat_list = pd.DataFrame(data=stat_list, index= stat_frame.index)
    for wf_index, wf_coord in wf_frame.coords.items():
        for stat_index, stat_coord in stat_frame.coords.items():
            dist = distance(wf_coord, stat_coord).km
            if dist < cutoff:
                stat_list.loc[idx[stat_index]] = int(1)
    return stat_list