import requests
import pandas as pd
from datetime import date
from geopy.geocoders import Nominatim # For Geocoding

# -------------------- 1. GEOCORING: Convert Place Name to Coordinates (Lat/Lon) --------------------
def get_lat_lon(place_name, user_agent_suffix="unified_environmental_analysis_tool"):
    """
    Uses Nominatim to search for coordinates (Lat/Lon) from a place name.
    """
    geolocator = Nominatim(user_agent=user_agent_suffix)
    try:
        location = geolocator.geocode(place_name, addressdetails=True, extratags=True, timeout=10) 
        if location:
            # Returns: Latitude, Longitude, Full Location Name
            return location.latitude, location.longitude, location.address
        else:
            print(f"ERROR: Coordinates not found for: {place_name} (Please try a more specific name)")
            return None, None, None
    except Exception as e:
        print(f"ERROR: An error occurred during geocoding: {e}")
        return None, None, None

# -------------------- 2. NASA POWER: Fetch and Process Weather Data --------------------
def fetch_nasa_power_data(start_date_str, end_date_str, lat, lon, parameters):
    """
    Fetches daily weather data from NASA POWER API and converts it to a DataFrame.
    """
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "start": start_date_str,
        "end": end_date_str,
        "latitude": lat,
        "longitude": lon,
        "parameters": ",".join(parameters),
        "community": "ag",
        "format": "JSON"
    }
    
    print(f"\n--- Requesting NASA POWER Data ({', '.join(parameters)}) ---")
    
    try:
        response = requests.get(base_url, params=params, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        
        if "properties" in data and "parameter" in data["properties"]:
            parameters_data = data["properties"]["parameter"]
            
            # Create DataFrame
            df = pd.DataFrame(parameters_data)
            df.index = pd.to_datetime(df.index, format='%Y%m%d')
            df.index.name = "Date"
            
            # Fetch units metadata
            metadata = data.get('parameters', {}) 
            unit_map = {
                param: metadata.get(param, {}).get('units', 'N/A') 
                for param in df.columns
            }
            
            print(f"SUCCESS: NASA POWER data fetched! ({len(df)} days of data)")
            return df, unit_map
        else:
            print("ERROR: Parameter data not found in the NASA API response.")
            return None, None
    except requests.RequestException as e:
        print(f"ERROR: An error occurred while fetching NASA POWER data: {e}")
        return None, None

# -------------------- 3. OPEN-METEO: Fetch and Process PM2.5 Data --------------------
def fetch_and_process_pm25_data(latitude, longitude, start_date_dt, end_date_dt):
    """
    Fetches hourly PM2.5 data from Open-Meteo Air Quality API and calculates the daily average.
    """
    start_str = start_date_dt.strftime("%Y-%m-%d")
    end_str = end_date_dt.strftime("%Y-%m-%d")
    
    parameters = "pm2_5"
    base_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    
    api_url = f"{base_url}?latitude={latitude}&longitude={longitude}&hourly={parameters}&start_date={start_str}&end_date={end_str}&timezone=auto"
    
    print(f"\n--- Requesting Hourly PM2.5 Data from Open-Meteo ---")
    
    try:
        response = requests.get(api_url, verify=True, timeout=30.0)
        response.raise_for_status() 
        data = response.json()
        
        if 'hourly' in data and 'time' in data['hourly'] and 'pm2_5' in data['hourly']:
            raw_data = data['hourly']
            
            df_hourly = pd.DataFrame({
                'Date_Time': raw_data['time'],
                'PM25_Hourly': raw_data['pm2_5'] # Renamed for clarity
            })
            
            df_hourly = df_hourly.set_index(pd.to_datetime(df_hourly['Date_Time'])).drop(columns=['Date_Time'])
            # Calculate daily mean
            df_daily = df_hourly.resample('D').mean().dropna()
            df_daily.columns = ['PM25_Daily_Mean'] # Column name change
            df_daily.index.name = 'Date'
            
            unit_map = {'PM25_Daily_Mean': 'μg/m³'}
            
            if df_daily.empty:
                print("\nERROR: PM2.5 data was retrieved but contained no usable numeric values.")
                return None, None
            
            print(f"SUCCESS: PM2.5 data fetched and processed! ({len(df_daily)} days of data)")
            return df_daily, unit_map
        else:
            print("ERROR: PM2.5 data not found in the Open-Meteo API response.")
            return None, None
    except requests.exceptions.RequestException as e:
        print(f"ERROR: An error occurred while connecting to Open-Meteo API: {e}")
        return None, None

# -------------------- 4. OPEN-METEO: Fetch and Process Snow Data (Snowfall/Depth) --------------------
def get_snow_data(latitude, longitude, start_date_dt, end_date_dt):
    """
    Fetches daily total snowfall (Snowfall Sum) and maximum snow depth (Snow Depth Max)
    from Open-Meteo Historical Weather API.
    """
    start_str = start_date_dt.strftime("%Y-%m-%d")
    end_str = end_date_dt.strftime("%Y-%m-%d")
    
    daily_parameters = "snowfall_sum,snow_depth_max"
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    api_url = (
        f"{base_url}?"
        f"latitude={latitude}&longitude={longitude}&"
        f"daily={daily_parameters}&"
        f"start_date={start_str}&end_date={end_str}&"
        f"timezone=auto"
    )
    
    print(f"\n--- Requesting Daily Snow Data (Historical) ---")
    
    try:
        response = requests.get(api_url, verify=True, timeout=60.0)
        response.raise_for_status()
            
        data = response.json()
        
        if 'daily' not in data or 'time' not in data['daily'] or len(data['daily']['time']) == 0:
            print("ERROR: Daily data not found in the API response.")
            return None
        
        # Process daily data
        raw_daily = data['daily']
        df_daily = pd.DataFrame({
            'Date': raw_daily['time'],
            'Snowfall_Sum_cm': raw_daily.get('snowfall_sum'), 
            'Snow_Depth_Max_m': raw_daily.get('snow_depth_max') 
        })
        
        df_daily = df_daily.set_index(pd.to_datetime(df_daily['Date'])).drop(columns=['Date'])
        df_daily.index.name = 'Date'
        
        if df_daily.empty:
            print("\nERROR: API returned data structure is empty.")
            return None
            
        # Convert snow depth from meters (m) to centimeters (cm) and create new column
        df_daily['Snow_Depth_Max_cm'] = df_daily['Snow_Depth_Max_m'] * 100
        df_daily = df_daily.drop(columns=['Snow_Depth_Max_m'])
        
        # Replace all NaN (missing values) with 0.0
        df_daily = df_daily.fillna(0.0)
        
        print(f"SUCCESS: Daily snow data fetched! (Depth in cm)")
        return df_daily
        
    except requests.exceptions.RequestException as e:
        print(f"ERROR: An error occurred while connecting to API: {e}")
        return None

# -------------------- 5. OPEN-METEO: Fetch and Process Relative Humidity Data --------------------
def get_humidity_data(latitude, longitude, start_date_dt, end_date_dt):
    """
    Fetches daily Relative Humidity (Mean, Max, Min)
    from Open-Meteo Historical Weather API.
    """
    start_str = start_date_dt.strftime("%Y-%m-%d")
    end_str = end_date_dt.strftime("%Y-%m-%d")
    
    daily_parameters = "relative_humidity_2m_mean,relative_humidity_2m_max,relative_humidity_2m_min"
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    api_url = f"{base_url}?latitude={latitude}&longitude={longitude}&daily={daily_parameters}&start_date={start_str}&end_date={end_str}&timezone=auto"
    
    print(f"\n--- Requesting Relative Humidity Data from Open-Meteo Historical Weather API ---")
    
    try:
        response = requests.get(api_url, verify=True, timeout=30.0)
        response.raise_for_status() 
        data = response.json()
        
        if 'daily' not in data or 'time' not in data['daily'] or len(data['daily']['time']) == 0:
            print("ERROR: Daily data not found in the API response.")
            return None, None
        
        raw_daily = data['daily']
        df_daily = pd.DataFrame({
            'Date': raw_daily['time'],
            'Relative_Humidity_Mean_pct': raw_daily.get('relative_humidity_2m_mean'), # Column name change
            'Relative_Humidity_Max_pct': raw_daily.get('relative_humidity_2m_max'),   # Column name change
            'Relative_Humidity_Min_pct': raw_daily.get('relative_humidity_2m_min')   # Column name change
        })
        
        df_daily = df_daily.set_index(pd.to_datetime(df_daily['Date'])).drop(columns=['Date'])
        df_daily.index.name = 'Date'
        
        unit_map = {col: '%' for col in df_daily.columns}
        df_cleaned = df_daily.dropna(how='all')
        
        if df_cleaned.empty:
            print("\nERROR: Data retrieved but contained no usable numeric values (all NaN).")
            return None, None
        
        print(f"SUCCESS: Daily relative humidity data fetched! (Mean, Max, Min)")
        return df_daily, unit_map
        
    except requests.exceptions.RequestException as e:
        print(f"ERROR: An error occurred while connecting to API: {e}")
        return None, None

# -------------------- 6. OPEN-METEO: Fetch and Process Marine Data (Daily Summary) --------------------
def get_marine_data_daily_summary(latitude, longitude, start_date_dt, end_date_dt):
    """
    Fetches hourly marine data from Open-Meteo Marine API and calculates 'Daily Summary' using Pandas Resampling.
    
    Returns: Daily summary DataFrame or None
    """
    start_str = start_date_dt.strftime("%Y-%m-%d")
    end_str = end_date_dt.strftime("%Y-%m-%d")
    
    hourly_parameters = "wave_height,ocean_current_velocity,swell_wave_period,wind_wave_height,sea_level_height_msl"
    base_url = "https://marine-api.open-meteo.com/v1/marine"
    
    api_url = (
        f"{base_url}?"
        f"latitude={latitude}&longitude={longitude}&"
        f"hourly={hourly_parameters}&"
        f"start_date={start_str}&end_date={end_str}&"
        f"timezone=auto"
    )
    
    print(f"\n--- Requesting Hourly Marine Data for Daily Summary ---")
    
    try:
        response = requests.get(api_url, verify=True, timeout=120.0) 
        response.raise_for_status()
            
        data = response.json()
        
        if 'hourly' not in data or 'time' not in data['hourly'] or len(data['hourly']['time']) == 0:
            print("ERROR: Hourly data not found in the API response.")
            return None
        
        # 1. Process hourly data into a DataFrame
        raw_hourly = data['hourly']
        df_hourly = pd.DataFrame(raw_hourly)
        df_hourly = df_hourly.set_index(pd.to_datetime(df_hourly['time'])).drop(columns=['time'])
        df_hourly.index.name = 'Time'
        
        column_mapping = {
            'wave_height': 'Wave_Height_m',
            'ocean_current_velocity': 'Ocean_Current_Velocity_ms',
            'swell_wave_period': 'Swell_Wave_Period_s',
            'wind_wave_height': 'Wind_Wave_Height_m',
            'sea_level_height_msl': 'Sea_Level_Height_m'
        }
        df_hourly = df_hourly.rename(columns=column_mapping)
        
        df_cleaned = df_hourly.dropna(how='all')
        
        if df_cleaned.empty:
            print("\nWARNING: Marine data retrieved but contains no usable numeric values (all NaN) - This may indicate an inland location.")
            return None
        
        # 2. Daily Summary (Resampling)
        daily_max_cols = [col for col in df_cleaned.columns if 'Height' in col or 'Velocity' in col]
        df_daily_max = df_cleaned[daily_max_cols].resample('D').max().rename(
            columns={col: f'Daily_Max_{col}' for col in daily_max_cols}
        )
        
        daily_mean_cols = [col for col in df_cleaned.columns if 'Period' in col]
        df_daily_mean = df_cleaned[daily_mean_cols].resample('D').mean().rename(
            columns={col: f'Daily_Mean_{col}' for col in daily_mean_cols}
        )
        
        df_daily_summary = pd.merge(
            df_daily_max, 
            df_daily_mean, 
            left_index=True, 
            right_index=True,
            how='outer'
        )
        df_daily_summary.index.name = 'Date'

        print(f"SUCCESS: Marine data fetched and summarized daily! (Total {len(df_daily_summary)} days)")
        return df_daily_summary
        
    except requests.exceptions.RequestException as e:
        print(f"ERROR: An error occurred while connecting to Marine API: {e}")
        return None

# -------------------- 7. Main Execution Section --------------------
if __name__ == "__main__":
    
    # Define start dates for APIs
    NASA_START_DATE = date(1981, 1, 1) 
    PM25_START_DATE = date(2013, 1, 1)
    ARCHIVE_START_DATE = date(2020, 1, 1) 
    MARINE_START_DATE = date(2020, 1, 1)
    
    END_DATE = date.today() 
    
    # User Input
    place_name = input("Enter Place Name (e.g., 'London, UK' or 'Phuket, Thailand'): ")
    
    # 1. Geocoding
    lat, lon, full_location_name = get_lat_lon(place_name) 
    
    if lat is not None and lon is not None:
        print(f"\nSUCCESS: Coordinates found for {full_location_name}: Latitude={lat:.4f}, Longitude={lon:.4f}")
        
        # --- Fetch Data Set 1: NASA POWER (Basic Weather) ---
        nasa_params = ["T2M", "PRECTOTCORR", "WS2M"] 
        nasa_df, nasa_units = fetch_nasa_power_data(
            NASA_START_DATE.strftime("%Y%m%d"), 
            END_DATE.strftime("%Y%m%d"), 
            lat, lon, nasa_params
        )
        
        # --- Fetch Data Set 2: Open-Meteo (PM2.5 Air Quality) ---
        pm25_df, pm25_units = fetch_and_process_pm25_data(
            lat, lon, PM25_START_DATE, END_DATE
        )

        # --- Fetch Data Set 3: Open-Meteo (Relative Humidity) ---
        humidity_df, humidity_units = get_humidity_data(
            lat, lon, ARCHIVE_START_DATE, END_DATE
        )

        # --- Fetch Data Set 4: Open-Meteo (Snow Data) ---
        snow_df = get_snow_data(
            lat, lon, ARCHIVE_START_DATE, END_DATE
        )

        # --- Fetch Data Set 5: Open-Meteo (Marine Weather - Daily Summary) ---
        marine_df = get_marine_data_daily_summary(
            lat, lon, MARINE_START_DATE, END_DATE 
        )

        # -------------------- 8. Summarize and Display All Averages --------------------
        print("\n" + "="*70)
        print(f"DATA RETRIEVAL SUMMARY FOR: {full_location_name}")
        print("="*70)

        # NASA POWER Summary
        if nasa_df is not None and not nasa_df.empty:
            print("\n--- NASA POWER (Basic Weather) ---")
            print(f"Date Range: {nasa_df.index.min().date()} to {nasa_df.index.max().date()}")
            print("[Average Value Over Period]")
            summary = nasa_df.mean(numeric_only=True).to_frame(name='Average Value')
            summary['Unit'] = summary.index.map(nasa_units)
            print(summary.to_string(float_format="%.4f"))
            print("\n[DataFrame Preview (First 5 Rows)]")
            print(nasa_df.head())
        
        # PM2.5 Summary
        if pm25_df is not None and not pm25_df.empty:
            print("\n--- Open-Meteo (PM2.5 Air Quality) ---")
            print(f"Date Range: {pm25_df.index.min().date()} to {pm25_df.index.max().date()}")
            print("[Average Value Over Period]")
            avg_pm25 = pm25_df['PM25_Daily_Mean'].mean()
            print(f"PM25_Daily_Mean: {avg_pm25:.2f} {pm25_units.get('PM25_Daily_Mean')}")
            print("\n[DataFrame Preview (First 5 Rows)]")
            print(pm25_df.head())
        
        # Humidity Summary
        if humidity_df is not None and not humidity_df.empty:
            print("\n--- Open-Meteo (Daily Relative Humidity) ---")
            df_clean = humidity_df.dropna(how='all')
            print(f"Date Range: {df_clean.index.min().date()} to {df_clean.index.max().date()}")
            print("[Average Value Over Period]")
            summary = df_clean.mean(numeric_only=True).to_frame(name='Average Value')
            summary['Unit'] = summary.index.map(humidity_units)
            print(summary.to_string(float_format="%.2f"))
            print("\n[DataFrame Preview (First 5 Rows)]")
            print(df_clean.head())

        # Snow Summary
        if snow_df is not None and not snow_df.empty:
            print("\n--- Open-Meteo (Snow Data) ---")
            print(f"Date Range: {snow_df.index.min().date()} to {snow_df.index.max().date()}")
            total_snowfall = snow_df['Snowfall_Sum_cm'].sum()
            max_snow_depth = snow_df['Snow_Depth_Max_cm'].max()
            df_snowy_days = snow_df[snow_df['Snowfall_Sum_cm'] > 0.0]
            print(f"Total Cumulative Snowfall: {total_snowfall:.2f} cm")
            print(f"Maximum Snow Depth: {max_snow_depth:.2f} cm")
            print(f"Total Days with Snowfall: {len(df_snowy_days)} days")
            print("\n[DataFrame Preview (First 5 Rows)]")
            print(snow_df.head())

        # Marine Summary
        if marine_df is not None and not marine_df.empty:
            print("\n--- Open-Meteo (Daily Marine Weather Summary) ---")
            df_clean = marine_df.dropna(how='all')
            print(f"Date Range: {df_clean.index.min().date()} to {df_clean.index.max().date()}")
            print("[Average Daily Max/Mean Over Period]")
            summary = df_clean.mean(numeric_only=True).to_frame(name='Average Value').T
            print(summary.to_string(float_format="%.4f"))
            print("\n[DataFrame Preview (First 5 and Last 5 Rows)]")
            print(pd.concat([df_clean.head(), df_clean.tail()])) 
        else:
            print("\n--- Open-Meteo (Daily Marine Weather Summary) ---")
            print("WARNING: No marine data found! This may be because the coordinates are inland, too close to the coast, or the Open-Meteo Marine API does not support this region.")
            
        if all(df is None or df.empty for df in [nasa_df, pm25_df, humidity_df, snow_df, marine_df]):
            print("\nWARNING: No data was successfully retrieved. Please check the place name or date ranges.")
    else:
        print("ERROR: Cannot proceed because coordinates were not found.")