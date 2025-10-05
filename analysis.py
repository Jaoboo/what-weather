import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import requests
from geopy.geocoders import Nominatim
import warnings
import logging

warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

np.random.seed(42)


# ===== GEOLOCATION FUNCTIONS =====
def get_lat_lon_from_name(place_name):
    """Convert place name to coordinates"""
    import time
    
    try:
        from geopy.geocoders import Photon
        geolocator = Photon(user_agent="weather_app", timeout=10)
        time.sleep(1)
        location = geolocator.geocode(place_name, language='en')
        if location:
            return location.latitude, location.longitude, location.address
    except Exception as e:
        print(f"Photon geocoding failed: {e}")
    
    try:
        geolocator = Nominatim(
            user_agent="weather_prediction_app_v1",
            timeout=10
        )
        time.sleep(2)
        location = geolocator.geocode(place_name, timeout=10, language='en')
        if location:
            return location.latitude, location.longitude, location.address
    except Exception as e:
        print(f"Nominatim geocoding failed: {e}")
    
    print(f"Error geocoding: Could not find coordinates for {place_name}")
    return None, None, None


# ===== NASA POWER DATA FUNCTIONS =====

def get_nasa_data(location_name):
    """Fetch NASA POWER data (T2M, PRECTOTCORR only)"""
    lat, lon, _ = get_lat_lon_from_name(location_name)
    if not lat or not lon:
        return None
    
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "start": "20150101",
        "end": "20250922",
        "latitude": lat,
        "longitude": lon,
        "parameters": "T2M,PRECTOTCORR",
        "community": "ag",
        "format": "JSON"
    }
    
    try:
        print(f"   Fetching NASA POWER data...")
        response = requests.get(base_url, params=params, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        print(f"   NASA data fetched successfully")
        return data
    except Exception as e:
        print(f"   Error fetching NASA data: {e}")
        return None


def nasa_json_to_dataframe(data):
    """Convert NASA JSON to DataFrame"""
    if "properties" in data and "parameter" in data["properties"]:
        parameters = data["properties"]["parameter"]
        df = pd.DataFrame()
        for param_name, values in parameters.items():
            series = pd.Series(values, name=param_name)
            df = pd.concat([df, series], axis=1)
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        df.index.name = "Date"
        return df
    else:
        return pd.DataFrame()


def fetch_wind_data(start_date_str, end_date_str, lat, lon):
    """Fetch wind data (WS2M) from NASA POWER"""
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "start": start_date_str,
        "end": end_date_str,
        "latitude": lat,
        "longitude": lon,
        "parameters": "WS2M",
        "community": "ag",
        "format": "JSON"
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        
        if "properties" in data and "parameter" in data["properties"]:
            wind_data = data["properties"]["parameter"]["WS2M"]
            df = pd.DataFrame.from_dict(wind_data, orient='index', columns=['WS2M'])
            df.index = pd.to_datetime(df.index, format='%Y%m%d')
            df.index.name = "Date"
            print(f"   Wind data: {len(df)} days")
            return df
        else:
            print(f"   No wind data in response")
            return None
    except Exception as e:
        print(f"   Error fetching wind data: {e}")
        return None


# ===== OPEN-METEO DATA FUNCTIONS =====

def fetch_open_meteo_historical(latitude, longitude, start_date, end_date):
    """Fetch historical weather data from Open-Meteo"""
    today = date.today()
    if isinstance(end_date, datetime):
        end_date = end_date.date()
    if isinstance(start_date, datetime):
        start_date = start_date.date()
    
    max_available_date = today - pd.Timedelta(days=1)
    
    if end_date > max_available_date:
        print(f"   Adjusting end_date from {end_date} to {max_available_date} (Archive API limit)")
        end_date = max_available_date
    
    if start_date > end_date:
        print(f"   Warning: start_date {start_date} is after adjusted end_date {end_date}")
        return None
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_str,
        "end_date": end_str,
        "daily": "temperature_2m_mean,precipitation_sum,wind_speed_10m_mean",
        "timezone": "auto"
    }
    
    try:
        print(f"   Fetching Open-Meteo historical data ({start_str} to {end_str})...")
        response = requests.get(base_url, params=params, timeout=30.0)
        
        if response.status_code != 200:
            print(f"   Open-Meteo API returned status code: {response.status_code}")
            
            if response.status_code == 400 and (end_date - start_date).days > 365:
                print(f"   Retrying with last year of data only...")
                new_start = end_date - pd.Timedelta(days=365)
                return fetch_open_meteo_historical(latitude, longitude, new_start, end_date)
            
            return None
            
        response.raise_for_status()
        data = response.json()
        
        if 'daily' in data:
            df = pd.DataFrame({
                'Date': pd.to_datetime(data['daily']['time']),
                'T2M_OM': data['daily']['temperature_2m_mean'],
                'PRECTOTCORR_OM': data['daily']['precipitation_sum'],
                'WS2M_OM': data['daily']['wind_speed_10m_mean'],
            })
            df.set_index('Date', inplace=True)
            print(f"   Open-Meteo data: {len(df)} days")
            return df
        else:
            print(f"   No 'daily' data in Open-Meteo response")
            return None
    except Exception as e:
        print(f"   Error fetching Open-Meteo data: {e}")
        return None

def fetch_humidity_snow_historical(latitude, longitude, start_date, end_date):
    """Fetch from Historical Weather API (3 years of data)"""
    
    # Select data for the previous 3 years
    today = date.today()
    actual_start = today - timedelta(days=1095)  # 3 ปี = 365 * 3
    actual_end = today - timedelta(days=1)
    
    start_str = actual_start.strftime("%Y-%m-%d")
    end_str = actual_end.strftime("%Y-%m-%d")
    
    # Use Historical Weather API
    base_url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_str,
        "end_date": end_str,
        "daily": "relative_humidity_2m_mean,snowfall_sum,snow_depth_max",
        "timezone": "auto"
    }
    
    try:
        print(f"   Requesting from Historical Weather API")
        print(f"   Date range: {start_str} to {end_str} (3 years)")
        print(f"   Coordinates: {latitude:.4f}, {longitude:.4f}")
        
        response = requests.get(base_url, params=params, timeout=60.0)
        
        print(f"   Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"   Error response: {response.text[:300]}")
            return None
        
        data = response.json()
        
        if 'daily' not in data or 'time' not in data['daily']:
            print(f"   Invalid response structure")
            return None
        
        daily = data['daily']
        print(f"   Response keys: {list(daily.keys())}")
        
        df = pd.DataFrame({
            'Date': pd.to_datetime(daily['time']),
            'RH2M': daily.get('relative_humidity_2m_mean'),
            'Snowfall_Sum': daily.get('snowfall_sum'),
            'Snow_Depth_Max': daily.get('snow_depth_max')
        })
        df.set_index('Date', inplace=True)
        
        # Convert snow depth from m to cm
        if 'Snow_Depth_Max' in df.columns and df['Snow_Depth_Max'].notna().any():
            df['Snow_Depth_Max'] = df['Snow_Depth_Max'] * 100

        # Fill NaN
        if 'Snowfall_Sum' in df.columns:
            df['Snowfall_Sum'] = df['Snowfall_Sum'].fillna(0.0)
        if 'Snow_Depth_Max' in df.columns:
            df['Snow_Depth_Max'] = df['Snow_Depth_Max'].fillna(0.0)
        if 'RH2M' in df.columns and df['RH2M'].notna().any():
            df['RH2M'] = df['RH2M'].fillna(df['RH2M'].mean())
        
        print(f"   SUCCESS: Fetched {len(df)} days (3 years)")
        
        # Show statistics
        if 'Snowfall_Sum' in df.columns:
            snow_days = (df['Snowfall_Sum'] > 0).sum()
            max_snow = df['Snowfall_Sum'].max()
            mean_snow = df[df['Snowfall_Sum'] > 0]['Snowfall_Sum'].mean() if snow_days > 0 else 0
            print(f"   Snowfall: {snow_days} days with snow")
            print(f"            max={max_snow:.1f}mm, mean={mean_snow:.1f}mm")
        
        if 'Snow_Depth_Max' in df.columns:
            depth_days = (df['Snow_Depth_Max'] > 0).sum()
            max_depth = df['Snow_Depth_Max'].max()
            print(f"   Snow depth: {depth_days} days with snow cover")
            print(f"               max={max_depth:.1f}cm")
        
        if 'RH2M' in df.columns:
            avg_hum = df['RH2M'].mean()
            min_hum = df['RH2M'].min()
            max_hum = df['RH2M'].max()
            print(f"   Humidity: avg={avg_hum:.1f}%, range={min_hum:.1f}-{max_hum:.1f}%")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"   Request error: {e}")
        return None
    except Exception as e:
        print(f"   Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def fetch_pm25_data(latitude, longitude, start_date_dt, end_date_dt):
    """Fetch PM2.5 data from Open-Meteo Air Quality API"""
    start_str = start_date_dt.strftime("%Y-%m-%d")
    end_str = end_date_dt.strftime("%Y-%m-%d")
    
    base_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "pm2_5",
        "start_date": start_str,
        "end_date": end_str,
        "timezone": "auto"
    }
    
    try:
        print(f"   Fetching PM2.5 data...")
        response = requests.get(base_url, params=params, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        
        if 'hourly' in data and 'time' in data['hourly'] and 'pm2_5' in data['hourly']:
            raw_data = data['hourly']
            
            df_hourly = pd.DataFrame({
                'Date_Time': raw_data['time'],
                'PM25': raw_data['pm2_5']
            })
            
            df_hourly = df_hourly.set_index(pd.to_datetime(df_hourly['Date_Time'])).drop(columns=['Date_Time'])
            
            # Convert to daily data (average)
            df_daily = df_hourly.resample('D').mean().dropna()
            df_daily.columns = ['PM25']
            df_daily.index.name = 'Date'
            
            print(f"   PM2.5 data: {len(df_daily)} days")
            return df_daily
        else:
            print(f"   No PM2.5 data available")
            return None
    except Exception as e:
        print(f"   Error fetching PM2.5 data: {e}")
        return None

# ===== OPTIONAL PARAMETERS FUNCTIONS =====

def fetch_optional_params(lat, lon, selected_params):
    """Fetch optional ocean parameters (synthetic data)"""
    results = {}
    
    # Ocean-related parameters (synthetic data)
    for param in selected_params:
        if param in ['WAVE_HEIGHT', 'OCEAN_CURRENT', 'SWELL_PERIOD']:
            print(f"Note: {param} data not available, generating synthetic data")
            dates = pd.date_range(start='2015-01-01', end='2025-09-22', freq='D')
            if param == 'WAVE_HEIGHT':
                values = np.random.uniform(0.5, 3.0, len(dates))
            elif param == 'OCEAN_CURRENT':
                values = np.random.uniform(0.1, 1.5, len(dates))
            elif param == 'SWELL_PERIOD':
                values = np.random.uniform(4, 12, len(dates))
            df = pd.DataFrame(values, index=dates, columns=[param])
            df.index.name = "Date"
            results[param] = df
    
    return results


# ===== PREDICTION FUNCTIONS =====

def predict_with_prophet(df, target_date, parameter_name):
    """Predict using Prophet"""
    try:
        from prophet import Prophet
    except ImportError:
        print("Prophet not installed. Install with: pip install prophet")
        return None
    
    prophet_df = pd.DataFrame({
        'ds': df.index,
        'y': df[parameter_name]
    }).dropna()
    
    if len(prophet_df) == 0:
        return None
    
    if parameter_name == 'PRECTOTCORR':
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive',
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
    elif parameter_name == 'PM25':
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.1,
            seasonality_prior_scale=5.0
        )
    else:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
    
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    model.fit(prophet_df)
    
    target_datetime = pd.to_datetime(target_date)
    
    target_month = target_datetime.month
    target_day = target_datetime.day
    current_year = datetime.now().year
    
    similar_data_recent = []
    similar_data_all = []
    
    for idx, row in prophet_df.iterrows():
        date = row['ds']
        day_diff = abs((date.month - target_month) * 30 + (date.day - target_day))
        
        if day_diff <= 5 or day_diff >= 360:
            similar_data_all.append(row['y'])
            if date.year >= current_year - 3:
                similar_data_recent.append(row['y'])
    
    future = pd.DataFrame({'ds': [target_datetime]})
    forecast = model.predict(future)
    prophet_prediction = forecast['yhat'].values[0]
    
    if len(similar_data_recent) > 2:
        recent_mean = np.mean(similar_data_recent)
        recent_percentile_75 = np.percentile(similar_data_recent, 75)
        recent_std = np.std(similar_data_recent)
        
        print(f"  {parameter_name} Recent data (last 3 years):")
        print(f"    Mean={recent_mean:.2f}, P75={recent_percentile_75:.2f}")
        
        if parameter_name == 'T2M':
            base_value = recent_percentile_75
        else:
            base_value = recent_mean
        
        final_prediction = 0.95 * base_value + 0.05 * prophet_prediction
        
        lower_bound = final_prediction - 1.5 * recent_std
        upper_bound = final_prediction + 1.5 * recent_std
        
        print(f"    Prophet={prophet_prediction:.2f}, Final={final_prediction:.2f}")

    elif len(similar_data_all) > 3:
        historical_mean = np.mean(similar_data_all)
        historical_std = np.std(similar_data_all)
        
        final_prediction = 0.9 * historical_mean + 0.1 * prophet_prediction
        
        lower_bound = final_prediction - 1.5 * historical_std
        upper_bound = final_prediction + 1.5 * historical_std
        
        print(f"    Prophet={prophet_prediction:.2f}, Final={final_prediction:.2f}")
    else:
        final_prediction = prophet_prediction
        lower_bound = forecast['yhat_lower'].values[0]
        upper_bound = forecast['yhat_upper'].values[0]
        
        print(f"  {parameter_name}: Using Prophet only={prophet_prediction:.2f}")
    
    # Temp can < 0
    if parameter_name == 'T2M':
        pass
    else:
        final_prediction = max(0, final_prediction)
        lower_bound = max(0, lower_bound)
        upper_bound = max(0, upper_bound)
    
    train_forecast = model.predict(prophet_df[['ds']])
    actual = prophet_df['y'].values
    predicted = train_forecast['yhat'].values
    
    if parameter_name == 'PRECTOTCORR':
        mask = np.abs(actual) > 1.0
    elif parameter_name == 'PM25':
        mask = np.abs(actual) > 5.0
    else:
        mask = np.abs(actual) > 0.1
    
    if mask.sum() > 0:
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        mape = None
    
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    
    return {
        'prediction': final_prediction,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'mape': mape,
        'mae': mae,
        'rmse': rmse
    }


def calculate_probability_in_range(df, parameter, target_date, threshold, condition='above', days_range=3):
    """Calculate probability within ±days_range days of target date"""
    target_dt = pd.to_datetime(target_date)
    
    date_list = []
    for i in range(-days_range, days_range + 1):
        temp_date = target_dt + pd.Timedelta(days=i)
        date_list.append((temp_date.month, temp_date.day))
    
    filtered_data = df[
        df.index.map(lambda x: (x.month, x.day) in date_list)
    ][parameter].dropna()
    
    if len(filtered_data) == 0:
        return None, 0, 0
    
    if condition == 'above':
        exceed_count = (filtered_data >= threshold).sum()
    else:
        exceed_count = (filtered_data <= threshold).sum()
    
    probability = (exceed_count / len(filtered_data)) * 100
    
    return probability, len(filtered_data), exceed_count


# ===== UTILITY FUNCTIONS =====

def validate_date_format(date_string):
    """Validate date format"""
    try:
        datetime.strptime(date_string, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def get_range_values(df, param, target_date, days_range=3):
    """Get actual values ±days_range around target_date"""
    target_dt = pd.to_datetime(target_date)
    
    result = {}
    
    if param not in df.columns:
        print(f"WARNING: {param} not in dataframe columns")
        return result
    
    print(f"\n--- Processing {param} for graph ---")
    
    current_year = datetime.now().year
    
    for i in range(-days_range, days_range + 1):
        current_date = target_dt + pd.Timedelta(days=i)
        date_str = current_date.strftime('%Y-%m-%d')
        
        mask = (df.index.month == current_date.month) & (df.index.day == current_date.day)
        historical_data = df.loc[mask, param]
        
        if len(historical_data) > 0:
            recent_mask = mask & (df.index.year >= current_year - 3)
            recent_data = df.loc[recent_mask, param]
            
            if len(recent_data) > 0:
                mean_val = recent_data.mean()
                if not pd.isna(mean_val):
                    result[date_str] = float(mean_val)
                else:
                    result[date_str] = None
            else:
                mean_val = historical_data.mean()
                if not pd.isna(mean_val):
                    result[date_str] = float(mean_val)
                else:
                    result[date_str] = None
        else:
            result[date_str] = None
    
    valid_count = sum(1 for v in result.values() if v is not None)
    print(f"{param} for graph: {len(result)} dates, {valid_count} valid values")
    
    return result


# ===== MAIN ANALYSIS FUNCTION =====

def run_analysis(location_name, target_date, dust_file_path=None, selected_params=None):
    """Main function to run analysis"""
    
    if selected_params is None:
        selected_params = []
    
    if not validate_date_format(target_date):
        return {'error': 'Invalid date format. Please use YYYY-MM-DD'}
    
    print(f"Fetching coordinates for {location_name}...")
    lat, lon, full_name = get_lat_lon_from_name(location_name)
    
    if not lat or not lon:
        return {'error': 'Failed to get coordinates for the location'}
    
    print(f"Location: {full_name}")
    print(f"Coordinates: {lat:.4f}, {lon:.4f}")
    print(f"Selected optional parameters: {selected_params}")
    
    # Fetch NASA data (T2M, PRECTOTCORR)
    data = get_nasa_data(location_name)
    
    if not data:
        return {'error': 'Failed to fetch NASA POWER data'}
    
    df = nasa_json_to_dataframe(data)
    
    # Fetch wind data
    print("Fetching wind data...")
    wind_df = fetch_wind_data("20150101", "20250922", lat, lon)
    
    if wind_df is not None:
        df = df.join(wind_df, how='left')
    
    if any(p in selected_params for p in ['RH2M', 'SNOWFALL', 'SNOW_DEPTH']):
        print(f"\n=== Fetching Humidity & Snow Data ===")
        hum_snow_df = fetch_humidity_snow_historical(lat, lon, None, None)
    
        if hum_snow_df is not None:
            print(f"\n=== Data Fetched Successfully ===")
            print(f"Total records: {len(hum_snow_df)}")
        
            if 'RH2M' in hum_snow_df.columns:
                df = df.join(hum_snow_df[['RH2M']], how='left')
        
            if 'Snowfall_Sum' in hum_snow_df.columns:
                df = df.join(hum_snow_df[['Snowfall_Sum']], how='left')
        
            if 'Snow_Depth_Max' in hum_snow_df.columns:
                df = df.join(hum_snow_df[['Snow_Depth_Max']], how='left')
        else:
            print("Failed to fetch data")
    
    # Fetch Ocean parameters (synthetic)
    if selected_params:
        ocean_params = [p for p in selected_params if p in ['WAVE_HEIGHT', 'OCEAN_CURRENT', 'SWELL_PERIOD']]
        if ocean_params:
            print("Fetching ocean parameters...")
            optional_data = fetch_optional_params(lat, lon, ocean_params)
            for param_name, param_df in optional_data.items():
                df = df.join(param_df, how='left')
                print(f"   Added {param_name}")
    
    # Ensemble with Open-Meteo
    USE_OPEN_METEO = True
    OPEN_METEO_WEIGHT = 0.3
    
    if USE_OPEN_METEO:
        om_start = date.today() - pd.Timedelta(days=365*5)
        om_df = fetch_open_meteo_historical(lat, lon, om_start, date.today())
        
        if om_df is not None:
            for col in om_df.columns:
                base_param = col.replace('_OM', '')
                if base_param in df.columns:
                    mask = om_df.index.isin(df.index)
                    common_dates = om_df[mask].index
                    
                    nasa_values = df.loc[common_dates, base_param]
                    om_values = om_df.loc[common_dates, col]
                    
                    ensemble_values = (1 - OPEN_METEO_WEIGHT) * nasa_values + OPEN_METEO_WEIGHT * om_values
                    df.loc[common_dates, base_param] = ensemble_values
                    
                    print(f"   Ensemble {base_param}: {len(common_dates)} days")
    
    # Fetch PM2.5
    pm25_end = date(2025, 9, 22)
    pm25_df = fetch_pm25_data(lat, lon, date(2013, 1, 1), pm25_end)
    
    if pm25_df is not None:
        df = df.join(pm25_df, how='left')
    
    print(f"Total records: {len(df)} days")
    print(f"Parameters: {', '.join(df.columns.tolist())}")
    
    # Predictions
    results = {}

    print(f"\n=== STARTING PREDICTIONS ===")
    print(f"Columns to predict: {list(df.columns)}")

    for param in df.columns:
        print(f"   Predicting {param}...")
        result = predict_with_prophet(df, target_date, param)
        if result:
            results[param] = result
            print(f"    {param} = {result['prediction']:.2f}")
        else:
            print(f"    {param} prediction failed")
    
    # Probabilities
    print("Calculating probabilities...")
    thresholds = {
        'T2M': [
            {'name': 'Very Hot', 'threshold': 35, 'condition': 'above'},
            {'name': 'Hot', 'threshold': 30, 'condition': 'above'}
        ],
        'PRECTOTCORR': [
            {'name': 'Very Heavy Rain', 'threshold': 50, 'condition': 'above'},
            {'name': 'Heavy Rain', 'threshold': 35, 'condition': 'above'},
            {'name': 'Moderate Rain', 'threshold': 15, 'condition': 'above'}
        ],
        'WS2M': [
            {'name': 'Very Windy', 'threshold': 10, 'condition': 'above'},
            {'name': 'Windy', 'threshold': 5, 'condition': 'above'}
        ],
        'PM25': [
            {'name': 'Hazardous', 'threshold': 150, 'condition': 'above'},
            {'name': 'Unhealthy', 'threshold': 100, 'condition': 'above'},
            {'name': 'Moderate', 'threshold': 50, 'condition': 'above'}
        ],
        'Snowfall_Sum': [
            {'name': 'Heavy Snowfall', 'threshold': 10, 'condition': 'above'},
            {'name': 'Moderate Snowfall', 'threshold': 5, 'condition': 'above'}
        ],
        'Snow_Depth_Max': [
            {'name': 'Deep Snow', 'threshold': 20, 'condition': 'above'},
            {'name': 'Moderate Snow Depth', 'threshold': 10, 'condition': 'above'}
        ],
        'RH2M': [
            {'name': 'High Humidity', 'threshold': 70, 'condition': 'above'},
            {'name': 'Low Humidity', 'threshold': 30, 'condition': 'below'}
        ]
    }
    
    probabilities = {}
    for param, configs in thresholds.items():
        if param in df.columns:
            probabilities[param] = []
            for config in configs:
                prob, total, exceed = calculate_probability_in_range(
                    df, param, target_date, 
                    config['threshold'], config['condition']
                )
                if prob is not None:
                    probabilities[param].append({
                        'name': config['name'],
                        'threshold': config['threshold'],
                        'probability': prob,
                        'total': total,
                        'exceed': exceed
                    })
    
    # Recommendations
    recommendations = []
    
    if 'T2M' in results:
        temp = results['T2M']['prediction']
        if temp < 0:
            recommendations.append("Extremely Cold weather expected")
        elif temp < 10:
            recommendations.append("Very Cold - Wear warm clothes")
        elif temp < 15:
            recommendations.append("Cold - Light jacket recommended")
        elif temp < 25:
            recommendations.append("Comfortable weather expected")
        elif temp < 30:
            recommendations.append("Warm and pleasant")
        elif temp < 35:
            recommendations.append("Hot - Stay hydrated")
        else:
            recommendations.append("Extremely hot - Avoid outdoor activities")
    
    if 'PRECTOTCORR' in results:
        rain = results['PRECTOTCORR']['prediction']
        if rain < 5:
            recommendations.append("No rain expected - Good for outdoor activities")
        elif rain < 15:
            recommendations.append("Light rain possible - Umbrella recommended")
        elif rain < 35:
            recommendations.append("Moderate rain - Bring umbrella and raincoat")
        else:
            recommendations.append("Heavy rain - Stay indoors if possible")
    
    if 'WS2M' in results:
        wind = results['WS2M']['prediction']
        if wind < 5:
            recommendations.append("Calm wind conditions")
        elif wind < 10:
            recommendations.append("Moderate wind - Secure loose items")
        else:
            recommendations.append("Strong wind - Exercise caution outdoors")
    
    if 'PM25' in results:
        pm = results['PM25']['prediction']
        if pm < 50:
            recommendations.append("Good air quality - Safe for all activities")
        elif pm < 100:
            recommendations.append("Moderate air quality - Sensitive groups should limit outdoor exposure")
        elif pm < 150:
            recommendations.append("Unhealthy air - Wear mask outdoors, limit strenuous activities")
        else:
            recommendations.append("Hazardous air quality - Stay indoors, use air purifier")
    
    if 'RH2M' in results:
        humidity = results['RH2M']['prediction']
        if humidity < 30:
            recommendations.append("Low humidity - Use moisturizer, stay hydrated")
        elif humidity > 70:
            recommendations.append("High humidity - May feel muggy, drink plenty of water")
    
    if 'Snowfall_Sum' in results and results['Snowfall_Sum']['prediction'] > 0:
        snow = results['Snowfall_Sum']['prediction']
        if snow < 5:
            recommendations.append("Light snowfall expected")
        else:
            recommendations.append("Heavy snowfall - Drive carefully, dress warmly")
    
    if 'WAVE_HEIGHT' in results:
        wave = results['WAVE_HEIGHT']['prediction']
        if wave < 1:
            recommendations.append("Calm seas - Good for swimming and water activities")
        elif wave < 2:
            recommendations.append("Moderate waves - Safe for experienced swimmers")
        else:
            recommendations.append("Rough seas - Avoid swimming, high surf advisory")
    
    # Generate Activity Recommendations
    activities = []
    warnings = []
    
    temp = results.get('T2M', {}).get('prediction', None)
    rain = results.get('PRECTOTCORR', {}).get('prediction', None)
    wind = results.get('WS2M', {}).get('prediction', None)
    pm25 = results.get('PM25', {}).get('prediction', None)
    humidity = results.get('RH2M', {}).get('prediction', None)
    snowfall = results.get('Snowfall_Sum', {}).get('prediction', None)
    snow_depth = results.get('Snow_Depth_Max', {}).get('prediction', None)
    wave_height = results.get('WAVE_HEIGHT', {}).get('prediction', None)
    ocean_current = results.get('OCEAN_CURRENT', {}).get('prediction', None)
    swell_period = results.get('SWELL_PERIOD', {}).get('prediction', None)
    
    # Check warnings
    if temp is not None:
        if temp > 35:
            warnings.append("WARNING: Extreme heat - Stay indoors during peak hours")
        elif temp < 5:
            warnings.append("WARNING: Extreme cold - Risk of hypothermia")
    
    if rain is not None and rain > 15:
        warnings.append("WARNING: Heavy rain - Indoor activities recommended")
    
    if wind is not None and wind > 12:
        warnings.append("WARNING: Strong winds - Dangerous for outdoor activities")
    
    if pm25 is not None and pm25 > 100:
        warnings.append("WARNING: Poor air quality - Wear mask, stay indoors")
    
    if humidity is not None and humidity > 80 and temp is not None and temp > 30:
        warnings.append("WARNING: High heat index - Risk of heat stroke")
    
    if wave_height is not None and wave_height > 2.5:
        warnings.append("WARNING: High waves - Swimming dangerous")
    
    if ocean_current is not None and ocean_current > 1.5:
        warnings.append("WARNING: Strong currents - Do not enter water")
    
    # Activity suggestions
    if warnings:
        activities.extend(warnings)
        activities.append("")
        activities.append("Safe alternatives: Indoor gym, malls, museums")
    else:
        # Winter activities
        # 1. Main Condition: Is there snow? (fresh snowfall OR existing depth > 5cm)
        if (snowfall is not None and snowfall > 0) or (snow_depth is not None and snow_depth > 5):
    
            # 2. Temperature Check and Safety Alert
            if temp is not None:
        
                if temp < -15: # Extreme Cold Alert (Example threshold: -15C or 5F)
                    activities.append("EXTREME COLD WARNING:")
                    activities.append("Temperatures are dangerously low. Limit time outdoors to prevent frostbite.")
        
                elif temp <= 5: # Ideal range for winter sports (5C or 41F)
                    activities.append("WINTER ACTIVITIES:")
            
                    # Check Snow Depth for Activity Suitability
                    if snow_depth is not None and snow_depth < 10:
                        activities.append("Light Snow: Building snowman, Winter Walks, Small Hill Sledding.")
                 
                    elif snow_depth is not None and snow_depth < 30:
                        activities.append("Moderate Snow: Downhill Skiing, Snowboarding, Cross-Country Skiing.")
                 
                else: # Snow depth >= 30cm
                    activities.append("Deep Snow: Advanced Skiing/Snowboarding, Backcountry Snowshoeing.")
                 
                    # Add an empty line for visual separation in the output
                activities.append("")
            
            else:
                # Snow is present, but it's too warm (e.g., 8C)
                activities.append("WET SNOW/SLUSHY CONDITIONS")
                activities.append("Snow is melting. Activities like skiing may be difficult or dangerous.")
                activities.append("")
        
        # Water activities
        # 1. Safety Check: Ensure temp is high enough and critical safety data is present
        if (temp is not None and temp > 30 and 
            wave_height is not None and ocean_current is not None):
            # 2. Condition 1: Calm Conditions (Safety Focused)
            if wave_height < 0.8 and ocean_current < 0.5:
                activities.append("CALM WATER ACTIVITIES (Beginner Friendly):")
                activities.append("Swimming, Snorkeling, Kayaking, Paddleboarding, Fishing")
        
            # 3. Condition 2: Moderate Conditions (Water Sports)
            elif wave_height < 1.5 and ocean_current < 1.0:
                activities.append("MODERATE WATER SPORTS (Experienced Recommended):")
                activities.append("Surfing, Bodyboarding, Sailing")
        
                # Swell period check (must check for None again)
                if swell_period is not None and swell_period > 8:
                    activities.append("Note: Long period swell - Great wave quality for surfing.")
            
            # 4. Critical Else Block: High Danger
            else:
                activities.append("SAFETY ALERT: HIGH DANGER")
                activities.append("Wave height or ocean current exceeds safe limits.")
                activities.append("AVOID ALL WATER ACTIVITIES.")
        
        # Outdoor activities
        # Check 1: Temperature is within the comfortable range (15°C to 30°C)
        #          AND rain is either unknown/None OR very light (less than 5mm)
        if (temp is not None and 15 < temp < 30 and (rain is None or rain < 5)):
            activities.append("OUTDOOR ACTIVITIES (Ideal Weather):")
            activities.append("— Active: Cycling, Jogging, Hiking")
            activities.append("— Leisure: Picnics, Photography, Reading Outdoors")
    
            # Check 2: Air Quality (PM2.5) check
            if pm25 is not None and pm25 < 50:
                activities.append("Air Quality: Excellent for strenuous outdoor sports and exercise.")
            elif pm25 is not None and 50 <= pm25 < 100:
                activities.append("Air Quality: Moderate. Suitable for most, sensitive groups should be cautious.")
        
            activities.append("") # Keep or remove this line based on desired output spacing

        # ELSE: Suggest indoor activities or issue a warning if conditions are bad
        else:
            activities.append("WEATHER ALERT:")
            if temp is not None and (temp <= 15 or temp >= 30):
                activities.append(f"Temperature ({temp}°C) is outside the comfortable range. Stay Cool/Warm.")
            if rain is not None and rain >= 5:
                activities.append(f"Rainfall ({rain}mm) is high. Consider Indoor Activities.")
            activities.append("")
    
    # Format location
    location_parts = [part.strip() for part in full_name.split(',')]
    if len(location_parts) >= 2:
        city = location_parts[0]
        country = location_parts[-1]
        formatted_location = f"{city}, {country}"
    else:
        formatted_location = full_name
    
    # Generate range data for graph
    range_data = {}
    print("\n=== GENERATING RANGE_DATA ===")

    param_mapping = {
        'RH2M': 'RH2M',
        'SNOWFALL': 'Snowfall_Sum',
        'SNOW_DEPTH': 'Snow_Depth_Max',
        'WAVE_HEIGHT': 'WAVE_HEIGHT',
        'OCEAN_CURRENT': 'OCEAN_CURRENT',
        'SWELL_PERIOD': 'SWELL_PERIOD'
    }

    # เริ่มจาก 4 params พื้นฐาน
    all_params_real = ['T2M', 'PRECTOTCORR', 'WS2M', 'PM25']

    # เพิ่ม optional params ที่ user เลือก (แปลงเป็นชื่อจริง)
    for param in selected_params:
        real_name = param_mapping.get(param, param)
        if real_name not in all_params_real:
            all_params_real.append(real_name)

    print(f"Selected params from UI: {selected_params}")
    print(f"Mapped to real names: {all_params_real}")
    print(f"Available in df: {[p for p in all_params_real if p in df.columns]}")

    # สร้าง range_data โดยใช้ชื่อจริงเป็น key
    for param in all_params_real:
        if param in df.columns:
            range_data[param] = get_range_values(df, param, target_date, days_range=3)
            valid_count = sum(1 for v in range_data[param].values() if v is not None)
            print(f"   {param}: {valid_count} valid values")
        
            # แสดงตัวอย่างค่า
            if valid_count > 0:
                sample = {k: v for k, v in list(range_data[param].items())[:2]}
                print(f"      Sample: {sample}")
        else:
            print(f"{param} NOT in df.columns - SKIPPED")
    
    print(f"\n=== FINAL CHECK BEFORE RETURN ===")
    print(f"predictions keys: {list(results.keys())}")
    print(f"range_data keys: {list(range_data.keys())}")

    # เช็คเฉพาะพารามิเตอร์ที่ user เลือก
    for param in selected_params:
        real_name = param_mapping.get(param, param)
        print(f"\nChecking {param} (real name: {real_name}):")
        print(f"  - In predictions? {real_name in results}")
        print(f"  - In range_data? {real_name in range_data}")
        if real_name in results:
            print(f"  - Prediction value: {results[real_name]['prediction']:.2f}")
        if real_name in range_data:
            print(f"  - Range data points: {len(range_data[real_name])}")

    return {
        'location': formatted_location,
        'date': target_date,
        'predictions': results,
        'probabilities': probabilities,
        'recommendations': recommendations,
        'activity_recommendations': activities,
        'range_data': range_data,
        'selected_params': selected_params  
    }