# WhatWeather - Weather Prediction System

A weather forecasting application using historical data and Prophet machine learning to predict weather conditions up to 7 days in advance.

## Features

- **Core Weather Parameters**: Temperature, Rainfall, Wind Speed, PM2.5
- **Optional Parameters**: Humidity, Snowfall, Snow Depth, Wave Height, Ocean Current, Swell Period
- **Interactive Map**: Location visualization using OpenStreetMap
- **7-Day Trend Graph**: Visual representation of weather patterns
- **Activity Recommendations**: AI-generated suggestions based on weather conditions
- **Export Options**: Download results as CSV, JSON, or PDF

---

## System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **RAM**: Minimum 4GB recommended
- **Internet**: Required for API data fetching

---

## Installation

### 1. Clone or Download the Project

```bash
git clone <https://github.com/Jaoboo/what-weather>
cd whatweather
```

### 2. Install Required Dependencies

```bash
pip install kivy==2.3.0
pip install pandas numpy requests geopy
pip install prophet
pip install matplotlib
pip install kivy-garden.mapview
pip install reportlab
pip install cmdstanpy
```

### 3. Download Loading Animation

Place a file named `loading.gif` in the project root directory (same folder as `main.py`).

---

## File Structure

```
whatweather/
├── main.py              # Main application file
├── analysis.py          # Weather prediction logic
├── design.kv            # UI design file
├── loading.gif          # Loading animation
└── README.md            # This file
```

---

## How to Run

```bash
python main.py
```

The application window (800x600) will open automatically.

---

## Usage Guide

### 1. **Input Location**
   - Enter city and country in English only
   - Example: `Bangkok, Thailand` or `New York, USA`
   - The system will automatically search coordinates

### 2. **Select Date**
   - Format: `DD/MM/YYYY`
   - Example: `25/12/2025`
   - Date must be within reasonable forecast range

### 3. **Choose Optional Parameters** (Max 2)
   - ☑ Humidity
   - ☑ Snowfall
   - ☑ Snow Depth
   - ☑ Wave Height
   - ☑ Ocean Current
   - ☑ Swell Period

### 4. **Click "Prediction"**
   - Wait for data analysis (30-60 seconds)
   - Results will display automatically

### 5. **View Results**
   - **Weather Parameters**: Current predictions with status
   - **Summary**: Weather recommendations
   - **Activity Recommendations**: Safe activities for the conditions
   - **Map**: Location visualization
   - **Graph**: 7-day weather trend

### 6. **Download Results**
   - Click **"Download"** button
   - Choose format: CSV, JSON, or PDF
   - File saves to your Downloads folder

---

## Data Sources

- **NASA POWER**: Temperature, Rainfall (2015-present)
- **Open-Meteo**: Wind Speed, Historical Weather Archive
- **Air Quality API**: PM2.5 data
- **Nominatim**: Location geocoding
- **Synthetic Data**: Ocean parameters (Wave Height, Current, Swell)

---

## Limitations

1. **Forecast Accuracy**: Decreases for dates >7 days ahead
2. **Ocean Data**: Currently synthetic/simulated
3. **Rate Limits**: Geocoding APIs have request limits
4. **Network Required**: Cannot work offline
5. **Snow Data**: Limited to 3 years history
6. **Date Range**: Historical data availability varies by parameter

---

## Troubleshooting

### **"Failed to get coordinates"**
- Check spelling of location name
- Use English names only
- Include country name
- Try broader location (e.g., state instead of specific district)

### **"Analysis failed"**
- Check internet connection
- Verify date format is correct
- Location may not have sufficient historical data
- Try a different date or location

### **Graph not showing optional parameters**
- Ensure parameters have data for your location
- Snow data only available where applicable
- Ocean data is synthetic (for demonstration)

### **Prophet installation fails**
**Windows:**
```bash
# Install Visual C++ Build Tools first
pip install prophet --no-cache-dir
```

**macOS:**
```bash
# Install Xcode Command Line Tools
xcode-select --install
pip install prophet
```

### **Loading animation not showing**
- Ensure `loading.gif` exists in project folder
- Any GIF animation will work (recommended: 150x150px)

### **Map not displaying**
- Check internet connection
- Geocoding may have failed (location coordinates not found)
- Map tiles require internet to load

---

## Performance Tips

1. **First run is slower**: Prophet model compilation takes time
2. **Cache improves speed**: Repeated predictions for same location are faster
3. **Reduce parameters**: Selecting fewer optional parameters speeds up analysis
4. **Network speed**: Faster internet = faster data fetching

---

## Development

### Adding New Weather Parameters

1. **Fetch data** in `analysis.py`:
   ```python
   def fetch_new_param(lat, lon, start, end):
       # API call here
       return dataframe
   ```

2. **Add to predictions**:
   ```python
   for param in df.columns:
       result = predict_with_prophet(df, target_date, param)
   ```

3. **Update UI** in `main.py`:
   - Add to `param_mapping`
   - Add to `param_info`
   - Add colors to graph

---

## Credits

- **Prophet**: Meta's forecasting library
- **Kivy**: Cross-platform UI framework
- **NASA POWER**: Solar and meteorological data
- **Open-Meteo**: Open-source weather API
- **OpenStreetMap**: Map visualization

---

## License

This project is for educational purposes. Please respect API usage limits and terms of service for all data providers.

---

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure Python version is 3.8+
4. Check console output for detailed error messages

---

## Version History

- **v1.0** (2025): Initial release
  - Core weather prediction
  - 6 optional parameters
  - Export to CSV/JSON/PDF
  - Interactive map and graphs

---