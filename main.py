from kivy.config import Config

Config.set('graphics', 'fullscreen', '0')
Config.set('graphics', 'resizable', '0')
Config.set('graphics', 'width', '800')
Config.set('graphics', 'height', '600')
Config.set('graphics', 'borderless', '0')
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

from kivy.app import App
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.image import AsyncImage
from kivy.uix.scrollview import ScrollView
from kivy.lang import Builder
from kivy.graphics import Color, Rectangle, RoundedRectangle, Line
from kivy.properties import ListProperty, StringProperty
from kivy.clock import Clock
from kivy_garden.mapview import MapView, MapMarkerPopup
import re
import csv
from datetime import datetime
import os
import subprocess
import platform
import requests
from threading import Thread
from analysis import run_analysis

class DarkTextInput(TextInput):
    pass

class FormContentFrame(BoxLayout):
    pass

class FormFrame(BoxLayout):
    pass

class LoadingScreen(Screen):
    pass

class DescriptionScreen(Screen):
    def go_to_main(self):
        self.manager.current = "main"

class WeatherCard(BoxLayout):
    title_text = StringProperty("")
    value_text = StringProperty("")
    subtitle_text = StringProperty("")
    value_color = ListProperty([0.2, 0.2, 0.2, 1])
    bg_color = ListProperty([1, 1, 1, 0.95])

class MainFormScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.selected_params = []
        self.location_coords = None
    
    def on_enter(self):
        self.ids.form_frame.ids.form_content.ids.location_input.text = ""
        self.ids.form_frame.ids.form_content.ids.date_input.text = ""
        self.selected_params = []
        self.location_coords = None
    
        # Reset all checkboxes
        self.ids.form_frame.ids.form_content.ids.cb_humidity.active = False
        self.ids.form_frame.ids.form_content.ids.cb_snowfall.active = False
        self.ids.form_frame.ids.form_content.ids.cb_snow_depth.active = False
        self.ids.form_frame.ids.form_content.ids.cb_wave_height.active = False
        self.ids.form_frame.ids.form_content.ids.cb_ocean_current.active = False
        self.ids.form_frame.ids.form_content.ids.cb_swell_period.active = False
    
    def on_location_input_change(self, text):
        query = text.strip()
        if len(query) < 3:
            return
        
        Thread(target=self._search_location, args=(query,), daemon=True).start()
    
    def _search_location(self, query):
        try:
            headers = {
                'User-Agent': 'WhatWeather/1.0',
                'Accept-Language': 'en'
            }
            params = {
                'q': query, 
                'format': 'json', 
                'limit': 1, 
                'addressdetails': 1,
                'accept-language': 'en'
            }
            response = requests.get(
                'https://nominatim.openstreetmap.org/search', 
                params=params, 
                headers=headers, 
                timeout=10
            )
            response.raise_for_status()
            results = response.json()
        
            if results:
                result = results[0]
                lat, lon = float(result['lat']), float(result['lon'])
                self.location_coords = (lat, lon)
                print(f"Found location: {query} at {lat}, {lon}")
        except Exception as e:
            print(f"Search location error: {e}")
            self.location_coords = None

    def on_checkbox_change(self, checkbox, param_name):
        if checkbox.active:
            if len(self.selected_params) >= 2:
                checkbox.active = False
                self.show_warning("You can only select 2 additional parameters!")
                return
            self.selected_params.append(param_name)
        else:
            if param_name in self.selected_params:
                self.selected_params.remove(param_name)
    
    def show_warning(self, message):
        content = BoxLayout(orientation='vertical', padding=20)
        content.add_widget(Label(
            text=message,
            color=(0.3, 0.5, 0.7, 1),
            font_size='16sp'
        ))
    
        with content.canvas.before:
            Color(1, 1, 1, 1)
            rect = RoundedRectangle(
                pos=content.pos, 
                size=content.size,
                radius=[15]
            )
        content.bind(pos=lambda obj, pos: setattr(rect, 'pos', pos))
        content.bind(size=lambda obj, size: setattr(rect, 'size', size))
    
        popup = Popup(
            title='Warning',
            content=content,
            size_hint=(0.7, 0.3),
            background='',
            separator_color=(0.3, 0.5, 0.7, 1),
            title_color=(0.3, 0.5, 0.7, 1)
        )
        popup.open()

    def on_start_button_press(self):
        location_name = self.ids.form_frame.ids.form_content.ids.location_input.text.strip()
        date_input = self.ids.form_frame.ids.form_content.ids.date_input.text.strip()

        if not location_name:
            self.show_warning("Please enter a location!")
            return
        
        if not date_input:
            self.show_warning("Please enter a date!")
            return
        
        if not all(ord(char) < 128 for char in location_name if char.isalpha()):
            self.show_warning("Please enter location in English only!\n\nExample: Bangkok, New York, Tokyo")
            return
        
        try:
            date_obj = datetime.strptime(date_input, '%d/%m/%Y')
            formatted_date = date_obj.strftime('%Y-%m-%d')
        except ValueError:
            self.show_warning("Invalid date format!\nPlease use DD/MM/YYYY\nExample: 31/12/2025")
            return

        self.manager.current = "loading"
        
        Thread(target=self._run_analysis, args=(location_name, formatted_date), daemon=True).start()

    def _run_analysis(self, location_name, formatted_date):
        try:
            print(f"\n=== Starting Analysis ===")
            print(f"Location: {location_name}")
            print(f"Date: {formatted_date}")
            print(f"Selected additional params: {self.selected_params}")
            
            # ส่ง selected_params ไปยัง run_analysis
            results = run_analysis(location_name, formatted_date, selected_params=self.selected_params)
            
            if 'error' in results:
                Clock.schedule_once(lambda dt: self.show_warning(results['error']), 0)
                Clock.schedule_once(lambda dt: setattr(self.manager, 'current', 'main'), 0)
            else:
                results['selected_params'] = self.selected_params
                
                def update_ui(dt):
                    result_screen = self.manager.get_screen("result")
                    result_screen.update_with_analysis(results, self.location_coords)
                    self.manager.current = "result"
                
                Clock.schedule_once(update_ui, 0)
                
        except Exception as e:
            print(f"Analysis error: {e}")
            import traceback
            traceback.print_exc()
            Clock.schedule_once(lambda dt: self.show_warning(f"Analysis failed:\n{str(e)}"), 0)
            Clock.schedule_once(lambda dt: setattr(self.manager, 'current', 'main'), 0)

class ResultScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_location = ""
        self.current_date = ""
        self.selected_params = []
        self.location_coords = None
        self.map_initialized = False
        self.analysis_results = None
    
    def update_with_analysis(self, analysis_results, coords=None):
        self.current_location = analysis_results['location']
        self.current_date = analysis_results['date']
        self.analysis_results = analysis_results
        self.location_coords = coords
        self.selected_params = analysis_results.get('selected_params', [])
        
        self.ids.location_value.text = self.current_location
        self.ids.date_value.text = self.current_date
        
        self.ids.summary_box.clear_widgets()
        self.ids.graph_box.clear_widgets()
        
        predictions = analysis_results.get('predictions', {})
        recommendations = analysis_results.get('recommendations', [])
        
        # สร้าง summary text โดยเน้นที่ผลสรุปสำคัญ
        summary_text = "[b]Weather Summary:[/b]\n\n"
        
        # ข้อมูลสำหรับตรวจสอบ
        temp_value = predictions.get('T2M', {}).get('prediction', 0)
        rain_value = predictions.get('PRECTOTCORR', {}).get('prediction', 0)
        pm_value = predictions.get('PM25', {}).get('prediction', 0)
        humidity_value = predictions.get('RH2M', {}).get('prediction', 0)
        snowfall_value = predictions.get('SNOWFALL', {}).get('prediction', 0)
        
        # 1. สรุปอุณหภูมิ (ลำดับแรก)
        if 'T2M' in predictions:
            temp = temp_value
            if temp < 0:
                summary_text += f"• Temperature: [b]{temp:.1f}°C[/b] - Extremely cold weather\n"
            elif temp < 10:
                summary_text += f"• Temperature: [b]{temp:.1f}°C[/b] - Very cold, wear warm clothes\n"
            elif temp < 15:
                summary_text += f"• Temperature: [b]{temp:.1f}°C[/b] - Cold, light jacket recommended\n"
            elif temp < 25:
                summary_text += f"• Temperature: [b]{temp:.1f}°C[/b] - Comfortable weather\n"
            elif temp < 30:
                summary_text += f"• Temperature: [b]{temp:.1f}°C[/b] - Warm and pleasant\n"
            elif temp < 35:
                summary_text += f"• Temperature: [b]{temp:.1f}°C[/b] - Hot, stay hydrated\n"
            else:
                summary_text += f"• Temperature: [b]{temp:.1f}°C[/b] - Extremely hot, avoid outdoor activities\n"
        
        # 2. สรุปปริมาณฝน (ลำดับสอง)
        if 'PRECTOTCORR' in predictions:
            rain = rain_value
            if rain < 5:
                summary_text += f"• Rainfall: [b]{rain:.1f} mm[/b] - No rain expected, good for outdoor activities\n"
            elif rain < 15:
                summary_text += f"• Rainfall: [b]{rain:.1f} mm[/b] - Light rain, bring umbrella\n"
            elif rain < 35:
                summary_text += f"• Rainfall: [b]{rain:.1f} mm[/b] - Moderate rain, umbrella and raincoat needed\n"
            else:
                summary_text += f"• Rainfall: [b]{rain:.1f} mm[/b] - Heavy rain, stay indoors if possible\n"
        
        # 2.1 ถ้าฝนน้อยกว่า 5 ให้แสดงความชื้นแทน
        if rain_value < 5 and 'RH2M' in predictions:
            humidity = humidity_value
            if humidity < 30:
                summary_text += f"• Humidity: [b]{humidity:.1f}%[/b] - Low humidity, use moisturizer\n"
            elif humidity < 60:
                summary_text += f"• Humidity: [b]{humidity:.1f}%[/b] - Comfortable humidity level\n"
            else:
                summary_text += f"• Humidity: [b]{humidity:.1f}%[/b] - High humidity, may feel muggy\n"
        
        # 3. สรุปฝุ่น PM2.5 (ลำดับสาม)
        if 'PM25' in predictions:
            pm = pm_value
            if pm < 50:
                summary_text += f"• PM2.5: [b]{pm:.1f} μg/m³[/b] - Good air quality, safe for all activities\n"
            elif pm < 100:
                summary_text += f"• PM2.5: [b]{pm:.1f} μg/m³[/b] - Moderate, sensitive groups limit outdoor exposure\n"
            elif pm < 150:
                summary_text += f"• PM2.5: [b]{pm:.1f} μg/m³[/b] - Unhealthy, wear mask outdoors\n"
            else:
                summary_text += f"• PM2.5: [b]{pm:.1f} μg/m³[/b] - Hazardous, stay indoors with air purifier\n"
        
        # 4. สรุปหิมะ (ถ้ามีค่าไม่เท่ากับศูนย์)
        if 'SNOWFALL' in predictions and snowfall_value > 0:
            snow = snowfall_value
            if snow < 1:
                summary_text += f"• Snowfall: [b]{snow:.1f} mm[/b] - Trace amounts of snow\n"
            elif snow < 5:
                summary_text += f"• Snowfall: [b]{snow:.1f} mm[/b] - Light snowfall expected\n"
            else:
                summary_text += f"• Snowfall: [b]{snow:.1f} mm[/b] - Heavy snowfall, drive carefully and dress warmly\n"
        
        # 5. ข้อมูล optional parameters อื่นๆ
        other_params_shown = False
        for param in self.selected_params:
            # ข้ามความชื้นถ้าแสดงไปแล้ว หรือข้ามหิมะถ้าแสดงไปแล้ว
            if param == 'RH2M' and rain_value < 5:
                continue
            if param == 'SNOWFALL' and snowfall_value > 0:
                continue
            
            if param in predictions:
                value = predictions[param]['prediction']
                
                if not other_params_shown:
                    summary_text += f"\n[b]Additional Parameters:[/b]\n"
                    other_params_shown = True
                
                if param == 'RH2M':
                    if value < 30:
                        summary_text += f"• Humidity: [b]{value:.1f}%[/b] - Low\n"
                    elif value < 60:
                        summary_text += f"• Humidity: [b]{value:.1f}%[/b] - Comfortable\n"
                    else:
                        summary_text += f"• Humidity: [b]{value:.1f}%[/b] - High\n"
                elif param == 'SNOW_DEPTH':
                    if value > 0:
                        summary_text += f"• Snow Depth: [b]{value:.1f} cm[/b]\n"
                elif param == 'WAVE_HEIGHT':
                    if value < 1:
                        summary_text += f"• Wave Height: [b]{value:.1f} m[/b] - Calm seas\n"
                    elif value < 2:
                        summary_text += f"• Wave Height: [b]{value:.1f} m[/b] - Moderate waves\n"
                    else:
                        summary_text += f"• Wave Height: [b]{value:.1f} m[/b] - Rough seas\n"
                elif param == 'OCEAN_CURRENT':
                    summary_text += f"• Ocean Current: [b]{value:.1f} m/s[/b]\n"
                elif param == 'SWELL_PERIOD':
                    summary_text += f"• Swell Period: [b]{value:.1f} s[/b]\n"
        
        summary_label = Label(
            text=summary_text,
            markup=True,
            font_size='14sp',
            color=(0.3, 0.3, 0.3, 1),
            halign='left',
            valign='top',
            size_hint_y=None,
            height=450
        )
        summary_label.bind(size=lambda obj, size: setattr(obj, 'text_size', (size[0] - 10, None)))
        self.ids.summary_box.add_widget(summary_label)
        
        # Update weather cards
        self.update_weather_cards(predictions)
        
        graph_label = Label(
            text="Graph data loaded\n(Visualization coming soon)",
            font_size='14sp',
            color=(0.6, 0.6, 0.6, 1)
        )
        self.ids.graph_box.add_widget(graph_label)
        
        self.initialize_map()
    
    def update_weather_cards(self, predictions):
        """Update weather parameter cards - แสดง 4-6 การ์ดตามที่ user เลือก"""
        default_params = ['T2M', 'PRECTOTCORR', 'WS2M', 'PM25']
        
        # รวม default + selected params
        all_params = default_params + self.selected_params
        
        param_info = {
            'T2M': {'title': 'Temperature', 'unit': '°C', 'getter': self.get_temp_status},
            'PRECTOTCORR': {'title': 'Rainfall', 'unit': 'mm', 'getter': self.get_rain_status},
            'WS2M': {'title': 'Wind Speed', 'unit': 'm/s', 'getter': self.get_wind_status},
            'PM25': {'title': 'PM2.5', 'unit': 'μg/m³', 'getter': self.get_pm_status},
            'RH2M': {'title': 'Humidity', 'unit': '%', 'getter': lambda v: self.get_generic_status(v, 'humidity')},
            'SNOWFALL': {'title': 'Snowfall', 'unit': 'mm', 'getter': lambda v: self.get_generic_status(v, 'snow')},
            'SNOW_DEPTH': {'title': 'Snow Depth', 'unit': 'cm', 'getter': lambda v: self.get_generic_status(v, 'snow')},
            'WAVE_HEIGHT': {'title': 'Wave Height', 'unit': 'm', 'getter': lambda v: self.get_generic_status(v, 'wave')},
            'OCEAN_CURRENT': {'title': 'Ocean Current', 'unit': 'm/s', 'getter': lambda v: self.get_generic_status(v, 'current')},
            'SWELL_PERIOD': {'title': 'Swell Period', 'unit': 's', 'getter': lambda v: self.get_generic_status(v, 'swell')},
        }
        
        # ปรับ grid layout ตามจำนวน parameters
        card_grid = self.ids.card_grid
        total_cards = len(all_params)
        
        if total_cards == 5:
            # 5 การ์ด: แถวแรก 3 การ์ด, แถวสอง 2 การ์ด (ตรงกลาง)
            card_grid.cols = 3
            card_grid.rows = 2
        elif total_cards == 6:
            # 6 การ์ด: 3x2 grid
            card_grid.cols = 3
            card_grid.rows = 2
        else:
            # 4 การ์ด: 2x2 grid
            card_grid.cols = 2
            card_grid.rows = 2
        
        # ซ่อนการ์ดทั้งหมดก่อน
        for i in range(1, 7):
            card = self.ids[f'card{i}']
            card.opacity = 0
            card.size_hint = (None, None)
            card.size = (0, 0)
        
        # แสดงและใส่ข้อมูลเฉพาะการ์ดที่ต้องการ
        for i, param in enumerate(all_params):
            if i < 6:  # สูงสุด 6 การ์ด
                card_id = f'card{i+1}'
                card = self.ids[card_id]
                
                # แสดงการ์ด
                card.opacity = 1
                card.size_hint = (1, 1)
                
                if param in predictions and param in param_info:
                    info = param_info[param]
                    value = predictions[param]['prediction']
                    
                    card.ids.param_title.text = info['title']
                    card.ids.param_value.text = f"{value:.1f}"
                    card.ids.param_unit.text = info['unit']
                    card.ids.param_status.text = info['getter'](value)
                else:
                    card.ids.param_title.text = param
                    card.ids.param_value.text = "-"
                    card.ids.param_unit.text = ""
                    card.ids.param_status.text = "No data"
    
    def get_temp_status(self, temp):
        if temp < 10:
            return "Very Cold"
        elif temp < 20:
            return "Cool"
        elif temp < 30:
            return "Warm"
        elif temp < 35:
            return "Hot"
        else:
            return "Very Hot"
    
    def get_rain_status(self, rain):
        if rain < 5:
            return "Dry"
        elif rain < 15:
            return "Light Rain"
        elif rain < 35:
            return "Moderate Rain"
        else:
            return "Heavy Rain"
    
    def get_wind_status(self, wind):
        if wind < 5:
            return "Calm"
        elif wind < 10:
            return "Moderate"
        else:
            return "Strong"
    
    def get_pm_status(self, pm):
        if pm < 50:
            return "Good"
        elif pm < 100:
            return "Moderate"
        elif pm < 150:
            return "Unhealthy"
        else:
            return "Hazardous"
    
    def get_generic_status(self, value, type_name):
        """Generic status for additional parameters"""
        if type_name == 'humidity':
            if value < 30:
                return "Low"
            elif value < 60:
                return "Comfortable"
            else:
                return "High"
        elif type_name == 'snow':
            if value < 1:
                return "None"
            elif value < 5:
                return "Light"
            else:
                return "Heavy"
        elif type_name == 'wave':
            if value < 1:
                return "Calm"
            elif value < 2:
                return "Moderate"
            else:
                return "Rough"
        elif type_name == 'current':
            if value < 0.5:
                return "Slow"
            elif value < 1:
                return "Moderate"
            else:
                return "Fast"
        elif type_name == 'swell':
            if value < 5:
                return "Short"
            elif value < 10:
                return "Medium"
            else:
                return "Long"
        return "Normal"
    
    def initialize_map(self):
        try:
            map_container = self.ids.map_box
            map_container.clear_widgets()
            
            if self.location_coords:
                lat, lon = self.location_coords
                self.mapview = MapView(
                    zoom=10,
                    lat=lat,
                    lon=lon,
                    snap_to_zoom=False,
                    double_tap_zoom=True,
                    map_source='osm'
                )
                
                marker = MapMarkerPopup(lat=lat, lon=lon)
                self.mapview.add_marker(marker)
                
                map_container.add_widget(self.mapview)
                self.map_initialized = True
            else:
                no_map_label = Label(
                    text="[Map location not available]",
                    font_size='16sp',
                    halign='center',
                    valign='middle',
                    color=(0.5, 0.5, 0.5, 1)
                )
                map_container.add_widget(no_map_label)
                
        except Exception as e:
            print(f"Error initializing map: {e}")
    
    def download_csv(self):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"weather_data_{self.current_location.replace(', ', '_')}_{timestamp}.csv"
            
            downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
            if not os.path.exists(downloads_path):
                downloads_path = os.path.expanduser("~")
            
            filepath = os.path.join(downloads_path, filename)
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                writer.writerow(['Weather Data Report'])
                writer.writerow(['Generated:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                writer.writerow([])
                
                writer.writerow(['Location:', self.current_location])
                writer.writerow(['Target Date:', self.current_date])
                writer.writerow([])
                
                writer.writerow(['Parameter', 'Prediction', 'Unit'])
                
                if self.analysis_results and 'predictions' in self.analysis_results:
                    predictions = self.analysis_results['predictions']
                    
                    if 'T2M' in predictions:
                        writer.writerow(['Temperature', f"{predictions['T2M']['prediction']:.1f}", '°C'])
                    if 'PRECTOTCORR' in predictions:
                        writer.writerow(['Rainfall', f"{predictions['PRECTOTCORR']['prediction']:.1f}", 'mm'])
                    if 'WS2M' in predictions:
                        writer.writerow(['Wind Speed', f"{predictions['WS2M']['prediction']:.1f}", 'm/s'])
                    if 'PM25' in predictions:
                        writer.writerow(['PM2.5', f"{predictions['PM25']['prediction']:.1f}", 'μg/m³'])
                else:
                    writer.writerow(['Temperature', 'N/A', '°C'])
                    writer.writerow(['Rainfall', 'N/A', 'mm'])
                    writer.writerow(['Wind Speed', 'N/A', 'm/s'])
                    writer.writerow(['PM2.5', 'N/A', 'μg/m³'])
                
                writer.writerow([])
                writer.writerow(['Recommendations'])
                if self.analysis_results and 'recommendations' in self.analysis_results:
                    for rec in self.analysis_results['recommendations']:
                        writer.writerow([rec])
            
            message = f"File saved to:\n{filename}\n\nLocation: Downloads folder"
            self.show_success(message, filepath)
            
        except Exception as e:
            self.show_error(f"Error saving file:\n{str(e)}")
    
    def show_success(self, message, filepath=None):
        content = BoxLayout(orientation='vertical', padding=20, spacing=15)
        
        success_label = Label(
            text="✓",
            font_size='48sp',
            size_hint_y=None,
            height=60,
            color=(0.2, 0.7, 0.3, 1),
            bold=True
        )
        content.add_widget(success_label)
        
        main_label = Label(
            text="Download Complete!",
            font_size='18sp',
            size_hint_y=None,
            height=30,
            color=(0.2, 0.2, 0.2, 1),
            bold=True
        )
        content.add_widget(main_label)
        
        detail_label = Label(
            text=message,
            color=(0.4, 0.4, 0.4, 1),
            font_size='13sp',
            size_hint_y=None,
            height=60,
            halign='center',
            valign='middle'
        )
        detail_label.bind(size=lambda obj, size: setattr(obj, 'text_size', (size[0] - 40, None)))
        content.add_widget(detail_label)
        
        if filepath:
            button_box = BoxLayout(
                orientation='horizontal',
                size_hint_y=None,
                height=50,
                spacing=10,
                padding=[20, 0]
            )
            
            open_folder_btn = Button(
                text="Open Folder",
                size_hint=(0.5, None),
                height=45,
                background_normal='',
                background_color=(0, 0, 0, 0),
                color=(1, 1, 1, 1),
                font_size='14sp',
                bold=True
            )
            
            with open_folder_btn.canvas.before:
                Color(0.2, 0.7, 0.3, 1)
                folder_btn_rect = RoundedRectangle(
                    pos=open_folder_btn.pos,
                    size=open_folder_btn.size,
                    radius=[8]
                )
            open_folder_btn.bind(pos=lambda obj, pos: setattr(folder_btn_rect, 'pos', pos))
            open_folder_btn.bind(size=lambda obj, size: setattr(folder_btn_rect, 'size', size))
            
            def open_folder(instance):
                folder_path = os.path.dirname(filepath)
                
                try:
                    if platform.system() == "Windows":
                        subprocess.Popen(f'explorer /select,"{filepath}"')
                    elif platform.system() == "Darwin":
                        subprocess.Popen(["open", "-R", filepath])
                    else:
                        subprocess.Popen(["xdg-open", folder_path])
                except Exception as e:
                    print(f"Error opening folder: {e}")
            
            open_folder_btn.bind(on_release=open_folder)
            
            close_btn = Button(
                text="Close",
                size_hint=(0.5, None),
                height=45,
                background_normal='',
                background_color=(0, 0, 0, 0),
                color=(0.4, 0.4, 0.4, 1),
                font_size='14sp',
                bold=True
            )
            
            with close_btn.canvas.before:
                Color(0.9, 0.9, 0.9, 1)
                close_btn_rect = RoundedRectangle(
                    pos=close_btn.pos,
                    size=close_btn.size,
                    radius=[8]
                )
            close_btn.bind(pos=lambda obj, pos: setattr(close_btn_rect, 'pos', pos))
            close_btn.bind(size=lambda obj, size: setattr(close_btn_rect, 'size', size))
            
            button_box.add_widget(close_btn)
            button_box.add_widget(open_folder_btn)
            content.add_widget(button_box)
        
        with content.canvas.before:
            Color(1, 1, 1, 1)
            rect = RoundedRectangle(
                pos=content.pos, 
                size=content.size,
                radius=[15]
            )
        content.bind(pos=lambda obj, pos: setattr(rect, 'pos', pos))
        content.bind(size=lambda obj, size: setattr(rect, 'size', size))
        
        popup = Popup(
            title='',
            content=content,
            size_hint=(0.6, 0.55),
            background='',
            separator_height=0
        )
        
        if filepath:
            close_btn.bind(on_release=popup.dismiss)
        
        popup.open()
    
    def show_error(self, message):
        content = BoxLayout(orientation='vertical', padding=20)
        content.add_widget(Label(
            text=message,
            color=(0.8, 0.2, 0.2, 1),
            font_size='14sp'
        ))
        
        with content.canvas.before:
            Color(1, 1, 1, 1)
            rect = RoundedRectangle(
                pos=content.pos, 
                size=content.size,
                radius=[15]
            )
        content.bind(pos=lambda obj, pos: setattr(rect, 'pos', pos))
        content.bind(size=lambda obj, size: setattr(rect, 'size', size))
        
        popup = Popup(
            title='Error',
            content=content,
            size_hint=(0.7, 0.4),
            background='',
            separator_color=(0.8, 0.2, 0.2, 1),
            title_color=(0.8, 0.2, 0.2, 1)
        )
        popup.open()

class WhatWeather(App):
    def build(self):
        Window.fullscreen = False
        Window.borderless = False
        Window.resizable = False
        
        try:
            Builder.load_file('design.kv')
        except Exception as e:
            print(f"Error loading KV file: {e}")
            return Label(text=f"Error loading interface:\n{str(e)}")
        
        sm = ScreenManager()
        sm.add_widget(DescriptionScreen(name='description'))
        sm.add_widget(MainFormScreen(name='main'))
        sm.add_widget(LoadingScreen(name='loading'))
        sm.add_widget(ResultScreen(name='result'))
        sm.current = 'description'
        return sm

if __name__ == '__main__':
    WhatWeather().run()