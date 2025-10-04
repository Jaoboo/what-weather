from kivy.config import Config

Config.set('graphics', 'fullscreen', '0')
Config.set('graphics', 'resizable', '0')
Config.set('graphics', 'width', '1000')
Config.set('graphics', 'height', '800')
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
        self.location_coords = None  # เก็บพิกัด lat, lon
    
    def on_enter(self):
        self.ids.form_frame.ids.form_content.ids.location_input.text = ""
        self.ids.form_frame.ids.form_content.ids.date_input.text = ""
        self.selected_params = []
        self.location_coords = None
    
        # Reset checkboxes ตามที่มีใน design.kv
        self.ids.form_frame.ids.form_content.ids.cb_humidity.active = False
        self.ids.form_frame.ids.form_content.ids.cb_humidity_max.active = False
        self.ids.form_frame.ids.form_content.ids.cb_humidity_min.active = False
        self.ids.form_frame.ids.form_content.ids.cb_snowfall.active = False
        self.ids.form_frame.ids.form_content.ids.cb_snow_depth.active = False
        self.ids.form_frame.ids.form_content.ids.cb_wave_height.active = False
        self.ids.form_frame.ids.form_content.ids.cb_ocean_current.active = False
        self.ids.form_frame.ids.form_content.ids.cb_swell_period.active = False
    
    def on_location_input_change(self, text):
        """เรียกใช้เมื่อผู้ใช้พิมพ์ใน location input"""
        query = text.strip()
        if len(query) < 3:
            return
        
        Thread(target=self._search_location, args=(query,), daemon=True).start()
    
    def _search_location(self, query):
        """ค้นหาสถานที่และเก็บพิกัด"""
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
        """จัดการการเลือก checkbox พารามิเตอร์"""
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
        """แสดง popup เตือน"""
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
        """จัดการเมื่อกดปุ่ม Start"""
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
        
        # แปลงวันที่จากรูปแบบ dd/mm/yyyy เป็น yyyy-mm-dd
        try:
            date_obj = datetime.strptime(date_input, '%d/%m/%Y')
            formatted_date = date_obj.strftime('%Y-%m-%d')
        except ValueError:
            self.show_warning("Invalid date format!\nPlease use DD/MM/YYYY\nExample: 31/12/2025")
            return

        result_screen = self.manager.get_screen("result")
        result_screen.update_result(location_name, formatted_date, self.selected_params, self.location_coords)
        self.manager.current = "result"

class ResultScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_location = ""
        self.current_date = ""
        self.selected_params = []
        self.location_coords = None
        self.map_initialized = False
        
        self.param_labels = {
            'T2M': 'Temperature',
            'PRECTOTCORR': 'Rainfall',
            'WS2M': 'Wind Speed',
            'PM25': 'PM2.5',
            'RH2M': 'Humidity',
            'PS': 'Pressure',
            'CLOUD_AMT': 'Cloud Cover',
            'ALLSKY_SFC_SW_DWN': 'Solar Radiation'
        }
    
    def update_result(self, location, date, selected_params=[], coords=None):
        self.current_location = location
        self.current_date = date
        self.selected_params = selected_params
        self.location_coords = coords
        
        self.ids.location_value.text = location
        self.ids.date_value.text = date
        
        self.ids.summary_box.clear_widgets()
        self.ids.outdoor_activities_box.clear_widgets()
        self.ids.graph_box.clear_widgets()
        
        placeholder = Label(
            text="Data will be displayed here",
            font_size='14sp',
            color=(0.6, 0.6, 0.6, 1)
        )
        self.ids.summary_box.add_widget(placeholder)
        
        placeholder2 = Label(
            text="Activities will be displayed here",
            font_size='14sp',
            color=(0.6, 0.6, 0.6, 1)
        )
        self.ids.outdoor_activities_box.add_widget(placeholder2)
        
        placeholder3 = Label(
            text="Graph will be displayed here",
            font_size='14sp',
            color=(0.6, 0.6, 0.6, 1)
        )
        self.ids.graph_box.add_widget(placeholder3)
        
        # Update table rows
        self.ids.table1_row1_col1.text = "Temperature"
        self.ids.table1_row1_col2.text = "- °C"
        self.ids.table1_row1_col3.text = "Normal"
        
        self.ids.table1_row2_col1.text = "Rainfall"
        self.ids.table1_row2_col2.text = "- mm"
        self.ids.table1_row2_col3.text = "Normal"
        
        self.ids.table2_row1_col1.text = "Wind Speed"
        self.ids.table2_row1_col2.text = "- m/s"
        self.ids.table2_row1_col3.text = "Normal"
        
        self.ids.table2_row2_col1.text = "PM2.5"
        self.ids.table2_row2_col2.text = "- μg/m³"
        self.ids.table2_row2_col3.text = "Good"
        
        # Initialize map
        self.initialize_map()
    
    def initialize_map(self):
        """สร้างแผนที่และปักหมุดตำแหน่ง"""
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
                
                # เพิ่มหมุด
                marker = MapMarkerPopup(lat=lat, lon=lon)
                self.mapview.add_marker(marker)
                
                map_container.add_widget(self.mapview)
                self.map_initialized = True
            else:
                # ถ้าไม่มีพิกัด แสดงข้อความ
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
            filename = f"weather_data_{self.current_location}_{timestamp}.csv"
            
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
                writer.writerow(['Temperature', 'N/A', '°C'])
                writer.writerow(['Rainfall', 'N/A', 'mm'])
                writer.writerow(['Wind Speed', 'N/A', 'm/s'])
                writer.writerow(['PM2.5', 'N/A', 'μg/m³'])
                writer.writerow([])
                
                writer.writerow(['Summary'])
                writer.writerow(['Data will be available after analysis'])
            
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