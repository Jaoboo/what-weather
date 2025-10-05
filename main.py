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
import json
from threading import Thread
from analysis import run_analysis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from kivy.core.image import Image as CoreImage
from kivy.uix.image import Image

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
            error_msg = f"Analysis failed:\n{str(e)}"  # ✅ เก็บค่าไว้ก่อน
            Clock.schedule_once(lambda dt: self.show_warning(error_msg), 0)
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
        self.download_menu = None
    
    def update_with_analysis(self, analysis_results, coords=None):
        self.current_location = analysis_results['location']
        self.current_date = analysis_results['date']
        self.analysis_results = analysis_results
        self.location_coords = coords
        self.selected_params = analysis_results.get('selected_params', [])

        self.ids.location_value.text = self.current_location
        self.ids.date_value.text = self.current_date

        # ล้าง widgets ตั้งแต่ต้น
        self.ids.summary_box.clear_widgets()
        self.ids.activity_box.clear_widgets()
        self.ids.graph_box.clear_widgets()

        predictions = analysis_results.get('predictions', {})
        recommendations = analysis_results.get('recommendations', [])

        # === 1. Summary ===
        summary_text = "[b][size=20sp][color=#4099FF]Summary[/color][/size][/b]\n\n"

        if recommendations:
            for rec in recommendations:
                summary_text += f"• {rec}\n"
        else:
                summary_text += "No recommendations available"

        summary_label = Label(
            text=summary_text,
            markup=True,
            font_size='16sp',
            color=(0.3, 0.3, 0.3, 1),
            halign='left',
            valign='top',
            size_hint=[1, None],
            height=250,
            padding=[5, 5, 10, 10]
        )
        summary_label.bind(
            size=lambda obj, size: setattr(obj, 'text_size', size)
        )
        self.ids.summary_box.add_widget(summary_label)

        # === 2. Activity Recommendations ===
        activity_text = "[b][size=20sp][color=#994C1A]Activity Recommendations[/color][/size][/b]\n\n"

        activities = analysis_results.get('activity_recommendations', [])

        if not activities:
            activities = ["No activity recommendations available"]

        for activity in activities:
            activity_text += f"{activity}\n"

        activity_label = Label(
            text=activity_text.strip(),
            markup=True,
            font_size='16sp',
            color=(0.4, 0.2, 0.1, 1),
            halign='left',
            valign='top',
            size_hint=[1, None],
            height=250,
            padding=[5, 5, 10, 10]
        )
        activity_label.bind(
            size=lambda obj, size: setattr(obj, 'text_size', size)
        )
        self.ids.activity_box.add_widget(activity_label)
    
        # === 3. Weather Cards ===
        self.update_weather_cards(predictions)
    
        # === 4. Graph ===
        range_data = analysis_results.get('range_data', {})

        if range_data:
            # สร้างกราฟ
            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('#f8f8f8')

            colors = {
                'T2M': '#FF6B6B',
                'PRECTOTCORR': '#4ECDC4',
                'WS2M': '#45B7D1',
                'PM25': '#FFA07A',
                'RH2M': '#98D8C8',
                'SNOWFALL': '#A8E6CF',
                'SNOW_DEPTH': '#FFD3B6',
                'WAVE_HEIGHT': '#8E94F2',
                'OCEAN_CURRENT': '#FDA7DF',
                'SWELL_PERIOD': '#C7CEEA'
        }

            param_labels = {
                'T2M': 'Temperature (°C)',
                'PRECTOTCORR': 'Rainfall (mm)',
                'WS2M': 'Wind Speed (m/s)',
                'PM25': 'PM2.5 (μg/m³)',
                'RH2M': 'Humidity (%)',
                'SNOWFALL': 'Snowfall (mm)',
                'SNOW_DEPTH': 'Snow Depth (cm)',
                'WAVE_HEIGHT': 'Wave Height (m)',
                'OCEAN_CURRENT': 'Ocean Current (m/s)',
                'SWELL_PERIOD': 'Swell Period (s)'
            }

            for param, values in range_data.items():
                if values and param in colors:
                    dates = list(values.keys())
                    data = [values[d] if values[d] is not None else 0 for d in dates]
                    date_labels = [datetime.strptime(d, '%Y-%m-%d').strftime('%d/%m') for d in dates]
            
                    ax.plot(date_labels, data, 
                        marker='o', 
                        linewidth=2, 
                        markersize=6,
                        color=colors[param],
                        label=param_labels.get(param, param))

            ax.grid(True, alpha=0.3, linestyle='--')

            ax.yaxis.set_visible(False)

            num_params = len(range_data)
            if num_params == 4:
                ncol = 2
            elif num_params == 5:
                ncol = 3
            else:
                ncol = 3

            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), 
                      ncol=ncol, frameon=False, fontsize=9)

            plt.xticks(rotation=45, ha='right', fontsize=9)
            plt.yticks(fontsize=9)
            plt.tight_layout()

            # บันทึกกราฟเป็นไฟล์ชั่วคราว สำหรับใส่ใน PDF
            downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
            self.temp_graph_path = os.path.join(downloads_path, 'temp_weather_graph.png')
            plt.savefig(self.temp_graph_path, format='png', dpi=150, bbox_inches='tight', facecolor='white')

            # บันทึกกราฟเป็น image สำหรับแสดงใน Kivy
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)

            graph_image = Image()
            graph_image.texture = CoreImage(buf, ext='png').texture
            self.ids.graph_box.add_widget(graph_image)
        else:
            self.temp_graph_path = None
            graph_label = Label(
                text="No graph data available",
                font_size='14sp',
                color=(0.6, 0.6, 0.6, 1)
            )
            self.ids.graph_box.add_widget(graph_label)
    
        # === 5. Map ===
        self.initialize_map()
    
    def update_weather_cards(self, predictions):
        """Update weather parameter cards - แสดง 4-6 การ์ดตามที่ user เลือก"""
        default_params = ['T2M', 'PRECTOTCORR', 'WS2M', 'PM25']

        # รวม default + selected params
        all_params = default_params + self.selected_params

        # กำหนดสีพื้นหลัง (โปร่งแสง)
        param_bg_colors = {
            'T2M': [1.0, 0.42, 0.42, 0.15],
            'PRECTOTCORR': [0.31, 0.80, 0.77, 0.15],
            'WS2M': [0.27, 0.72, 0.82, 0.15],
            'PM25': [1.0, 0.63, 0.48, 0.15],
            'RH2M': [0.60, 0.85, 0.78, 0.15],
            'SNOWFALL': [0.66, 0.90, 0.81, 0.15],
            'SNOW_DEPTH': [1.0, 0.83, 0.71, 0.15],
            'WAVE_HEIGHT': [0.56, 0.58, 0.95, 0.15],
            'OCEAN_CURRENT': [0.99, 0.65, 0.87, 0.15],
            'SWELL_PERIOD': [0.78, 0.81, 0.92, 0.15],
        }

        # กำหนดสีตัวอักษร (เข้มกว่าพื้นหลัง)
        param_text_colors = {
            'T2M': [0.8, 0.2, 0.2, 1],           # แดงเข้ม
            'PRECTOTCORR': [0.15, 0.55, 0.52, 1], # เขียวน้ำทะเลเข้ม
            'WS2M': [0.15, 0.45, 0.58, 1],        # ฟ้าเข้ม
            'PM25': [0.8, 0.35, 0.2, 1],          # ส้มเข้ม
            'RH2M': [0.3, 0.6, 0.5, 1],           # เขียวอ่อนเข้ม
            'SNOWFALL': [0.4, 0.65, 0.55, 1],     # เขียวพาสเทลเข้ม
            'SNOW_DEPTH': [0.8, 0.5, 0.35, 1],    # ส้มอ่อนเข้ม
            'WAVE_HEIGHT': [0.35, 0.35, 0.7, 1],  # ม่วงน้ำเงินเข้ม
            'OCEAN_CURRENT': [0.75, 0.3, 0.55, 1], # ชมพูเข้ม
            'SWELL_PERIOD': [0.5, 0.55, 0.7, 1],  # ม่วงอ่อนเข้ม
        }

        param_info = {
            'T2M': {'title': 'Temperature', 'unit': '°C', 'getter': self.get_temp_status},
            'PRECTOTCORR': {'title': 'Rainfall', 'unit': 'mm', 'getter': self.get_rain_status},
            'WS2M': {'title': 'Wind Speed', 'unit': 'm/s', 'getter': self.get_wind_status},
            'PM25': {'title': 'PM2.5', 'unit': 'μg/m³', 'getter': self.get_pm_status},
            'RH2M': {'title': 'Humidity', 'unit': '%', 'getter': self.get_humidity_status},
            'SNOWFALL': {'title': 'Snowfall', 'unit': 'mm', 'getter': self.get_snow_status},
            'SNOW_DEPTH': {'title': 'Snow Depth', 'unit': 'cm', 'getter': self.get_snow_status},
            'WAVE_HEIGHT': {'title': 'Wave Height', 'unit': 'm', 'getter': self.get_wave_status},
            'OCEAN_CURRENT': {'title': 'Ocean Current', 'unit': 'm/s', 'getter': self.get_current_status},
            'SWELL_PERIOD': {'title': 'Swell Period', 'unit': 's', 'getter': self.get_swell_status},
        }

        # ปรับ grid layout ตามจำนวน parameters
        card_grid = self.ids.card_grid
        total_cards = len(all_params)

        if total_cards == 5:
            card_grid.cols = 3
            card_grid.rows = 2
        elif total_cards == 6:
            card_grid.cols = 3
            card_grid.rows = 2
        else:
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

                    # เปลี่ยนสีพื้นหลังการ์ด
                    if param in param_bg_colors:
                        card.canvas.before.clear()
                        with card.canvas.before:
                            Color(*param_bg_colors[param])
                            RoundedRectangle(pos=card.pos, size=card.size, radius=[12])
                        card.bind(pos=lambda obj, pos, c=card, clr=param_bg_colors[param]: 
                                  self._update_card_bg(c, clr))
                        card.bind(size=lambda obj, size, c=card, clr=param_bg_colors[param]: 
                                  self._update_card_bg(c, clr))

                    # เปลี่ยนสีตัวอักษร
                    text_color = param_text_colors.get(param, [0.2, 0.2, 0.2, 1])
                    card.param_title.color = text_color
                    card.param_value.color = text_color
                    card.param_status.color = [text_color[0]*0.8, text_color[1]*0.8, text_color[2]*0.8, 1]  # เข้มกว่าอีกนิด

                    # ใส่ข้อมูล
                    card.param_title.text = info['title']
                    card.param_value.text = f"{value:.1f} {info['unit']}"
                    card.param_unit.text = ""
                    card.param_status.text = info['getter'](value)
                else:
                    card.param_title.text = param
                    card.param_value.text = "-"
                    card.param_unit.text = ""
                    card.param_status.text = "No data"

    def _update_card_bg(self, card, color):
        """Update card background color when position/size changes"""
        card.canvas.before.clear()
        with card.canvas.before:
            from kivy.graphics import Color, RoundedRectangle
            Color(*color)
            RoundedRectangle(pos=card.pos, size=card.size, radius=[12])
    
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
    
    def get_humidity_status(self, humidity):
        if humidity < 30:
            return "Low"
        elif humidity > 30 and humidity < 70:
            return "Comfortable"
        else:
            return "High"
    
    def get_snow_status(self, snow):
        if snow < 5:
            return "Light"
        else:
            return "Heavy"
    
    def get_wave_status(self, wave):
        if wave < 1:
            return "Calm"
        elif wave >= 1:
            return "Moderate"
        else:
            return "Rough"
    
    def get_current_status(self, current):
        """Get ocean current status"""
        if current < 0.5:
            return "Weak"
        elif current < 1.0:
            return "Moderate"
        else:
            return "Strong"

    def get_swell_status(self, swell):
        """Get swell period status"""
        if swell < 6:
            return "Short"
        elif swell < 10:
            return "Medium"
        else:
            return "Long"
    
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
    
    def toggle_download_menu(self):
        """แสดง/ซ่อน dropdown menu"""
        if self.download_menu:
            self.download_menu.dismiss()
            self.download_menu = None
            return
    
        content = BoxLayout(orientation='vertical', spacing=0, padding=0)
    
        buttons_data = [
            ("CSV", 'csv'),
            ("JSON", 'json'),
            ("PDF", 'pdf')
        ]
    
        for i, (text, file_type) in enumerate(buttons_data):
            btn = Button(
                text=text,
                size_hint_y=None,
                height=45,
                background_normal='',
                background_color=(0, 0, 0, 0),
                color=(0.3, 0.3, 0.3, 1),
                font_size='13sp',
                bold=False
            )
        
            if i == 0:
                radius = [8, 8, 0, 0]
            elif i == len(buttons_data) - 1:
                radius = [0, 0, 8, 8]
            else:
                radius = [0, 0, 0, 0]
        
            with btn.canvas.before:
                btn.bg_color = Color(1, 1, 1, 1)
                btn.bg_rect = RoundedRectangle(pos=btn.pos, size=btn.size, radius=radius)
                btn.border_color = Color(0.85, 0.85, 0.85, 1)
                btn.border_line = Line(
                    rounded_rectangle=(btn.x, btn.y, btn.width, btn.height) + tuple(radius),
                    width=1
                )
        
            btn.radius_values = radius
            btn.bind(pos=self._update_btn_graphics, size=self._update_btn_graphics)
            Window.bind(mouse_pos=lambda w, pos, b=btn: self._on_mouse_move(b, pos))
            btn.bind(on_release=lambda x, ft=file_type: self.download_file(ft))
        
            content.add_widget(btn)
    
        with content.canvas.before:
            Color(0, 0, 0, 0.1)
            shadow = RoundedRectangle(
                pos=(content.x + 2, content.y - 2), 
                size=content.size, 
                radius=[8]
            )
    
        content.bind(pos=lambda obj, pos: setattr(shadow, 'pos', (pos[0] + 2, pos[1] - 2)))
        content.bind(size=lambda obj, size: setattr(shadow, 'size', size))
    
        # สร้าง popup โดยไม่ระบุ pos_hint
        self.download_menu = Popup(
            content=content,
            size_hint=(None, None),
            size=(140, 135),
            separator_height=0,
            auto_dismiss=True,
            background='',
            background_color=(0, 0, 0, 0),
            attach_to=None
        )
    
        # คำนวณตำแหน่งทันทีหลังสร้าง
        def set_position(dt):
            if self.download_menu:
                # ตำแหน่งปุ่ม Download
                btn_x = Window.width - 170  # 140 (width) + 30 (margin)
                btn_y = Window.height - 70 - 45  # จากบน: 70 (header) + 45 (button height)
            
                # วาง dropdown ใต้ปุ่ม
                menu_x = btn_x
                menu_y = btn_y - 138  # 135 (menu) + 3 (gap)
            
                self.download_menu.pos = (menu_x, menu_y)
    
        self.download_menu.open()
        Clock.schedule_once(set_position, 0)
    
        # เงา
        with content.canvas.before:
            Color(0, 0, 0, 0.1)
            shadow = RoundedRectangle(
                pos=(content.x + 2, content.y - 2), 
                size=content.size, 
                radius=[8]
            )
    
        content.bind(pos=lambda obj, pos: setattr(shadow, 'pos', (pos[0] + 2, pos[1] - 2)))
        content.bind(size=lambda obj, size: setattr(shadow, 'size', size))
    
        self.download_menu.open()

    def _update_btn_graphics(self, btn, *args):
        """อัพเดทกราฟิกของปุ่ม"""
        if hasattr(btn, 'bg_rect'):
            btn.bg_rect.pos = btn.pos
            btn.bg_rect.size = btn.size
        if hasattr(btn, 'border_line') and hasattr(btn, 'radius_values'):
            btn.border_line.rounded_rectangle = (btn.x, btn.y, btn.width, btn.height) + tuple(btn.radius_values)

    def _on_mouse_move(self, btn, pos):
        """เปลี่ยนสีเมื่อเมาส์ผ่าน"""
        if not self.download_menu or not self.download_menu._window:
            return
    
        if btn.collide_point(*btn.to_widget(*pos)):
            if hasattr(btn, 'bg_color'):
                btn.bg_color.rgba = [0.93, 0.96, 1, 1]
            if hasattr(btn, 'border_color'):
                btn.border_color.rgba = [0.2, 0.6, 1, 1]
            btn.color = [0.2, 0.6, 1, 1]
            btn.bold = True
        else:
            if hasattr(btn, 'bg_color'):
                btn.bg_color.rgba = [1, 1, 1, 1]
            if hasattr(btn, 'border_color'):
                btn.border_color.rgba = [0.85, 0.85, 0.85, 1]
            btn.color = [0.3, 0.3, 0.3, 1]
            btn.bold = False

    def _update_menu_position(self, *args):
        """อัพเดทตำแหน่ง dropdown menu ให้ติดใต้ปุ่ม Download"""
        if not self.download_menu:
            return
    
        try:
            download_btn = self.ids.download_btn
        
            # คำนวณตำแหน่งให้ติดใต้ปุ่ม
            menu_x = download_btn.x
            menu_y = download_btn.y - 135
        
            self.download_menu.pos = (menu_x, menu_y)
        
        except AttributeError as e:
        # ถ้าหาปุ่มไม่เจอ คำนวณจากขอบขวาบน
            menu_x = Window.width - 170
            menu_y = Window.height - 70 - 45 - 135

            self.download_menu.pos = (menu_x, menu_y)

    def download_file(self, file_type):
        """ดาวน์โหลดไฟล์ตามประเภทที่เลือก"""
        if self.download_menu:
            self.download_menu.dismiss()
            self.download_menu = None
    
        if file_type == 'csv':
            self.download_csv()
        elif file_type == 'json':
            self.download_json()
        elif file_type == 'pdf':
            self.download_pdf()
    
    def download_csv(self):
        """ดาวน์โหลดข้อมูลเป็น CSV (รูปแบบละเอียด)"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"weather_data_{self.current_location.replace(', ', '_')}_{timestamp}.csv"
        
            downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
            filepath = os.path.join(downloads_path, filename)
        
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
            
                # Header
                writer.writerow(['Weather Forecast Report'])
                writer.writerow(['Generated', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                writer.writerow(['Location', self.current_location])
                writer.writerow(['Target Date', self.current_date])
                writer.writerow([])
            
                # Weather Parameters
                writer.writerow(['Weather Parameters'])
                writer.writerow(['Parameter', 'Value', 'Lower Bound', 'Upper Bound', 'Unit'])
            
                if self.analysis_results and 'predictions' in self.analysis_results:
                    param_names = {
                        'T2M': ('Temperature', '°C'),
                        'PRECTOTCORR': ('Rainfall', 'mm'),
                        'WS2M': ('Wind Speed', 'm/s'),
                        'PM25': ('PM2.5', 'μg/m³'),
                        'RH2M': ('Humidity', '%'),
                        'SNOWFALL': ('Snowfall', 'mm'),
                        'SNOW_DEPTH': ('Snow Depth', 'cm'),
                        'WAVE_HEIGHT': ('Wave Height', 'm'),
                        'OCEAN_CURRENT': ('Ocean Current', 'm/s'),
                        'SWELL_PERIOD': ('Swell Period', 's')
                    }
                
                    for param, data in self.analysis_results['predictions'].items():
                        name, unit = param_names.get(param, (param, ''))
                        writer.writerow([
                            name,
                            f"{data['prediction']:.2f}",
                            f"{data.get('lower_bound', 0):.2f}",
                            f"{data.get('upper_bound', 0):.2f}",
                            unit
                        ])
            
                writer.writerow([])
            
                # Recommendations
                writer.writerow(['Weather Recommendations'])
                if self.analysis_results and 'recommendations' in self.analysis_results:
                    for rec in self.analysis_results['recommendations']:
                        writer.writerow([rec])
            
                writer.writerow([])
            
                # Activity Recommendations
                writer.writerow(['Activity Recommendations'])
                if self.analysis_results and 'activity_recommendations' in self.analysis_results:
                    for activity in self.analysis_results['activity_recommendations']:
                        writer.writerow([activity])
        
            message = f"CSV file saved:\n{filename}\n\nLocation: Downloads folder"
            self.show_success(message, filepath)
        
        except Exception as e:
            self.show_error(f"Error saving CSV:\n{str(e)}")

    def download_json(self):
        """ดาวน์โหลดข้อมูลเป็น JSON"""
    
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"weather_data_{self.current_location.replace(', ', '_')}_{timestamp}.json"
        
            downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
            filepath = os.path.join(downloads_path, filename)
        
            # รวบรวมข้อมูลทั้งหมด
            export_data = {
                "report_info": {
                    "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "location": self.current_location,
                    "target_date": self.current_date
                },
                "predictions": {},
                "recommendations": self.analysis_results.get('recommendations', []),
                "activity_recommendations": self.analysis_results.get('activity_recommendations', []),
                "range_data": self.analysis_results.get('range_data', {})
            }
        
            # เพิ่มข้อมูล predictions
            if self.analysis_results and 'predictions' in self.analysis_results:
                for param, data in self.analysis_results['predictions'].items():
                    export_data['predictions'][param] = {
                        'value': round(data['prediction'], 2),
                        'lower_bound': round(data.get('lower_bound', 0), 2),
                        'upper_bound': round(data.get('upper_bound', 0), 2)
                    }
        
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        
            message = f"JSON file saved:\n{filename}\n\nLocation: Downloads folder"
            self.show_success(message, filepath)
        
        except Exception as e:
            self.show_error(f"Error saving JSON:\n{str(e)}")

    def download_pdf(self):
        """ดาวน์โหลด PDF พร้อมกราฟ"""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas as pdf_canvas
            from reportlab.lib.units import cm
            from reportlab.lib.utils import ImageReader
        
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"weather_report_{self.current_location.replace(', ', '_')}_{timestamp}.pdf"
        
            downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
            filepath = os.path.join(downloads_path, filename)
        
            c = pdf_canvas.Canvas(filepath, pagesize=A4)
            width, height = A4
        
            # === Page 1: Summary ===
            c.setFont("Helvetica-Bold", 24)
            c.drawString(2*cm, height - 2*cm, "Weather Forecast Report")
        
            c.setFont("Helvetica", 12)
            c.drawString(2*cm, height - 3*cm, f"Location: {self.current_location}")
            c.drawString(2*cm, height - 3.5*cm, f"Target Date: {self.current_date}")
            c.drawString(2*cm, height - 4*cm, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
            y = height - 5.5*cm
        
            # Weather Parameters
            c.setFont("Helvetica-Bold", 16)
            c.drawString(2*cm, y, "Weather Parameters")
            y -= 1*cm
        
            param_names = {
                'T2M': ('Temperature', '°C'),
                'PRECTOTCORR': ('Rainfall', 'mm'),
                'WS2M': ('Wind Speed', 'm/s'),
                'PM25': ('PM2.5', 'μg/m³'),
                'RH2M': ('Humidity', '%'),
                'SNOWFALL': ('Snowfall', 'mm'),
                'SNOW_DEPTH': ('Snow Depth', 'cm'),
                'WAVE_HEIGHT': ('Wave Height', 'm'),
                'OCEAN_CURRENT': ('Ocean Current', 'm/s'),
                'SWELL_PERIOD': ('Swell Period', 's')
            }
        
            if self.analysis_results and 'predictions' in self.analysis_results:
                c.setFont("Helvetica", 11)
                for param, data in self.analysis_results['predictions'].items():
                    name, unit = param_names.get(param, (param, ''))
                    value = data['prediction']
                
                    c.drawString(2.5*cm, y, f"{name}: {value:.1f} {unit}")
                    y -= 0.6*cm
                
                    if y < 3*cm:
                        c.showPage()
                        y = height - 2*cm
                        c.setFont("Helvetica", 11)
        
            # Recommendations
            y -= 1*cm
            c.setFont("Helvetica-Bold", 16)
            c.drawString(2*cm, y, "Recommendations")
            y -= 0.8*cm
        
            c.setFont("Helvetica", 10)
            for rec in self.analysis_results.get('recommendations', []):
                c.drawString(2.5*cm, y, f"• {rec[:75]}")
                y -= 0.5*cm
                if y < 3*cm:
                    c.showPage()
                    y = height - 2*cm
                    c.setFont("Helvetica", 10)
        
            # Activity Recommendations
            y -= 1*cm
            c.setFont("Helvetica-Bold", 16)
            c.drawString(2*cm, y, "Activity Recommendations")
            y -= 0.8*cm
        
            c.setFont("Helvetica", 10)
            for activity in self.analysis_results.get('activity_recommendations', []):
                c.drawString(2.5*cm, y, activity[:75])
                y -= 0.5*cm
                if y < 3*cm:
                    c.showPage()
                    y = height - 2*cm
                    c.setFont("Helvetica", 10)
        
            # === Page 2: Graph ===
            c.showPage()
            c.setFont("Helvetica-Bold", 18)
            c.drawString(2*cm, height - 2*cm, "Weather Forecast Graph")
        
            # ใส่กราฟ
            if hasattr(self, 'temp_graph_path') and self.temp_graph_path and os.path.exists(self.temp_graph_path):
                try:
                    img = ImageReader(self.temp_graph_path)
                    # วางกราฟตรงกลาง ขนาดเหมาะสม
                    c.drawImage(img, 1.5*cm, height - 18*cm, width=18*cm, height=12*cm, preserveAspectRatio=True)
                except Exception as e:
                    c.setFont("Helvetica", 10)
                    c.drawString(2*cm, height - 4*cm, f"Graph unavailable: {str(e)}")
            else:
                c.setFont("Helvetica", 10)
                c.drawString(2*cm, height - 4*cm, "Graph data not available")
        
            # Footer
            c.setFont("Helvetica-Oblique", 9)
            c.drawString(2*cm, 1.5*cm, "Generated by WhatWeather - Weather Prediction System")
        
            c.save()
        
            # ลบไฟล์กราฟชั่วคราว
            if hasattr(self, 'temp_graph_path') and self.temp_graph_path and os.path.exists(self.temp_graph_path):
                try:
                    os.remove(self.temp_graph_path)
                except:
                    pass
        
            message = f"PDF saved:\n{filename}\n\nLocation: Downloads folder"
            self.show_success(message, filepath)
        
        except ImportError:
            self.show_error("Please install:\npip install reportlab")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.show_error(f"Error creating PDF:\n{str(e)}")
    
    def show_success(self, message, filepath=None):
        content = BoxLayout(orientation='vertical', padding=20, spacing=15)
        
        success_label = Label(
            text="SUCCESS",
            font_size='48sp',
            size_hint_y=None,
            height=60,
            color=(0.2, 0.7, 0.3, 1),
            bold=True
        )
        content.add_widget(success_label)
        
        main_label = Label(
            text="Download Complete",
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
    def on_mouse_pos(self, window, pos):
        """เพิ่ม hover effect สำหรับปุ่ม"""
        # หา widget ที่เมาส์อยู่เหนือ
        widgets = window.children[0].walk(restrict=True)
        for widget in widgets:
            if isinstance(widget, Button) and widget.collide_point(*pos):
                # Hover state
                if hasattr(widget, 'canvas'):
                    # เปลี่ยนสีเมื่อ hover
                    if 'Download' in widget.text or 'Prediction' in widget.text or 'GET STARTED' in widget.text:
                        # ปุ่มสีน้ำเงิน - สีเข้มขึ้นเมื่อ hover
                        for instruction in widget.canvas.before.children:
                            if isinstance(instruction, Color):
                                if widget.state == 'normal':
                                    instruction.rgba = [0.18, 0.55, 0.95, 1]  # สีน้ำเงินเข้มขึ้น
                                break
                    elif 'New Predict' in widget.text:
                        # ปุ่มสีเทา - สีเข้มขึ้นเมื่อ hover
                        for instruction in widget.canvas.before.children:
                            if isinstance(instruction, Color):
                                if widget.state == 'normal':
                                    instruction.rgba = [0.8, 0.8, 0.82, 1]
                                break
            elif isinstance(widget, Button):
                # Reset สี normal state
                if 'Download' in widget.text or 'Prediction' in widget.text or 'GET STARTED' in widget.text:
                    for instruction in widget.canvas.before.children:
                        if isinstance(instruction, Color):
                            if widget.state == 'normal':
                                instruction.rgba = [0.25, 0.6, 1, 1]
                            break
                elif 'New Predict' in widget.text:
                    for instruction in widget.canvas.before.children:
                        if isinstance(instruction, Color):
                            if widget.state == 'normal':
                                instruction.rgba = [0.9, 0.9, 0.92, 1]
                            break

if __name__ == '__main__':
    WhatWeather().run()