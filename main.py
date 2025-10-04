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
from kivy.graphics import Color, Rectangle
from kivy.properties import ListProperty, StringProperty
from kivy.graphics import RoundedRectangle
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
        self.active_marker = None
        self.map_initialized = False
        self.red_dot = None  # จุดแดงอ้างอิง
    
    def on_enter(self):
        self.ids.form_frame.ids.form_content.ids.location_input.text = ""
        self.ids.form_frame.ids.form_content.ids.date_input.text = ""
        self.selected_params = []
    
        self.ids.form_frame.ids.form_content.ids.cb_humidity.active = False
        self.ids.form_frame.ids.form_content.ids.cb_pressure.active = False
        self.ids.form_frame.ids.form_content.ids.cb_cloud.active = False
        self.ids.form_frame.ids.form_content.ids.cb_solar.active = False
    
        if not self.map_initialized:
            self.initialize_map()
        else:
            # Reset แผนที่
            if self.active_marker:
                try:
                    self.mapview.remove_marker(self.active_marker)
                except:
                    pass
                self.active_marker = None
            
            # ลบจุดแดงถ้ามี
            if self.red_dot:
                try:
                    self.mapview.remove_widget(self.red_dot)
                except:
                    pass
                self.red_dot = None
        
        # Reset Label
        if hasattr(self, 'coord_label'):
            self.coord_label.text = 'Lat: -, Lon: -'
        
            self.mapview.center_on(20, 0)
            self.mapview.zoom = 2
    
    def initialize_map(self):
        try:
            map_container = self.ids.map_container
            map_container.clear_widgets()
        
            self.mapview = MapView(
                zoom=2,
                lat=20,
                lon=0,
                snap_to_zoom=False, 
                double_tap_zoom=False, 
                map_source='osm'
            )
            self.mapview.bind(on_touch_down=self.on_map_touch)
            self.mapview.bind(on_touch_up=self.on_map_touch_up)
        
            map_container.add_widget(self.mapview)
        
            # ไม่มีหมุดตอนเริ่มต้น
            self.active_marker = None
            self.red_dot = None
        
            # สร้าง Label แสดง Lat/Lon
            self.coord_label = Label(
                text='Lat: -, Lon: -',
                size_hint=(None, None),
                size=(150, 30),
                pos_hint={'right': 0.98, 'top': 0.98},
                color=(0.2, 0.2, 0.2, 1),
                font_size='12sp',
                bold=True
            )
        
            # เพิ่ม canvas พื้นหลังให้ Label
            with self.coord_label.canvas.before:
                from kivy.graphics import Color, RoundedRectangle
                Color(1, 1, 1, 0.9)
                self.coord_rect = RoundedRectangle(
                    pos=self.coord_label.pos,
                    size=self.coord_label.size,
                    radius=[8]
                )
        
            self.coord_label.bind(pos=self._update_coord_rect)
            self.coord_label.bind(size=self._update_coord_rect)
        
            map_container.add_widget(self.coord_label)
        
            self.map_initialized = True
        except Exception as e:
            print(f"Error initializing map: {e}")

    def _update_coord_rect(self, *args):
        """อัปเดตตำแหน่งและขนาดของพื้นหลัง Label"""
        self.coord_rect.pos = self.coord_label.pos
        self.coord_rect.size = self.coord_label.size
    
    def on_map_touch(self, instance, touch):
        """ดักจับ touch down event บนแผนที่"""
        # ตรวจสอบว่าคลิกภายในพื้นที่แผนที่
        if not self.mapview.collide_point(*touch.pos):
            return False
        
        # ตรวจสอบว่าคลิกที่หมุดหรือไม่
        if self.active_marker and self.active_marker.collide_point(*touch.pos):
            # ถ้า double-click ที่หมุด ให้สามารถย้ายหมุดได้
            if touch.is_double_tap:
                touch.ud['moving_marker'] = True
                return True
            else:
                # คลิกครั้งเดียว ให้ center แผนที่ไปที่หมุด
                self.mapview.center_on(self.active_marker.lat, self.active_marker.lon)
                return True
        
        # Double-click บนแผนที่ -> ปักหมุดใหม่
        if touch.is_double_tap:
            lat, lon = self.mapview.get_latlon_at(*touch.pos)
            self._place_marker(lat, lon)
            return True
        
        # คลิกขวา -> วางจุดแดงอ้างอิง (ป้องกันแผนที่เคลื่อนไหว)
        if touch.button == 'right':
            self._place_red_dot(touch.pos)
            return True
        
        # Mouse scroll -> ซูม
        if touch.button == 'scrolldown':
            self.mapview.zoom = max(2, self.mapview.zoom - 1)
            return True
        elif touch.button == 'scrollup':
            self.mapview.zoom = min(20, self.mapview.zoom + 1)
            return True
        
        # คลิกซ้ายปกติ -> ให้แผนที่จัดการเอง (pan)
        return False
    
    def on_map_touch_up(self, instance, touch):
        """ดักจับ touch up event สำหรับการย้ายหมุด"""
        if touch.ud.get('moving_marker'):
            # ย้ายหมุดไปตำแหน่งใหม่
            lat, lon = self.mapview.get_latlon_at(*touch.pos)
            self._place_marker(lat, lon)
            return True
        return False
    
    def _place_red_dot(self, pos):
        """วางจุดแดงอ้างอิงบนแผนที่"""
        from kivy.uix.widget import Widget
        from kivy.graphics import Color, Ellipse
        
        # ลบจุดแดงเก่า
        if self.red_dot:
            try:
                self.mapview.remove_widget(self.red_dot)
            except:
                pass
        
        # สร้างจุดแดงใหม่
        self.red_dot = Widget(size=(10, 10))
        with self.red_dot.canvas:
            Color(1, 0, 0, 0.8)  # สีแดง
            Ellipse(pos=(pos[0] - 5, pos[1] - 5), size=(10, 10))
        
        self.red_dot.pos = (pos[0] - 5, pos[1] - 5)
        self.mapview.add_widget(self.red_dot)
    
    def _place_marker(self, lat, lon):
        """วางหมุดใหม่และทำ reverse geocoding"""
        # ลบหมุดเก่า
        self._remove_active_marker()
        
        # ลบจุดแดงถ้ามี
        if self.red_dot:
            try:
                self.mapview.remove_widget(self.red_dot)
            except:
                pass
            self.red_dot = None
        
        # สร้างหมุดใหม่
        self.active_marker = MapMarkerPopup(
            lat=lat, 
            lon=lon
        )
        self.mapview.add_marker(self.active_marker)
        
        # อัปเดต Label แสดง Lat/Lon
        self.coord_label.text = f'Lat: {lat:.4f}, Lon: {lon:.4f}'
        
        # ค้นหาชื่อสถานที่ในพื้นหลัง
        Thread(target=self._reverse_geocode, args=(lat, lon), daemon=True).start()

    def _remove_active_marker(self):
        """ลบหมุดที่แสดงอยู่ออกจากแผนที่"""
        if self.active_marker:
            try:
                self.mapview.remove_marker(self.active_marker)
            except Exception as e:
                print(f"Could not remove marker: {e}")
            self.active_marker = None

    def _reverse_geocode(self, lat, lon):
        """ค้นหาชื่อสถานที่จากพิกัด Lat/Lon"""
        try:
            headers = {
                'User-Agent': 'WhatWeather/1.0',
                'Accept-Language': 'en'
            }
            params = {
                'lat': lat, 
                'lon': lon, 
                'format': 'json', 
                'addressdetails': 1,
                'accept-language': 'en'
            }
            response = requests.get(
                'https://nominatim.openstreetmap.org/reverse', 
                params=params, 
                headers=headers, 
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
        
            # แยกเอาเฉพาะ city และ state/country
            address = result.get('address', {})
        
            # ลองหาชื่อเมือง
            city = (address.get('city') or 
                    address.get('town') or 
                    address.get('village') or 
                    address.get('municipality') or
                    address.get('county'))
        
            # ลองหาชื่อรัฐ/ประเทศ
            state = (address.get('state') or 
                     address.get('province') or 
                     address.get('region'))
        
            country = address.get('country')
        
            # สร้างชื่อสถานที่แบบสั้น
            location_parts = []
            if city:
                location_parts.append(city)
            if state:
                location_parts.append(state)
            elif country:  # ถ้าไม่มี state ให้ใช้ country แทน
                location_parts.append(country)
        
            display_name = ', '.join(location_parts) if location_parts else result.get('display_name', 'Unknown Location')
        
            Clock.schedule_once(lambda dt: self._update_location_input(display_name), 0)
        except Exception as e:
            print(f"Reverse geocode error: {e}")
    
    def _update_location_input(self, location_name):
        """อัปเดตช่อง location input ด้วยชื่อสถานที่"""
        self.ids.form_frame.ids.form_content.ids.location_input.text = location_name
    
    def on_location_input_change(self, text):
        """เรียกใช้เมื่อผู้ใช้พิมพ์ใน location input"""
        query = text.strip()
        if len(query) < 3:
            return
        
        Thread(target=self._search_location, args=(query,), daemon=True).start()
    
    def _search_location(self, query):
        """ค้นหาสถานที่จากชื่อ"""
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
            
                Clock.schedule_once(lambda dt: self._update_map(lat, lon), 0)
        except Exception as e:
            print(f"Search location error: {e}")
    
    def _update_map(self, lat, lon):
        """อัปเดตแผนที่และวางหมุดใหม่จากการค้นหา"""
        if not self.map_initialized:
            return
    
        # วางหมุดใหม่
        self._place_marker(lat, lon)
    
        # เลื่อนแผนที่และซูม
        self.mapview.center_on(lat, lon)
        self.mapview.zoom = 13

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
        
        date_pattern = r'^\d{4}-\d{2}-\d{2}$'
        if not re.match(date_pattern, date_input):
            self.show_warning("Invalid date format!\nPlease use YYYY-MM-DD\nExample: 2025-12-31")
            return

        result_screen = self.manager.get_screen("result")
        result_screen.update_result(location_name, date_input, self.selected_params)
        self.manager.current = "result"
class ResultScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_location = ""
        self.current_date = ""
        self.selected_params = []
        
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
    
    def update_result(self, location, date, selected_params=[]):
        self.current_location = location
        self.current_date = date
        self.selected_params = selected_params
        
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