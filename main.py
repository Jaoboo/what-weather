from kivy.config import Config

# กำหนดค่าก่อน import widget
Config.set('graphics', 'fullscreen', '0')
Config.set('graphics', 'resizable', '0')
Config.set('graphics', 'width', '1000')
Config.set('graphics', 'height', '800')
Config.set('graphics', 'borderless', '0')

from kivy.app import App
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.image import AsyncImage
from kivy.lang import Builder
from kivy.graphics import Color, Rectangle
from kivy.properties import ListProperty, StringProperty
from kivy.graphics import RoundedRectangle
import re
import csv
from datetime import datetime
import os

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
    
    def on_enter(self):
        self.ids.form_frame.ids.form_content.ids.location_input.text = ""
        self.ids.form_frame.ids.form_content.ids.date_input.text = ""
        self.selected_params = []
        # Reset checkboxes
        self.ids.form_frame.ids.form_content.ids.cb_humidity.active = False
        self.ids.form_frame.ids.form_content.ids.cb_pressure.active = False
        self.ids.form_frame.ids.form_content.ids.cb_cloud.active = False
        self.ids.form_frame.ids.form_content.ids.cb_solar.active = False
    
    def on_checkbox_change(self, checkbox, param_name):
        if checkbox.active:
            # ตรวจสอบว่าเลือกครบ 2 ตัวแล้วหรือยัง
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
        
        date_pattern = r'^\d{4}-\d{2}-\d{2}$'
        if not re.match(date_pattern, date_input):
            self.show_warning("Invalid date format!\nPlease use YYYY-MM-DD\nExample: 2025-12-31")
            return

        # ไปหน้า Result โดยตรง
        result_screen = self.manager.get_screen("result")
        result_screen.update_result(location_name, date_input, self.selected_params)
        self.manager.current = "result"

class ResultScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_location = ""
        self.current_date = ""
        self.selected_params = []
        
        # Default parameter labels
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
        # เก็บข้อมูลไว้ใช้ตอน export
        self.current_location = location
        self.current_date = date
        self.selected_params = selected_params
        
        # แสดงแค่ข้อมูลพื้นฐาน
        self.ids.location_value.text = location
        self.ids.date_value.text = date
        
        # เคลียร์ box ต่างๆ
        self.ids.summary_box.clear_widgets()
        self.ids.outdoor_activities_box.clear_widgets()
        self.ids.graph_box.clear_widgets()
        
        # ใส่ placeholder
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
        
        # อัปเดต Parameter Cards - Default 4 cards + 2 selected
        # Card 1: Temperature (Default)
        self.ids.param1_value.text = "- °C"
        self.ids.param1_status.text = "Normal"
        
        # Card 2: Rainfall (Default)
        self.ids.param2_value.text = "- mm"
        self.ids.param2_status.text = "Normal"
        
        # Card 3: Wind Speed (Default)
        self.ids.param3_value.text = "- m/s"
        self.ids.param3_status.text = "Normal"
        
        # Card 4: PM2.5 (Default)
        self.ids.param4_value.text = "- μg/m³"
        self.ids.param4_status.text = "Good"
        
        # Card 5 & 6: Selected parameters
        if len(selected_params) > 0:
            param_name = self.param_labels.get(selected_params[0], selected_params[0])
            self.ids.param5_name.text = param_name
            self.ids.param5_value.text = "-"
            self.ids.param5_status.text = "Normal"
        else:
            self.ids.param5_name.text = "-"
            self.ids.param5_value.text = "-"
            self.ids.param5_status.text = "-"
        
        if len(selected_params) > 1:
            param_name = self.param_labels.get(selected_params[1], selected_params[1])
            self.ids.param6_name.text = param_name
            self.ids.param6_value.text = "-"
            self.ids.param6_status.text = "Normal"
        else:
            self.ids.param6_name.text = "-"
            self.ids.param6_value.text = "-"
            self.ids.param6_status.text = "-"
    
    def download_csv(self):
        """ดาวน์โหลดข้อมูลเป็น CSV"""
        try:
            # สร้างชื่อไฟล์
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"weather_data_{self.current_location}_{timestamp}.csv"
            
            # สร้างโฟลเดอร์ Downloads ถ้ายังไม่มี
            downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
            if not os.path.exists(downloads_path):
                downloads_path = os.path.expanduser("~")
            
            filepath = os.path.join(downloads_path, filename)
            
            # เขียนข้อมูลลง CSV
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Header
                writer.writerow(['Weather Data Report'])
                writer.writerow(['Generated:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                writer.writerow([])
                
                # Basic Info
                writer.writerow(['Location:', self.current_location])
                writer.writerow(['Target Date:', self.current_date])
                writer.writerow([])
                
                # Predictions (ตัวอย่าง - คุณสามารถแก้ไขให้เหมาะกับข้อมูลจริงของคุณ)
                writer.writerow(['Parameter', 'Prediction', 'Unit'])
                writer.writerow(['Temperature', 'N/A', '°C'])
                writer.writerow(['Rainfall', 'N/A', 'mm'])
                writer.writerow(['Wind Speed', 'N/A', 'm/s'])
                writer.writerow(['PM2.5', 'N/A', 'μg/m³'])
                writer.writerow([])
                
                # Summary
                writer.writerow(['Summary'])
                writer.writerow(['Data will be available after analysis'])
            
            # แสดง popup สำเร็จ
            self.show_success(f"File saved successfully!\n\n{filepath}")
            
        except Exception as e:
            self.show_error(f"Error saving file:\n{str(e)}")
    
    def show_success(self, message):
        content = BoxLayout(orientation='vertical', padding=20)
        content.add_widget(Label(
            text=message,
            color=(0.2, 0.7, 0.3, 1),
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
            title='Success',
            content=content,
            size_hint=(0.7, 0.4),
            background='',
            separator_color=(0.2, 0.7, 0.3, 1),
            title_color=(0.2, 0.7, 0.3, 1)
        )
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
    
    def show_app_info(self):
        """แสดงข้อมูลเกี่ยวกับแอป"""
        info_text = """[b]WHATWEATHER[/b]
        
Weather Prediction System

[b]About:[/b]
This application provides weather forecasting and analysis for any location worldwide.

[b]Features:[/b]
• Temperature prediction
• Rainfall forecast
• Wind speed analysis
• PM2.5 air quality monitoring
• Historical trend visualization
• Activity recommendations

[b]How to use:[/b]
1. Enter a location (in English)
2. Select a target date (YYYY-MM-DD)
3. Click START to get predictions
4. Download results as CSV

[b]Version:[/b] 1.0.0
[b]Developed by:[/b] Your Team Name
        """
        
        content = BoxLayout(orientation='vertical', padding=20, spacing=10)
        
        scroll = ScrollView()
        info_label = Label(
            text=info_text,
            markup=True,
            color=(0.2, 0.2, 0.2, 1),
            font_size='13sp',
            size_hint_y=None,
            halign='left',
            valign='top'
        )
        info_label.bind(texture_size=info_label.setter('size'))
        info_label.bind(size=lambda obj, size: setattr(obj, 'text_size', (size[0] - 20, None)))
        scroll.add_widget(info_label)
        content.add_widget(scroll)
        
        close_btn = Button(
            text='Close',
            size_hint=(None, None),
            size=(100, 40),
            pos_hint={'center_x': 0.5}
        )
        content.add_widget(close_btn)
        
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
            title='About WHATWEATHER',
            content=content,
            size_hint=(0.7, 0.8),
            background='',
            separator_color=(0.2, 0.6, 1, 1),
            title_color=(0.2, 0.6, 1, 1)
        )
        close_btn.bind(on_release=popup.dismiss)
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