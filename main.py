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
    def on_enter(self):
        self.ids.form_frame.ids.form_content.ids.location_input.text = ""
        self.ids.form_frame.ids.form_content.ids.date_input.text = ""
    
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
        result_screen.update_result(location_name, date_input)
        self.manager.current = "result"

class ResultScreen(Screen):
    def update_result(self, location, date):
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