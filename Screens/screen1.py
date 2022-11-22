from kivy.lang import Builder
from kivy.uix.screenmanager import Screen

class Screen1(Screen):

    def __init__(self, **kwargs):
        Builder.load_file('Screens/screen1.kv')
        super(Screen1, self).__init__(**kwargs)