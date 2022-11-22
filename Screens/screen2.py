from kivy.lang import Builder
from kivy.uix.screenmanager import Screen

class Screen2(Screen):

    def __init__(self, **kwargs):
        Builder.load_file('Screens/screen2.kv')
        super(Screen2, self).__init__(**kwargs)