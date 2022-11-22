from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
from camera4kivy import Preview
from kivy.properties import ObjectProperty

class Screen3(Screen):
    cam = ObjectProperty(None)
    
    def __init__(self, **kwargs):
        Builder.load_file('Screens/screen3.kv')
        super(Screen3, self).__init__(**kwargs)

    def on_enter(self):
        self.cam.connect_camera(filepath_callback= self.capture_path)

    def on_pre_leave(self):
        self.cam.disconnect_camera()
    
    def capture_path(self,file_path):
        pass

class S3Layout(BoxLayout):

    def __init__(self, **kwargs):
        Builder.load_file('Screens/s3layout.kv')
        super(S3Layout, self).__init__(**kwargs)

    def select_camera(self, facing):
        self.ids.preview.select_camera(facing)

    def capture(self):
        self.ids.preview.capture_photo(location = 'shared', subdir = 'images', name = 'test.jpg')

    