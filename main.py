from kivy.app import App
from kivy.uix.screenmanager import ScreenManager
from kivy.core.window import Window
from Screens.screen1 import Screen1
from Screens.screen2 import Screen2
from Screens.screen3 import Screen3
from Screens.screen4 import Screen4

Window.clearcolor = (16/255, 34/255, 48/255, 1)

class DigitRecognitionApp(App):

    def build(self):
        self.enable_swipe = False
        self.sm = ScreenManager()
        self.screens = [Screen1(name='1'),
                        Screen2(name='2'),
                        Screen3(name='3'),
                        Screen4(name='4')]
        for s in self.screens:
            self.sm.add_widget(s)
        return self.sm

if __name__ == '__main__':
    DigitRecognitionApp().run()