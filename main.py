# main.py
from kivy.app import App
from app.ui import NovabotUI

class KidsApp(App):
    def build(self):
        # Return your UI layout
        return NovabotUI()

if __name__ == "__main__":
    KidsApp().run()
