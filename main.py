# main.py
from kivy.app import App
from app.ui import KidsUI  # Your Kivy UI class (inherits from FloatLayout)

class KidsApp(App):
    def build(self):
        # Return your UI layout
        return KidsUI()

if __name__ == "__main__":
    KidsApp().run()
