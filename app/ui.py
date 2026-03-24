from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle, Ellipse, InstructionGroup
from kivy.core.window import Window
from kivy.core.text import LabelBase
from kivy.config import Config
import random
import threading
from app.backend import VoiceAssistant

# Fullscreen & dark blue background
Window.fullscreen = True


class KidsUI(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # --- Background ---
        with self.canvas.before:
            Color(0.05, 0.05, 0.2, 1)  # Dark blue
            self.bg_rect = Rectangle(pos=(0, 0), size=Window.size)
        Window.bind(size=self.update_bg)

        # --- Floating bubbles ---
        self.bubbles = [{
            "x": random.randint(0, Window.width),
            "y": random.randint(0, Window.height),
            "r": random.randint(10, 25),
            "speed": random.uniform(0.2, 0.8)
        } for _ in range(15)]
        self.bubble_groups = []  # Store InstructionGroups instead of individual instructions

        # --- Mode label ---
        self.mode_label = Label(
            text="Listening...",
            font_size=50,
            color=(1, 1, 1, 1),
            pos_hint={"center_x": 0.5, "center_y": 0.9}
        )
        self.add_widget(self.mode_label)
        self.mode = "listen"

        # Question box (background for the question)
        self.question_frame = FloatLayout(size_hint=(None, None), size=(900, 100))
        self.question_frame.pos_hint = {"center_x": 0.5, "y": 0.05}

        # White background for the question box
        with self.question_frame.canvas.before:
            Color(1, 1, 1, 1)  # White background
            self.question_bg = Rectangle(pos=self.question_frame.pos, size=self.question_frame.size)

        # Update rectangle when resized
        self.question_frame.bind(pos=self.update_question_bg, size=self.update_question_bg)

        # Question label (with large font size and centered text)
        self.question_label = Label(
            text="",
            font_size=45, 
            color=(0, 0, 0, 1),  
            size_hint=(None, None),
            size=(900, 100),  
            halign="center",  
            valign="middle",  
            text_size=(900, 100),  
        )

        self.question_label.center_x = self.question_frame.center_x
        self.question_label.center_y = self.question_frame.center_y

        self.question_frame.add_widget(self.question_label)
        self.add_widget(self.question_frame)
        
        self.question_frame.bind(pos=self.update_question_bg, size=self.update_question_bg)

        # --- Exit button ---
        self.exit_btn = Button(
            text="X",
            size_hint=(None, None),
            size=(50, 50),
            pos_hint={"right": 0.98, "top": 0.98},
            background_color=(1, 1, 1, 1),
            color=(0, 0, 0, 1)
        )
        self.exit_btn.bind(on_release=self.close_app)
        self.add_widget(self.exit_btn)

        # --- Language Selection Bar ---
        self.lang_bar = BoxLayout(
            orientation='horizontal',
            size_hint=(0.9, 0.08),
            pos_hint={"center_x": 0.5, "center_y": 0.8},
            spacing=10
        )
        languages = [
            "Hindi", "Kannada", "Tamil", "Malayalam", 
            "South English", "French", "Spanish", "English"
        ]
        for lang in languages:
            btn = Button(
                text=lang,
                background_color=(0.2, 0.6, 1, 1),
                color=(1, 1, 1, 1),
                font_size=20,
                bold=True
            )
            # Use a lambda that captures the current lang string
            btn.bind(on_release=lambda instance, l=lang: self.change_lang(l))
            self.lang_bar.add_widget(btn)
        self.add_widget(self.lang_bar)

        # --- Backend ---
        self.assistant = VoiceAssistant(
            on_listen=self.set_listen,
            on_speak=self.set_speak,
            on_question=self.display_question
        )
        threading.Thread(target=self.start_voice_engine, daemon=True).start()

        # --- Animation ---
        Clock.schedule_interval(self.animate, 1/30)

    # --- Update background / question box ---
    def update_bg(self, *args):
        self.bg_rect.size = Window.size

    def update_question_bg(self, instance, value):
        self.question_bg.pos = instance.pos
        self.question_bg.size = instance.size

        self.question_label.center_x = instance.center_x
        self.question_label.center_y = instance.center_y

    # --- Backend callbacks ---
    def start_voice_engine(self):
        self.assistant.run()

    def set_listen(self):
        self.mode = "listen"
        self.mode_label.text = "Listening..."
        self.question_label.text = ""  

    def set_speak(self):
        self.mode = "speak"
        self.mode_label.text = "Speaking..."

    def display_question(self, text):
        self.question_label.text = text

    def change_lang(self, lang_name):
        """Update the assistant's language."""
        if hasattr(self, "assistant"):
            self.assistant.set_language(lang_name)

    # --- Close app ---
    def close_app(self, *args):
        if hasattr(self, "assistant"):
            self.assistant.running = False
        App.get_running_app().stop()

    # --- Animation loop ---
    def animate(self, dt):
        for grp in self.bubble_groups:
            try:
                self.canvas.remove(grp)
            except Exception:
                pass
        self.bubble_groups = []

        # Draw new bubbles
        for bubble in self.bubbles:
            grp = InstructionGroup()
            grp.add(Color(0.63, 0.88, 1, 0.8))
            grp.add(Ellipse(pos=(bubble["x"] - bubble["r"], bubble["y"] - bubble["r"]),
                            size=(bubble["r"]*2, bubble["r"]*2)))
            self.canvas.add(grp)
            self.bubble_groups.append(grp)

            bubble["y"] -= bubble["speed"]
            if bubble["y"] + bubble["r"] < 0:
                bubble["y"] = Window.height + bubble["r"]
                bubble["x"] = random.randint(0, Window.width)


class KidsApp(App):
    def build(self):
        return KidsUI()

if __name__ == "__main__":
    KidsApp().run()
