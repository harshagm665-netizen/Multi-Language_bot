from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle, Ellipse, InstructionGroup
from kivy.core.window import Window
import random
import threading
from app.backend import VoiceAssistant

# Fullscreen & dark blue background settings
Window.fullscreen = True

class WelcomeScreen(Screen):
    def __init__(self, on_start_callback, on_lang_change, **kwargs):
        super().__init__(**kwargs)
        self.on_start_callback = on_start_callback
        self.on_lang_change = on_lang_change
        self.selected_lang = "English"

        with self.canvas.before:
            Color(0.05, 0.05, 0.2, 1)  # Dark blue
            self.rect = Rectangle(pos=(0,0), size=Window.size)
        Window.bind(size=self.update_rect)

        # Main Layout
        layout = FloatLayout()

        # Title
        layout.add_widget(Label(
            text="Novabot: Choose Your Language",
            font_size='40sp',
            bold=True,
            pos_hint={"center_x": 0.5, "center_y": 0.85}
        ))

        # Language Grid
        grid = GridLayout(cols=4, spacing=15, size_hint=(0.8, 0.4), pos_hint={"center_x": 0.5, "center_y": 0.5})
        languages = [
            "Hindi", "Kannada", "Tamil", "Malayalam", 
            "French", "Spanish", "English", "South English"
        ]
        
        self.lang_buttons = {}
        for lang in languages:
            btn = Button(
                text=lang,
                background_normal='',
                background_color=(0.1, 0.4, 0.8, 1),
                font_size='20sp',
                bold=True
            )
            btn.bind(on_release=lambda instance, l=lang: self.select_language(l))
            grid.add_widget(btn)
            self.lang_buttons[lang] = btn

        layout.add_widget(grid)

        # Start Button
        self.start_btn = Button(
            text="START BOT",
            size_hint=(0.3, 0.12),
            pos_hint={"center_x": 0.5, "center_y": 0.2},
            background_normal='',
            background_color=(0, 0.8, 0.2, 1),
            font_size='25sp',
            bold=True
        )
        self.start_btn.bind(on_release=lambda x: self.on_start_callback())
        layout.add_widget(self.start_btn)

        # Exit Button
        exit_btn = Button(
            text="EXIT",
            size_hint=(None, None),
            size=(100, 50),
            pos_hint={"right": 0.98, "top": 0.98},
            background_color=(0.8, 0.1, 0.1, 1)
        )
        exit_btn.bind(on_release=lambda x: App.get_running_app().stop())
        layout.add_widget(exit_btn)

        self.add_widget(layout)

    def update_rect(self, *args):
        self.rect.size = Window.size

    def select_language(self, lang):
        self.selected_lang = lang
        # Visual feedback
        for l, btn in self.lang_buttons.items():
            btn.background_color = (0.1, 0.4, 0.8, 1) # Reset
        self.lang_buttons[lang].background_color = (1, 0.6, 0, 1) # Highlight
        
        # Notify backend
        self.on_lang_change(lang)

class MainAssistantScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = FloatLayout()

        # --- Background ---
        with self.canvas.before:
            Color(0.05, 0.05, 0.2, 1)
            self.bg_rect = Rectangle(pos=(0, 0), size=Window.size)
        Window.bind(size=self.update_bg)

        # --- Floating bubbles ---
        self.bubbles = [{
            "x": random.randint(0, Window.width),
            "y": random.randint(0, Window.height),
            "r": random.randint(10, 25),
            "speed": random.uniform(0.2, 0.8)
        } for _ in range(15)]
        self.bubble_groups = []

        # --- Labels ---
        self.mode_label = Label(text="Initializing...", font_size=50, pos_hint={"center_x": 0.5, "center_y": 0.8})
        self.layout.add_widget(self.mode_label)

        # Question Frame
        self.question_frame = FloatLayout(size_hint=(0.85, 0.15), pos_hint={"center_x": 0.5, "y": 0.08})
        with self.question_frame.canvas.before:
            Color(1, 1, 1, 1)
            self.q_bg = Rectangle(pos=self.question_frame.pos, size=self.question_frame.size)
        self.question_frame.bind(pos=self.update_q_bg, size=self.update_q_bg)

        self.question_label = Label(
            text="", font_size=40, color=(0,0,0,1), halign="center", valign="middle",
            size_hint=(1, 1), text_size=(800, 150)
        )
        self.question_frame.add_widget(self.question_label)
        self.layout.add_widget(self.question_frame)

        # --- Navigation Buttons ---
        nav_bar = BoxLayout(size_hint=(1, 0.08), pos_hint={"top": 1}, padding=10, spacing=Window.width - 250)
        
        back_btn = Button(text="BACK", size_hint=(None, 1), width=100, background_color=(0.5, 0.5, 0.5, 1))
        back_btn.bind(on_release=self.go_back)
        
        exit_btn = Button(text="EXIT", size_hint=(None, 1), width=100, background_color=(0.8, 0.1, 0.1, 1))
        exit_btn.bind(on_release=lambda x: App.get_running_app().stop())

        nav_bar.add_widget(back_btn)
        nav_bar.add_widget(exit_btn)
        self.layout.add_widget(nav_bar)

        self.add_widget(self.layout)
        Clock.schedule_interval(self.animate, 1/30)

    def update_bg(self, *args):
        self.bg_rect.size = Window.size

    def update_q_bg(self, instance, value):
        self.q_bg.pos = instance.pos
        self.q_bg.size = instance.size

    def go_back(self, *args):
        self.manager.transition = FadeTransition()
        self.manager.current = 'welcome'

    def animate(self, dt):
        for grp in self.bubble_groups:
            try: self.canvas.remove(grp)
            except: pass
        self.bubble_groups = []
        for i, b in enumerate(self.bubbles):
            grp = InstructionGroup()
            grp.add(Color(0.63, 0.88, 1, 0.6))
            grp.add(Ellipse(pos=(b["x"]-b["r"], b["y"]-b["r"]), size=(b["r"]*2, b["r"]*2)))
            self.canvas.add(grp)
            self.bubble_groups.append(grp)
            b["y"] -= b["speed"]
            if b["y"] + b["r"] < 0:
                b["y"] = Window.height + b["r"]
                b["x"] = random.randint(0, Window.width)

class NovabotUI(ScreenManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transition = FadeTransition()
        
        # Initialize Backend
        self.assistant = VoiceAssistant(
            on_listen=self.set_listen,
            on_speak=self.set_speak,
            on_question=self.display_question
        )

        # Create Screens
        self.welcome = WelcomeScreen(name='welcome', 
                                   on_start_callback=self.start_interaction,
                                   on_lang_change=self.change_language)
        self.main = MainAssistantScreen(name='interaction')

        self.add_widget(self.welcome)
        self.add_widget(self.main)

        # Thread for engine
        threading.Thread(target=self.assistant.run, daemon=True).start()

    def start_interaction(self):
        self.current = 'interaction'

    def change_language(self, lang):
        self.assistant.set_language(lang)

    def set_listen(self):
        self.main.mode_label.text = "Listening..."
        self.main.question_label.text = ""

    def set_speak(self):
        self.main.mode_label.text = "Speaking..."

    def display_question(self, text):
        self.main.question_label.text = text

class KidsApp(App):
    def build(self):
        return NovabotUI()

if __name__ == "__main__":
    KidsApp().run()
