
import subprocess
import sounddevice as sd
import queue
import json
import time
import numpy as np
import os
os.environ["VOSK_LOG_LEVEL"] = "-1"
from scipy.signal import resample
import requests
from vosk import Model, KaldiRecognizer
from datetime import datetime
import re
import threading
import serial
import os
# import webrtcvad (unused, causes issues on 3.13)
from scipy.signal import resample_poly
from scipy.signal import butter, lfilter
from dotenv import load_dotenv
from groq import Groq
from bs4 import BeautifulSoup
from pytrends.request import TrendReq
import yfinance as yf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class VoiceAssistant:
    def __init__(self, on_listen, on_speak, on_question=None):
        self.on_listen = on_listen
        self.on_speak = on_speak
        self.on_question = on_question

        # --- Load API Key ---
        env_path = os.path.join(PROJECT_ROOT, ".env")
        load_dotenv(env_path)
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        
        # Manual fallback if load_dotenv fails
        if not self.api_key and os.path.exists(env_path):
            try:
                with open(env_path, "r") as f:
                    for line in f:
                        if line.strip().startswith("GROQ_API_KEY="):
                            self.api_key = line.split("=", 1)[1].strip().strip("'").strip('"')
                            break
            except Exception as e:
                print(f"⚠ Manual .env read failed: {e}")

        if not self.api_key:
            print(f"⚠ ERROR: GROQ_API_KEY missing in {env_path}")
        else:
            print(f"✔ Groq API Key loaded (starts with {self.api_key[:4]}...)")

        # --- Piper Executable Discovery ---
        self.PIPER_EXE = os.path.join(PROJECT_ROOT, "piper", "piper")
        if not os.path.exists(self.PIPER_EXE):
            # Try build folder fallback
            fallback = os.path.join(PROJECT_ROOT, "piper", "build", "piper")
            if os.path.exists(fallback):
                self.PIPER_EXE = fallback
            else:
                print(f"⚠ Warning: Piper executable not found at {self.PIPER_EXE} or {fallback}")

        # -------------------
        # Audio / model paths
        # -------------------
        self.MIC_RATE = 48000
        self.VOSK_RATE = 16000
        self.MIC_DEVICE = self.detect_usb_mic()
        self.audio_q = queue.Queue(maxsize=200)   # holds raw int16 chunks (48k)
        
        # # ---- VOSK MODEL ----
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.VOSK_MODEL_PATH = os.path.join(base_dir, "model", "vosk-model-small-en-in-0.4")

        if not os.path.exists(self.VOSK_MODEL_PATH):
            raise FileNotFoundError(f"Vosk model not found: {self.VOSK_MODEL_PATH}")

        self.model = Model(self.VOSK_MODEL_PATH)
        self.rec = KaldiRecognizer(self.model, self.VOSK_RATE)

        # # ---- OLLAMA ----
        # self.OLLAMA_URL = "http://127.0.0.1:11434/api/generate"   # safest localhost URL

        # ---- PIPER MODEL ----
        self.language = "English"
        self.language_code = "en"
        self.language_models = {
            "English": "en_US-amy-low.onnx",
            "South English": "en_GB-southern_english_female-low.onnx",
            "Hindi": "hi_IN-pratham-medium.onnx",
            "Kannada": "kn_IN-dharwad-medium.onnx",
            "Tamil": "ta_IN-roja-medium.onnx",
            "Malayalam": "ml_IN-arjun-medium.onnx",
            "French": "fr_FR-siwis-low.onnx",
            "Spanish": "es_ES-carlfm-x_low.onnx"
        }
        
        self.PIPER_MODEL = os.path.join(base_dir, "piper", "models", self.language_models[self.language])

        if not os.path.exists(self.PIPER_MODEL):
            print(f"⚠ Warning: Default Piper model missing: {self.PIPER_MODEL}")

        # Queues + state
        self.audio_q = queue.Queue(maxsize=50)
        self.mic_stream = None
        self.running = True
        self.rhyme_process = None
        self.listening = False
        self.is_busy = False
        
        self.serial = None  
        try:
            self.serial = serial.Serial("/dev/ttyUSB0", 9600, timeout=1)
            print("🔗 Arduino connected.")
        except:
            print("⚠ Arduino not found. Movement control disabled.")

        self.GROQ_LLM_URL = "https://api.groq.com/openai/v1/chat/completions"

        # VAD + resampling capability flags
        # self.vad = webrtcvad.Vad(3)  # 0=loose, 3=strict (1 is best for Pi)
        # self.sample_rate = 16000
        # self.frame_ms = 30
        # self.frame_len = int(self.sample_rate * self.frame_ms / 1000)
        # if webrtcvad is not None:
            # try:
                # self.vad = webrtcvad.Vad(3)
                # print("✔ webrtcvad enabled (aggressiveness=2)")
            # except Exception as e:
                # print("⚠ webrtcvad init error:", e)
                # self.vad = None
        # else:
            # print("⚠ webrtcvad not available; skipping VAD")
            
        # self.rms_threshold = 3000     

        # self.resampler_available = resample_poly is not None
        # if self.resampler_available:
        #     print("✔ scipy.resample_poly available for downsampling")
        # else:
        #     print("⚠ scipy.resample_poly not available; using fast downsample fallback")

        # BT sink detection
        self.BLUETOOTH_SINK = self.detect_bt_sink()
        self.output_device = self.BLUETOOTH_SINK or "auto"
        print(f"▶ Output device = {self.output_device}")

        # Warm things up in background to reduce first-response latency
        threading.Thread(target=self.warmup, daemon=True).start()

    def set_language(self, language_name):
        """Switch the assistant's language and TTS model."""
        if language_name in self.language_models:
            self.language = language_name
            # Simplified code mapping
            lang_map = {
                "English": "en", "South English": "en", "Hindi": "hi", 
                "Kannada": "kn", "Tamil": "ta", "Malayalam": "ml", 
                "French": "fr", "Spanish": "es"
            }
            self.language_code = lang_map.get(language_name, "en")
            self.PIPER_MODEL = os.path.join(PROJECT_ROOT, "piper", "models", self.language_models[language_name])
            print(f"🌐 Language switched to: {self.language} (Model: {self.PIPER_MODEL})")
            
            # Confirmation message
            confirm_msgs = {
                "Hindi": "Ab main Hindi mein baat karunga.",
                "Kannada": "Naanu iga Kannada dalli mathaduttene.",
                "Tamil": "Naan ippo Tamilil pesuven.",
                "Malayalam": "Njan ippo Malayalam parayaam.",
                "French": "Je vais maintenant parler en français.",
                "Spanish": "Ahora hablaré en español.",
                "English": "I will now speak in English.",
                "South English": "I will now speak in English."
            }
            self.speak_sentence_stream(confirm_msgs.get(language_name, "Language updated."))
        else:
            print(f"⚠ Unsupported language: {language_name}")

    # ---------------------------------------------------------
    # USB MIC DETECTION
    # ---------------------------------------------------------
    def detect_usb_mic(self):
        try:
            devices = sd.query_devices()
            for idx, dev in enumerate(devices):
                name = (dev.get("name") or "").lower()
                if "usb" in name and dev.get("max_input_channels", 0) > 0:
                    print(f"🎤 USB Mic: {dev['name']} (device {idx})")
                    return idx
        except Exception as e:
            print("⚠ detect_usb_mic error:", e)

        print("⚠ No USB mic found → Using default input")
        return None

    # ---------------------------------------------------------
    # BLUETOOTH SINK DETECTION
    # ---------------------------------------------------------
    def detect_bt_sink(self):
        try:
            out = subprocess.check_output(["pactl", "list", "sinks"], stderr=subprocess.DEVNULL).decode()
            sinks = []
            current = {}
            for line in out.splitlines():
                line = line.strip()
                if line.startswith("Sink #"):
                    if current:
                        sinks.append(current)
                    current = {"id": line}
                elif line.startswith("Name:"):
                    current["name"] = line.split("Name:", 1)[1].strip()
                elif line.startswith("Description:"):
                    current["desc"] = line.split("Description:", 1)[1].strip()
            if current:
                sinks.append(current)
            for s in sinks:
                if s.get("name", "").startswith("bluez_output"):
                    print(f"🔊 Bluetooth Sink Detected: {s['name']}")
                    return s["name"]
        except Exception as e:
            # pactl may not exist or fail; ignore
            pass
        return None

    # ---------------------------------------------------------
    # Mic start/stop (InputStream -> puts numpy arrays on audio_q)
    # ---------------------------------------------------------
    def start_mic(self):
        if self.mic_stream:
            return
        try:
            self.on_listen()
        except:
            pass

        def callback(indata, frames, t, status):
            # print("🎤 chunk received", indata.shape)
            try:
                self.audio_q.put_nowait(indata.copy())
            except queue.Full:
                pass


        try:
            self.mic_stream = sd.InputStream(
                device=self.MIC_DEVICE,
                channels=1,
                samplerate=self.MIC_RATE,
                dtype="int16",
                callback=callback,
            )
            self.mic_stream.start()
        except Exception as e:
            print("⚠ Failed to start mic InputStream:", e)
            self.mic_stream = None


    def stop_mic(self):
        try:
            self.on_speak()
        except:
            pass
        if self.mic_stream:
            try:
                self.mic_stream.stop()
                self.mic_stream.close()
            except:
                pass
            self.mic_stream = None

        
    def resample_48k_to_16k(self, audio_np):
        # 48000 → 16000 = divide by 3
        return resample_poly(audio_np, up=1, down=3).astype("int16")
    

    # def listen_until_silence(self):
        # """
        # Read mic audio, resample to 16k,
        # feed only full utterances to Vosk.
        # No partial processing at all.
        # """
        # while True:
            # audio = self.audio_q.get()     # 48k audio (numpy array)

            # # Resample
            # audio16 = self.resample_48k_to_16k(audio)
            # audio16_bytes = audio16.tobytes()

            # # Only check for full utterance
            # if self.rec.AcceptWaveform(audio16_bytes):
                # result = json.loads(self.rec.Result())
                # text = result.get("text", "").strip()
                # if text:
                    # return text
                    
    def listen_until_silence(self):
        """
        Collect full utterance audio, then send ENTIRE utterance to Whisper.
        """

        utterance_audio = []  # buffer of 48k audio chunks

        while True:
            audio = self.audio_q.get()   # numpy array @ 48kHz
            utterance_audio.append(audio)

            # Resample only for Vosk silence detection
            audio16 = self.resample_48k_to_16k(audio)
            audio16_bytes = audio16.tobytes()

            if self.rec.AcceptWaveform(audio16_bytes):
                # 🔥 FULL utterance audio (this is the key fix)
                full_audio = np.concatenate(utterance_audio)
                utterance_audio.clear()

                # Optional: Vosk text (debug / fallback)
                result = json.loads(self.rec.Result())
                vosk_text = result.get("text", "").strip()

                # Whisper transcription
                whisper_text = self.ask_whisper(full_audio)

                if whisper_text:
                    print("🗣 Whisper:", whisper_text)
                    return whisper_text

                if vosk_text:
                    print("🗣 Vosk:", vosk_text)
                    return vosk_text

                print("⚠ Utterance detected but no text.")


    def normalize_text(t):
        t = t.lower()
        t = re.sub(r"[^\w\s']", "", t)   # remove punctuation
        t = re.sub(r"\s+", " ", t).strip()
        return t
        
    def is_repeated_phrase(text, max_unique_ratio=0.5):
        """
        Returns True if text is mostly repetition
        """
        words = text.split()
        if len(words) < 5:
            return True

        unique = set(words)
        ratio = len(unique) / len(words)

        return ratio < max_unique_ratio
        
    def collapse_repetition(text):
        parts = re.split(r"[?.!]", text)
        seen = []
        for p in parts:
            p = p.strip()
            if p and p not in seen:
                seen.append(p)
        return ". ".join(seen)

    def clean_child_text(self, text: str):
        """
        Cleans Whisper output:
        - Removes filler noises
        - Rejects non-English text
        - Rejects repeated words (what what what)
        - Rejects short questions (<5 words)
        """

        if not text:
            return ""

        t = text.lower().strip()

        #Reject non-English characters (allow a–z, numbers, spaces, basic punctuation)
        if not re.fullmatch(r"[a-z0-9\s?.!,']+", t):
            return ""

        # Filler words to ignore completely
        ignore_exact = {
            "oh", "ok", "okay.", "hmm", "huh", "ha", "haa", "yes",
            "yeah", "yep", "umm", "um", "mmm", "hmm hmm",
            "hmmhmm", "aah", "ohh", "hmm yeah", "shhhh"
        }

        if t in ignore_exact:
            return ""
            
        # if self.is_repeated_phrase(t):
            # return ""

        #Ignore very short utterances
        if len(t) <= 2:
            return ""

        #Ignore repeated sounds (hmmmm, uhhhh)
        if all(c in "hmuaoe" for c in t.replace(" ", "")):
            return ""

        words = t.split()

        #Reject same word repeated continuously
        if len(words) >= 3 and len(set(words)) == 1:
            return ""

        #Reject questions with less than 5 words
        if t.endswith("?"):
            letters_only = re.sub(r"[^a-z]", "", t)
            if len(letters_only) < 3:
                return ""

        return text.strip()
    
    # ---------------------------------------------------------
    #  MAIN FACT-CHECK ROUTER
    # ---------------------------------------------------------
    def fact_check(self, text):
        t = text.lower().strip()

        # --------------------------------------
        # 1. WEATHER (detect city / location)
        # --------------------------------------
        weather_keys = ["weather", "temperature", "climate", "forecast"]
        if any(k in t for k in weather_keys):
            return self._get_weather(t)

        # --------------------------------------
        # 2. FUEL PRICE
        # --------------------------------------
        fuel_keys = ["petrol", "diesel", "fuel", "gasoline"]
        if any(k in t for k in fuel_keys):
            return self._get_fuel_price(t)

        # --------------------------------------
        # 3. METAL PRICES (gold, silver)
        # --------------------------------------
        metal_keys = ["gold", "silver", "metal rate", "gold rate", "silver rate"]
        if any(k in t for k in metal_keys):
            return self._get_metal_price(t)

        # --------------------------------------
        # 4. STOCK MARKET QUERIES
        # --------------------------------------
        stock_keys = [
            "stock", "share price", "share", "nse", "bse",
            "stock price", "today price", "company price"
        ]
        if any(k in t for k in stock_keys):
            return self._get_stock(t)

        # --------------------------------------
        # 5. CURRENCY CONVERSION
        # --------------------------------------
        currency_keys = [
            "dollar", "usd", "inr", "eur",
            "convert", "exchange rate", "currency"
        ]
        if any(k in t for k in currency_keys):
            return self._get_currency_rate(t)

        # --------------------------------------
        # 6. POPULATION QUERIES
        # --------------------------------------
        population_keys = ["population", "people count", "how many people"]
        if any(k in t for k in population_keys):
            return self._get_wikidata(t)

        # --------------------------------------
        # 7. LEADERS / GOVERNMENT
        # --------------------------------------
        leader_keys = [
            "president", "prime minister", "pm", "king", "queen",
            "governor", "chief minister", "cm", "leader",
            "vice president", "vice prime minister"
        ]
        if any(k in t for k in leader_keys):
            return self._get_wikidata(t)

        # --------------------------------------
        # 8. TRENDING NEWS
        # --------------------------------------
        trending_keys = ["trending", "news", "headlines", "hot topics"]
        if any(k in t for k in trending_keys):
            return self._get_trending()

        # --------------------------------------
        # Nothing matched → let LLM answer
        # --------------------------------------
        return None

    # ---------------------------------------------------------
    #  WEATHER — Open-Meteo (No key)
    # ---------------------------------------------------------
    def _get_weather(self, query):
        try:
            match = re.search(r"in ([a-zA-Z\s]+)", query)
            location = match.group(1).strip() if match else "India"

            geo = requests.get(
                f"https://geocoding-api.open-meteo.com/v1/search?name={location}"
            ).json()

            if "results" not in geo:
                return None

            lat = geo["results"][0]["latitude"]
            lon = geo["results"][0]["longitude"]

            data = requests.get(
                f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
            ).json()

            w = data.get("current_weather")
            if not w:
                return None

            return f"Current weather in {location}: {w['temperature']}°C, wind {w['windspeed']} km/h."
        except:
            return None


    # ---------------------------------------------------------
    #  STOCKS — Yahoo Finance
    # ---------------------------------------------------------
    def _get_stock(self, query):
        try:
            mapping = {
                "apple": "AAPL",
                "tesla": "TSLA",
                "google": "GOOGL",
                "microsoft": "MSFT",
                "bitcoin": "BTC-USD",
                "ethereum": "ETH-USD",
                "nifty": "^NSEI",
                "sensex": "^BSESN"
            }

            for name, ticker in mapping.items():
                if name in query:
                    info = yf.Ticker(ticker).history(period="1d")
                    if info.empty:
                        return None
                    price = round(info["Close"][0], 2)
                    return f"Current {name.title()} price is {price}."

            return None
        except:
            return None

    def _get_fuel_price(self, query):
        try:
            data = requests.get("https://dailyfuelpriceindia.com/api/v1/latest").json()
            city = "Delhi"

            for c in data.get("result", []):
                if "mumbai" in query:
                    city = "Mumbai"
                if c["city"].lower() == city.lower():
                    return (
                        f"Fuel price in {city}: petrol {c['petrol']} ₹/L, "
                        f"diesel {c['diesel']} ₹/L."
                    )
            return None
        except:
            return None
        
    def _get_metal_price(self, query):
        try:
            q = query.lower()

            data = requests.get(
                "https://api.metals.live/v1/spot",
                headers={"User-Agent": "Mozilla/5.0"}
            ).json()

            gold = None
            silver = None

            for item in data:
                if "gold" in item:
                    gold = float(item["gold"])
                if "silver" in item:
                    silver = float(item["silver"])

            if "silver" in q and silver:
                return f"Current silver price is {silver:.2f} USD per ounce."

            if "gold" in q and gold:
                return f"Current gold price is {gold:.2f} USD per ounce."

            return None
        except:
            return None

    def _get_currency_rate(self, query):
        try:
            data = requests.get(
                "https://api.exchangerate.host/latest?base=USD&symbols=INR"
            ).json()

            rate = data["rates"]["INR"]
            return f"1 US Dollar equals {rate:.2f} Indian Rupees."

        except:
            return None


    # ---------------------------------------------------------
    #  TRENDING — Google Trends
    # ---------------------------------------------------------
    def _get_trending(self):
        try:
            py = TrendReq()
            df = py.trending_searches(pn="india")
            trends = df[0].tolist()[:5]
            return "Top trending topics: " + ", ".join(trends)
        except:
            return None


    # ---------------------------------------------------------
    #  WIKIDATA — leaders / population
    # ---------------------------------------------------------
    def _get_wikidata(self, query):
        try:
            q = query.lower()

            # -----------------------------------------
            # 1. Detect the location (country/state)
            # -----------------------------------------
            # Find last word(s) after "of" or "in"
            m = re.search(r"(of|in)\s+([a-zA-Z\s]+)", q)
            location = m.group(2).strip() if m else None

            # If not found, try capturing last word as location
            if not location:
                parts = q.split()
                location = parts[-1] if len(parts) > 1 else "India"

            # Clean location
            location = location.replace("?", "").strip()

            # -----------------------------------------
            # 2. Search Wikidata for that location
            # -----------------------------------------
            search_url = (
                "https://www.wikidata.org/w/api.php?"
                f"action=wbsearchentities&search={location}&language=en&format=json"
            )
            lookup = requests.get(search_url).json()

            if not lookup.get("search"):
                return None   # LLM fallback

            qid = lookup["search"][0]["id"]

            entity_url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
            entity = requests.get(entity_url).json()["entities"][qid]

            # -----------------------------------------
            # 3. Population query
            # -----------------------------------------
            if "population" in q:
                pop = self._extract_population(entity)
                if pop:
                    return {
                        "type": "population",
                        "text": f"The population of {location.title()} is {pop}."
                    }
                return None   # LLM fallback

            # -----------------------------------------
            # 4. Leader roles (auto detect)
            # -----------------------------------------
            role_props = {
                "president": "P35",
                "prime minister": "P6",
                "pm": "P6",
                "chief minister": "P1313",
                "cm": "P1313",
                "king": "P97",
                "queen": "P97",
                "leader": "P6"
            }

            # Detect requested role
            requested_role = None
            for role in role_props:
                if role in q:
                    requested_role = role
                    break

            # If no role found → generic leader fetch
            if not requested_role:
                requested_role = "leader"

            prop = role_props[requested_role]

            claims = entity.get("claims", {})

            # If the property exists in entity claims
            if prop in claims:
                leader_qid = claims[prop][0]["mainsnak"]["datavalue"]["value"]["id"]
                leader_name = self._get_label(leader_qid)
                return {
                    "type": "leader",
                    "text": f"{leader_name} is the {requested_role.title()} of {location.title()}."
                }

            return None  # LLM fallback

        except Exception as e:
            return None


    def _extract_population(self, entity):
        claims = entity.get("claims", {})
        if "P1082" in claims:
            return claims["P1082"][0]["mainsnak"]["datavalue"]["value"]["amount"]
        return None

    def _extract_leader(self, entity):
        claims = entity.get("claims", {})
        for prop in ["P6", "P35", "P1313"]:
            if prop in claims:
                obj = claims[prop][0]["mainsnak"]["datavalue"]["value"]
                qid = obj.get("id", "")
                return self._get_label(qid)
        return None


    def _get_label(self, qid):
        url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
        data = requests.get(url).json()
        return data["entities"][qid]["labels"]["en"]["value"]


    # ---------------------------------------------------------
    #  SPORTS — ESPN scrape
    # ---------------------------------------------------------
    def _get_sports(self, query):
        try:
            html = requests.get("https://www.espn.com/latest-news").text
            soup = BeautifulSoup(html, "html.parser")
            headlines = [h.text.strip() for h in soup.select("section h1, section h2")][:5]
            return "Latest sports updates: " + "; ".join(headlines)
        except:
            return None


    # ---------------------------------------------------------
    # Piper TTS streaming helper (streams one sentence immediately)
    # - starts Piper process which writes raw PCM to stdout
    # - pipes raw PCM to pw-play targeting desired device
    # - includes a short silence pad to wake BT
    # ---------------------------------------------------------
    def _play_raw_pcm_via_pw(self, raw_stream_generator):
        """
        raw_stream_generator: iterable yielding bytes chunks (raw s16le at 22050 Hz mono)
        This helper opens pw-play and writes everything to it.
        """
        play_cmd = [
            "pw-play",
            "--target",
            self.output_device,
            "--raw",
            "--rate",
            "16800",
            "--channels",
            "1",
            "--format",
            "s16",
            "-"
        ]
        try:
            p = subprocess.Popen(play_cmd, stdin=subprocess.PIPE)
        except Exception as e:
            print("⚠ pw-play launch failed:", e)
            return

        # initial short silence to wake Bluetooth devices
        try:
            silence = (np.zeros(int(0.03 * 22050))).astype(np.int16).tobytes()
            p.stdin.write(silence)
        except BrokenPipeError:
            try:
                p.stdin.close()
            except:
                pass
            p.wait()
            return
        except Exception:
            pass

        # stream raw chunks
        try:
            for chunk in raw_stream_generator:
                if not chunk:
                    continue
                try:
                    p.stdin.write(chunk)
                except BrokenPipeError:
                    break
        finally:
            try:
                p.stdin.close()
            except:
                pass
            try:
                p.wait()
            except:
                pass

    def _piper_raw_generator_for_text(self, text):
        """
        Launch piper and yield raw bytes as they appear.
        Piper returns signed 16-bit LE PCM at 22050Hz by default with --output_raw.
        """
        piper_cmd = [
            self.PIPER_EXE,  # binary path (dynamic)
            "--model",
            self.PIPER_MODEL,                          # e.g., "models/en_US-amy-low.onnx"
            "--output_raw"                              # outputs raw audio (or use --output_file for WAV)
        ]
        try:
            proc = subprocess.Popen(piper_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except Exception as e:
            print("⚠ Piper launch failed:", e)
            return

        try:
            proc.stdin.write(text.encode("utf-8"))
            proc.stdin.close()
        except Exception:
            pass

        # yield until EOF
        try:
            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                yield chunk
        finally:
            try:
                proc.wait(timeout=2)
            except:
                try:
                    proc.kill()
                except:
                    pass

    def speak_sentence_stream(self, sentence):
        """
        Speak exactly one sentence immediately (used for streaming).
        This blocks until that sentence audio finishes playing.
        """
        if not sentence:
            return

        # ✅ PRINT THE SPOKEN SENTENCE
        print(f"\n🔊 Speaking → {sentence}\n")

        # IMPORTANT: mic already stopped by caller
        raw_gen = self._piper_raw_generator_for_text(sentence)
        if raw_gen is None:
            return
        self._play_raw_pcm_via_pw(raw_gen)


    # ---------------------------------------------------------
    # Date/time/day helper
    # ---------------------------------------------------------
    def check_datetime_request(self, child_text):
        text = (child_text or "").lower()
        now = datetime.now()
        readable_date = now.strftime("%d %B %Y")
        readable_time = now.strftime("%I:%M %p")
        readable_day = now.strftime("%A")

        if any(w in text for w in ["today's date", "what is the date"]):
            return f"Today is {readable_date}."
        if any(w in text for w in ["which day is today", "weekday"]):
            return f"Today is {readable_day}."
        if any(w in text for w in ["time now", "what is the time now"]):
            return f"It is {readable_time}."
        return None

    # ---------------------------------------------------------
    # Rhyme play / stop (ffplay used; non-blocking)
    # ---------------------------------------------------------

    def send_to_arduino(self, message: str):
        """
        Safely send a single-character or short command to Arduino.
        """
        if self.serial and self.serial.is_open:
            try:
                self.serial.write(message.encode())
                print(f"📤 Sent to Arduino → {message}")
            except Exception as e:
                print("⚠ Arduino write error:", e)
        else:
            print("⚠ Arduino not connected, skipping command:", message)


    # -------------------------------
    # SINGLE RHYME HANDLER FUNCTION
    # -------------------------------
    def handle_rhyme(self):
        if self.is_busy:
            return  
        
        self.is_busy = True
        self.stop_mic()
        
        import os, subprocess, time

        file_path = "/home/nova/Documents/Novabot/wheels_on_bus.mp3"

        if not os.path.exists(file_path):
            print("❌ Rhyme file missing")
            self.is_busy = False
            self.start_mic()
            return
        
        self.speak_sentence_stream("Let us sing and dance!")

        # Robot start
        self.send_to_arduino("START")

        print("🎶 Playing rhyme...")

        # Play audio + wait
        try:
            proc = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", file_path]
            )
            proc.wait()
        except Exception as e:
            print("⚠ Error playing rhyme:", e)

        print("🎶 Rhyme finished naturally!")

        # Stop robot
        self.send_to_arduino("STOP")

        time.sleep(0.02)
        self.is_busy = False
        self.start_mic()

        # print("🎤 Mic resumed — ready to listen again!")
    
    def check_rhyme_intent(self, text):
        text = text.lower()

        keywords = [
            "wheels on the bus", "play rhyme", "play song", "dance","can you sing","sing one song",
            "kids rhyme", "poem", "rhyme", "bus song", "can you dance"
        ]

        return any(k in text for k in keywords)


    def is_direct_fact_question(self, text):
        text = text.lower()
        wh_words = ["what", "which", "who", "where", "when", "why", "how"]
        return any(text.startswith(w) for w in wh_words)
    

    def _is_weather(self, t):
        return any(k in t for k in ["weather", "temperature", "climate", "forecast"])

    def _is_fuel(self, t):
        return any(k in t for k in ["petrol", "diesel", "fuel", "gasoline"])

    def _is_metal(self, t):
        return any(k in t for k in ["gold", "silver", "gold rate", "silver rate"])

    def _is_stock(self, t):
        return any(k in t for k in ["stock", "share", "nse", "bse", "price of"])

    def _is_currency(self, t):
        return any(k in t for k in ["usd", "dollar", "inr", "exchange", "currency"])

    def _is_leader(self, t):
        return any(k in t for k in [
            "president", "prime minister", "pm",
            "chief minister", "cm", "leader", "governor"
        ])

    def _is_population(self, t):
        return "population" in t

    # ---------------------------------------------------------
    # Force one natural sentence OR allow multi-sentence when needed
    # - Short-def questions -> single sentence
    # - Long/explain questions -> return full (allow streaming multiple sentences)
    # ---------------------------------------------------------
    def force_single_sentence(self, question, answer_text):
        q = (question or "").lower()
        a = (answer_text or "").strip()

        # -------- KEYWORD GROUPS --------
        short_def_keywords = [
            "what is", "who is", "define", "meaning of", "which is", "what's",
            "largest", "biggest", "longest", "do you like"
        ]

        long_keywords = [
            "explain", "describe", "how does", "how do",
            "why", " teach alphabets", "teach me alphabets", "tell me more", "in detail", "story"
        ]

        list_keywords = [
            "name the", "list", "seven wonders", "which are",
            "wonders of the world", "planets", "continents",
            "oceans", "states", "countries", "rivers",
            "animals"
        ]

        # -------- SPECIAL CASE: LIST QUESTIONS --------
        if any(k in q for k in list_keywords):
            items = re.split(r'[\n•\-]+|\d+\.', a)
            cleaned = []

            for it in items:
                it = it.strip()
                if len(it) < 2:
                    continue
                short = re.split(r'[,.!?-]', it)[0].strip()
                cleaned.append(short)

            cleaned = cleaned[:8]
            return "\n".join(f"- {c}" for c in cleaned)

        # -------- SHORT DEFINITION QUESTIONS --------
        if any(k in q for k in short_def_keywords):
            parts = re.split(r'(?<=[.?!])\s+', a)
            return parts[0].strip()

        # -------- EXPLICIT LONG EXPLANATION --------
        if any(k in q for k in long_keywords):
            return a.strip()

        # -------- DEFAULT: MAX 2 SENTENCES --------
        parts = re.split(r'(?<=[.?!])\s+', a)
        parts = [p.strip() for p in parts if p.strip()]
        return " ".join(parts[:2])


    # ---------------------------------------------------------
    # Sentence detector that yields finished sentences as tokens stream arrives
    # (buffers until a punctuation ends a sentence; yields sentences as soon as completed)
    # ---------------------------------------------------------
    def sentence_generator(self, stream):
        """
        stream: iterator/generator of strings (token pieces) from ask_ollama
        yields: full sentences as soon as they are complete
        """
        buffer = ""
        for tok in stream:
            if not tok:
                continue
            buffer += tok
            # try to split into sentences; last element may be incomplete
            parts = re.split(r'(?<=[.!?])\s+', buffer)
            # if only one part and doesn't end with sentence punctuation, continue buffering
            if len(parts) == 1 and not re.search(r'[.!?]$', parts[0]):
                continue
            # keep last (possibly incomplete) in buffer
            buffer = parts.pop() if parts else ""
            for s in parts:
                s = s.strip()
                if s:
                    yield s
        # After stream finished, flush leftover
        leftover = buffer.strip()
        if leftover:
            yield leftover

    # ---------------------------------------------------------
    # Ask Ollama streaming (keeps same API used before)
    # returns generator of token strings
    # ---------------------------------------------------------
    def clean_for_groq(self, text):
        """ Remove broken UTF-8 and unsafe characters. """
        if not text:
            return ""

        # Remove corrupted UTF-8 bytes
        text = text.encode("utf-8", "ignore").decode("utf-8")

        # Remove invisible control chars
        text = "".join(ch for ch in text if ch.isprintable() or ch.isspace())

        return text.strip()
        
    
    def ask_whisper(self, audio_np, model_name="whisper-large-v3-turbo"):
        """
        SAFE, STABLE Groq Whisper transcription.
        - Resamples 48k → 16k
        - Writes temp WAV
        - Uses Groq REST API (NO openai module)
        - Retries on failure
        """

        import tempfile
        import wave
        import requests
        import time

        # 1️⃣ Resample to 16k
        audio16 = self.resample_48k_to_16k(audio_np)

        # 2️⃣ Write temporary WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            try:
                wf = wave.open(tmp.name, "wb")
                wf.setnchannels(1)
                wf.setsampwidth(2)   # int16
                wf.setframerate(16000)
                wf.writeframes(audio16.tobytes())
                wf.close()
            except Exception as e:
                print(f"⚠ WAV write failed: {e}")
                return ""

            url = "https://api.groq.com/openai/v1/audio/transcriptions"
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }

            for attempt in range(3):
                try:
                    with open(tmp.name, "rb") as f:
                        files = {
                            "file": ("audio.wav", f, "audio/wav")
                        }
                        data = {
                            "model": model_name,
                            "response_format": "text"
                        }

                        r = requests.post(
                            url,
                            headers=headers,
                            files=files,
                            data=data,
                            timeout=120
                        )

                    if r.status_code != 200:
                        print(f"⚠ Groq Whisper bad status {r.status_code}: {r.text[:200]}")
                        time.sleep(1)
                        continue

                    text = self.clean_for_groq(r.text.strip())
                    if text:
                        return text
                    else:
                        print("⚠ Empty Whisper transcript.")
                        return ""

                except Exception as e:
                    print(f"⚠ ask_whisper error (attempt {attempt+1}/3): {e}")
                    time.sleep(1)

        print("❌ Groq Whisper failed after 3 retries.")
        return ""

        


    def ask_groq(self, prompt, model_name="llama-3.1-8b-instant"):
        """
        SAFE, STABLE streaming generator for Groq API.
        Handles corrupted utf-8, empty text, 400 errors.
        """

        # Clean input
        prompt = self.clean_for_groq(prompt)
        if not prompt:
            print("⚠ Empty prompt after cleaning.")
            return

        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model_name,
            "stream": True,
            "messages": [
                {"role": "system", "content": f"You are NOVA, a friendly female Indian teacher. Answer strictly in {self.language}. Keep it child-friendly and concise."},
                {"role": "user", "content": prompt}
            ]
        }

        for attempt in range(3):
            try:
                r = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=120
                )

                if r.status_code != 200:
                    print(f"⚠ Groq bad status {r.status_code}: {r.text[:200]}")
                    continue

                for raw in r.iter_lines():
                    if not raw:
                        continue

                    if raw.startswith(b"data: "):
                        raw = raw[6:]

                    try:
                        evt = json.loads(raw.decode("utf-8", "ignore"))
                    except Exception:
                        continue

                    if "choices" in evt:
                        delta = evt["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta

                    if evt.get("choices", [{}])[0].get("finish_reason") == "stop":
                        return

                return

            except Exception as e:
                print(f"⚠ ask_groq error (attempt {attempt+1}/3): {e}")

        print("❌ Groq failed after 3 retries.")
        return


    # ---------------------------------------------------------
    # Warmup - runs in background at init to reduce first-latency
    # ---------------------------------------------------------
    def warmup(self):
        print("Warmup started...")
        # Vosk no-op (safe)
        print("✔ Vosk warmed")
        
        # Piper warm
        try:
            p = subprocess.Popen(
                [self.PIPER_EXE, "--model", self.PIPER_MODEL, "--output_raw"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            p.stdin.write(b"warmup")
            p.stdin.close()
            try:
                p.stdout.read(256)
            except:
                pass
            try:
                p.kill()
            except:
                pass
            print("✔ Piper warmed")
        except:
            pass
        # Ollama warm
        try:
            next(self.ask_groq("Hello"))
            print("LLM ready")
        except Exception:
            print("LLM warmup failed")
        print("Warmup complete.")

    # ---------------------------------------------------------
    # Main loop — listens, filters, stops mic, streams LLM + TTS
    # ---------------------------------------------------------
    def run(self):
        while self.running:

            self.start_mic()
            print("Listening!!!")
            text = self.listen_until_silence()
            text = self.clean_child_text(text)
            
            if not text:
                continue

            print("\n👧 Child:", text)

            if self.on_question:
                self.on_question(text)

                # Stop mic before replying
            self.stop_mic()

                # -------------------------
                # BASIC INTENT HANDLING
                # -------------------------
            tl = text.lower()

            if "hey" in tl or "hello" in tl:
                self.speak_sentence_stream("Hello! How can I assist you?")
                self.start_mic()
                continue
            
            if "your name" in tl:
                self.speak_sentence_stream("My name is NOVA.")
                self.start_mic()
                continue

            if "who are you" in tl:
                self.speak_sentence_stream("I'm NOVA, your friendly teacher.")
                self.start_mic()
                continue

            if "how are you" in tl:
                self.speak_sentence_stream("I'm fine, thank you!")
                self.start_mic()
                continue

            if "prime minister of india" in tl:
                self.speak_sentence_stream("Narendra Modi is the current prime minister of india")
                self.start_mic()
                continue
            
            if "president of america" in tl:
                self.speak_sentence_stream("Donald trump is the current president of america")
                self.start_mic()
                continue


            if "nova tech" in tl:
                self.speak_sentence_stream("Nova tech is a robotics company located in Bangalore.")
                self.start_mic()
                continue


            # Stop all and exit
            if "you can stop" in tl or "bye" in tl:
                self.speak_sentence_stream("Goodbye! See you soon.")
                self.start_mic()
                continue

            # Rhyme request
            if self.check_rhyme_intent(text):
                self.handle_rhyme()
                self.start_mic()
                continue

                # Date/time/day
            datetime_answer = self.check_datetime_request(text)
            if datetime_answer:
                self.speak_sentence_stream(datetime_answer)
                self.start_mic()
                continue

            # ---------------------------------------------------------
            # FACT CHECK — weather, stocks, population, leaders, trends
            # ----------------------------------------------------------

            if self._is_weather(tl):
                ans = self._get_weather(text)
                if ans:
                    self.speak_sentence_stream(ans)
                    self.start_mic()
                    continue

            if self._is_fuel(tl):
                ans = self._get_fuel_price(text)
                if ans:
                    self.speak_sentence_stream(ans)
                    self.start_mic()
                    continue

            if self._is_metal(tl):
                ans = self._get_metal_price(text)
                if ans:
                    self.speak_sentence_stream(ans)
                    self.start_mic()
                    continue

            if self._is_stock(tl):
                ans = self._get_stock(text)
                if ans:
                    self.speak_sentence_stream(ans)
                    self.start_mic()
                    continue

            if self._is_currency(tl):
                ans = self._get_currency_rate(text)
                if ans:
                    self.speak_sentence_stream(ans)
                    self.start_mic()
                    continue

            if self._is_population(tl):
                ans = self._get_wikidata(text)
                if ans:
                    self.speak_sentence_stream(ans)
                    self.start_mic()
                    continue

            if self._is_leader(tl):
                ans = self._get_wikidata(text)
                if ans:
                    self.speak_sentence_stream(ans)
                    self.start_mic()
                    continue

            # -------------------------
            # FACT QUESTIONS
            # -------------------------
            if self.is_direct_fact_question(text):
                prompt = (
                    f"You are NOVA, a knowledgeable Indian teacher for children. "
                    f"Answer the following question with ONE short, factual sentence ONLY. "
                    f"Do NOT add extra details, definitions, introductions, or guesses.\n\n"
                    f'Child asked: "{text}"\n'
                    f'Answer directly with the latest known fact:'
                )

                tokens = [tok for tok in self.ask_groq(prompt)]
                full = "".join(self.sentence_generator(tokens))
                final = self.force_single_sentence(text, full)

                self.speak_sentence_stream(final)
                self.start_mic()
                continue

           
                # -------------------------
                # GENERAL LLM PROMPT
                # -------------------------
            prompt = f"""
            You are NOVA, a friendly female Indian teacher for children.

            GENERAL RULES:
            - Use child-friendly, simple language.
            - Keep answers short and direct for normal questions.
            - Use 1–2 sentences unless the question clearly needs more.
            - Avoid greetings, intros, or repeated phrases.
            - Do not add “I’m NOVA your friendly teacher” in every answer.
            - If the child asks “Who are you?” reply exactly: “I'm NOVA, your friendly teacher.”
            - Avoid complex words unless necessary.
            - Do not read anything inside asterisks *like this*.
            - Never cut a sentence in the middle. Always complete the thought.
            - If President of America is asked then it is "Donald trump"


            FACT RULES:
            - If the question includes “biggest”, “largest”, “fastest”, “longest”, etc., answer using world facts unless a location is clearly given.

            STORY / EXPLANATION / LONG ANSWER RULES:
            - If the child asks:
            “tell me a story”, “story”, “explain”, “describe”, 
            “how does”, “how do”, “why”, “tell me more”, “in detail”
            → Give a full, natural multi-sentence explanation or story in maximum 10-12 sentence.
            → Make it simple and child-friendly.

            LIST RULES:
            - If the question asks for lists such as:
            “list”, “name the”, “continents”, “planets”, “countries”, 
            “oceans”, “states”, “rivers”, etc.
            → Give a clean list with each item on a new line.
            → No numbering unless asked.

            DEFINITION RULES:
            - If the child asks “what is”, “who is”, “define”, “meaning of”
            → Give one short sentence.
           
            IMPORTANT:
            - Do not add extra formatting like emojis, bullets, symbols, or prefixes.
            - Output clean plain text only.
            - Do not add unnecessary extra information.

            Child asked: "{text}"

            NOVA:
            """

                # -------------------------
                # STREAM LLM → PIPER LIVE
                # -------------------------


            sentence_buffer = ""
            sentence_end = re.compile(r"[.!?]")

            for tok in self.ask_groq(prompt):

                sentence_buffer += tok

                # Check for completed sentences
                while True:
                    m = sentence_end.search(sentence_buffer)
                    if not m:
                        break

                    # Extract one full sentence
                    end_index = m.end()
                    raw_sentence = sentence_buffer[:end_index].strip()
                    sentence_buffer = sentence_buffer[end_index:].lstrip()

                    # Apply your answer rules
                    final_sentence = self.force_single_sentence(text, raw_sentence)

                    if final_sentence:
                        self.speak_sentence_stream(final_sentence)

            # leftover text at the end
            if sentence_buffer.strip():
                final_sentence = self.force_single_sentence(text, sentence_buffer.strip())
                if final_sentence:
                    self.speak_sentence_stream(final_sentence)

            self.start_mic()
