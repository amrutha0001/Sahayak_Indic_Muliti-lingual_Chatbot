# translator.py
from deep_translator import GoogleTranslator

LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "Bengali": "bn",
    "Marathi": "mr",
    "Gujarati": "gu",
}

class Translator:
    def __init__(self):
        print("✅ Translator ready (Google, no model download)")

    def translate(self, text, src_lang, tgt_lang):
        if src_lang == tgt_lang:
            return text
        try:
            return GoogleTranslator(source=src_lang, target=tgt_lang).translate(text)
        except Exception as e:
            print(f"Translation failed: {e}")
            return text  # fallback: return original
