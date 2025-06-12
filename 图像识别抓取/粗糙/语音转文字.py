import whisper
import torch
from deep_translator import GoogleTranslator

model = whisper.load_model("turbo")

result = model.transcribe(r"C:\Users\zhong\Desktop\6月3日.WAV")
text=result["text"]

print(text)

translated = GoogleTranslator(source='zh', target='en').translate(text)
print(translated.text)