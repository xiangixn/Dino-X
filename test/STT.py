import whisper
import torch
from deep_translator import GoogleTranslator

model = whisper.load_model("turbo")

result = model.transcribe("1.WAV")
text = result["text"]

print(text)

# 使用deep-translator进行翻译
translator = GoogleTranslator(source='zh-CN', target='en')
translated = translator.translate(text)
print(translated)