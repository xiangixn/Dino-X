import whisper
import torch
from deep_translator import GoogleTranslator

class VoiceCommand:
    def __init__(self):
        self.model = whisper.load_model("turbo")

    def run(self):
        result = self.model.transcribe("2.WAV")
        text = result["text"]
        print(text)

        translator = GoogleTranslator(source='zh-CN', target='en')
        translated = translator.translate(text)
        print(translated)
        return translated

# import whisper
# import torch
# from deep_translator import GoogleTranslator
# from volcenginesdkarkruntime import Ark

# class VoiceCommand:
#     def __init__(self, api_key):
#         self.model = whisper.load_model("turbo")
#         self.client = Ark(api_key=api_key)

#     def run(self):
#         result = self.model.transcribe(r"C:\Users\zhong\Desktop\6月3日.WAV")
#         text = result["text"]
#         print(f"识别的文本: {text}")

#         messages = [
#             {"role": "user", "content": f"请提炼夹取的物体：\n{text}，并翻译成英文"}
#         ]
#         completion = self.client.chat.completions.create(
#             model="doubao-seed-1-6-250615", 
#             messages=messages
#         )
#         response_content = completion.choices[0].message.content
#         print(f"翻译的结果: {response_content}")
        
#         return response_content
