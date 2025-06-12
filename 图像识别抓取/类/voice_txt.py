import whisper
import torch
import sounddevice as sd
import numpy as np
from deep_translator import GoogleTranslator
import threading
import keyboard  

class VoiceCommand:
    def __init__(self, model_size="base"):
        self.model = whisper.load_model(model_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def record_until_enter(self, samplerate=16000) -> np.ndarray:
        print("开始讲话，按 Enter 结束录音...")
        recorded = []

        def callback(indata, frames, time, status):
            if status:
                print("erro", status)
            recorded.append(indata.copy())

        stream = sd.InputStream(samplerate=samplerate, channels=1, dtype='float32', callback=callback)
        with stream:
            keyboard.wait('enter')  # 用户按下 Enter 键
        print(" 录音结束")
        return np.concatenate(recorded, axis=0).squeeze()

    def record_and_recognize(self, source_lang="zh", target_lang="en") -> str:
        audio = self.record_until_enter()
        print("正在语音识别...")
        result = self.model.transcribe(audio, language=source_lang)
        text = result["text"].strip()
        print(f"识别内容: {text}")

        print("正在翻译...")
        translated = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        print(f" 翻译后 Prompt: {translated}")
        return translated
