from voice_txt import VoiceCommand

api_key = "cf0a7843-17b2-445c-b383-ecd0e88fd31e"
voice_command = VoiceCommand(api_key)
translation = voice_command.run() # 语音转英文

print(f"识别并翻译后的 Prompt: {translation}")