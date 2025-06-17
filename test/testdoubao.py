from volcenginesdkarkruntime import Ark

# 设置 API 密钥
client = Ark(api_key="cf0a7843-17b2-445c-b383-ecd0e88fd31e")

# 定义输入文本
input_text = """
熊猫想喝可乐
"""
messages = [
    {"role": "user", "content": f"请提炼夹取的物体：\n{input_text}"}
]
completion = client.chat.completions.create(
    model="doubao-seed-1-6-250615", 
    messages=messages
)
response_content = completion.choices[0].message.content
print(response_content)