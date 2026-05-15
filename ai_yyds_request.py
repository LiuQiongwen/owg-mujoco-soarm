import requests

# 设置 API 密钥
api_key = 'sk-A9q5TscQFLIV7ZTAB29f5c93E1D44f4880F91c24FcAa4eDd'  # 请替换为你的 AI-YYDS API 密钥

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

data = {
    'model': 'gpt-4o',  # 可以换成其他支持的模型
    'messages': [
        {'role': 'system', 'content': '你是一个友好的助手。'},
        {'role': 'user', 'content': '你好，AI！'}
    ]
}

response = requests.post('https://api.ai-yyds.com/v1/chat/completions', json=data, headers=headers)

if response.status_code == 200:
    result = response.json()
    print(f"AI 回复: {result['choices'][0]['message']['content']}")
else:
    print(f"请求失败，状态码：{response.status_code}，错误信息：{response.text}")

