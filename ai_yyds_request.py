import os

import requests

# 设置 API 密钥（从环境变量读取，不再硬编码）
api_key = os.environ.get('AI_YYDS_API_KEY')
if not api_key:
    raise RuntimeError("AI_YYDS_API_KEY is not set.")

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

