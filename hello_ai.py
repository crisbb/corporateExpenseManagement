import time
from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError

client = OpenAI(
    api_key = 'sk-882d93e0832348979b1a3f1702bac021',
    base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
)
messages = [{
    'role': 'system',
    'content': '你是一个毒舌但有用的编程助手，回答问题时带点嘲讽。'
}]
def chat_with_retry(messages, max_retries = 3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(  # type: ignore
                model = 'qwen-plus',
                messages = messages,  # type: ignore
                stream=True
            )
            return response
        
        except RateLimitError:
            wait_time = 2 ** attempt # 指数退避（Exponential Backoff）
            print(f"\n⚠️ 限流，{wait_time}秒后第 {attempt + 1} 次重试...")
            time.sleep(wait_time)

        except APIConnectionError:
            wait_time = 2 ** attempt
            print(f"\n⚠️ 网络异常，{wait_time}秒后第 {attempt + 1} 次重试...")
            time.sleep(wait_time)

        # 所有重试都失败了
    raise Exception(f"重试 {max_retries} 次后仍然失败")
while True:
    user_input = input('你:')

    if user_input.lower() == 'quit':
        print('退出')
        break

    messages.append({
        'role': 'user',
        'content': user_input
    })
    print(f'[DEBUG] 当前发送了 {len(messages)}条信息')
    if (len(messages)>10):
        messages = messages[-10:]
    try:
        response = chat_with_retry(messages)
        print('AI:', end = '')
        full_content = ''
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                text = chunk.choices[0].delta.content
                print(text, end='', flush=True)
                full_content+=text
        print('\n')
        messages.append({
            'role': 'assistant',
            'content': full_content
        })
    except Exception as e:
        print(f"\n⚠️ 最终失败: {e}\n")
        messages.pop()