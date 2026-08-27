import streamlit as st
from openai import OpenAI, RateLimitError, APIConnectionError

# === 页面配置 ===
st.set_page_config(page_title="AI 聊天助手", page_icon="💬", layout="wide")
# === 初始化 ===

if 'messages' not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header('⚙️ 设置')

    api_key = st.text_input("API Key", type='password')

    model_options = {
        "DeepSeek-V3": {"base_url": "https://api.deepseek.com", "model": "deepseek-chat"},
        "DeepSeek-R1": {"base_url": "https://api.deepseek.com", "model": "deepseek-reasoner"},
        "通义千问": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-turbo"},
        "智谱GLM": {"base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-4-flash"},
    }
    selected = st.selectbox('选择模型', list(model_options.keys()))
    base_url = model_options[selected]['base_url']
    model = model_options[selected]['model']

    system_prompt = st.text_area('system prompt人设', value='你是一个有帮助的AI助手。',height=100)
    st.divider()

    # 导出对话
    if "messages" in st.session_state and st.session_state.messages:
        chat_text = ''
        for msg in st.session_state.messages:
            role = '用户' if msg['role'] == 'user' else 'AI'
            chat_text += f'\n{role}:\n{msg['content']}\n\n'
        st.download_button(
            label='导出对话',
            data=chat_text.encode('utf-8'),
            file_name='chat_history.txt',
            mime='text/plain'
        )
    # 清空对话
    if (st.button('清空对话')):
        st.session_state.messages = []
        st.rerun()

def get_api_msgs():
    return [{'role':'system', 'content': system_prompt}] + st.session_state.messages
# === 主区域 ===
st.title("💬 AI 聊天助手")

for msg in st.session_state.messages:
    if msg['role'] == 'system':
        continue
    with st.chat_message(msg['role']):
        st.write(msg['content'])

# 聊天输入框（自动固定在底部）
if user_input := st.chat_input('输入你的问题...'):# 赋值 + 判断，一步完成
    if not api_key:
        st.warning('请先在侧边栏输入 API Key')
    else:
        # 保存并显示用户消息
        st.session_state.messages.append({
            'role': 'user',
            'content': user_input
        })
        # 消息气泡容器
        with st.chat_message('user'):
            st.write(user_input)

        try:
            client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            response = client.chat.completions.create(# type: ignore
                model=model,
                messages=get_api_msgs(),
                stream=True
            )
    
            with st.chat_message('assistant'):
                placeholder = st.empty() #创建一个可更新的占位符，不断覆盖内容，就实现了"逐字打出"的效果。
                full_content = ''
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        full_content += chunk.choices[0].delta.content
                        placeholder.write(f'{full_content} ▌')
                placeholder.write(full_content)
        
            st.session_state.messages.append({
                'role': 'assistant',
                'content': full_content
            })
        except RateLimitError:
            st.error("⚠️ 请求太频繁，请稍后再试")
        except APIConnectionError:
            st.error("❌ 网络连接失败，请检查网络")
        except Exception as e:
            st.error(f"❌ 出错: {e}")

# LangChain = 胶水层，把 LLM 调用、记忆、检索、工具调用粘在一起。

# LLM = Large Language Model（大语言模型） DeepSeek 通义千问 Claude
# GPT Generative Pre-trained Transformer
# NLP Natural Language Processing	自然语言处理
# RAG Retrieval Augmented Generation	检索增强生成（你第 4 周要学的）

