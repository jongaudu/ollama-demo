from pandasai.llm.local_llm import LocalLLM
from pandasai import SmartDataframe
import pandas as pd
import streamlit as st
from csclient import EventingCSClient

st.set_page_config(page_title="Local LLM Demo", layout="wide")

cp = EventingCSClient('ollama-demo')

model = LocalLLM(api_base='http://localhost:11434/v1', model='llama3')

st.title(f'Chat with your data using {model.model} 🦙...')
st.header('')
st.write('client_usage stats:')

client_uasge = cp.get('status/client_usage/stats')

df = pd.DataFrame(client_uasge)

df = df.drop(columns=['up_delta', 'up_packets', 'down_delta', 'down_packets', 'app_list'])
df = df.rename(columns={'up_bytes': 'up MB', 'down_bytes': 'down MB', 
                        'last_time': 'last seen', 'first_time': 'first seen', 'connect_time': 'connect duration'})
df['up MB'] = round(df['up MB'] / 1000000, 2)
df['down MB'] = round(df['down MB'] / 1000000, 2)
df.insert(8, 'total MB', 0)
df['total MB'] = df['up MB'] + df['down MB']
df['last seen'] = pd.to_datetime(df['last seen'] - 14400, unit='s')
df['first seen'] = pd.to_datetime(df['first seen'] - 14400, unit='s')
df['connect duration'] = pd.to_datetime(df['connect duration'], unit='s').dt.strftime('%H:%M:%S')

st.write(df)

df = SmartDataframe(df, config={'llm': model})

st.header('')
prompt = st.text_area('Prompt:')

if st.button('Submit'):
    if prompt:
        with st.spinner('Generating response...'):
            st.text(f'Response: {df.chat(prompt)}')


st._bottom.write('---')
st._bottom.write('Built using PandasAI (https://pandas-ai.com) and Streamlit (https://streamlit.io)')
st._bottom.write('Powered by Ollama (https://ollama.com) and llama3 8B (https://llama.meta.com/llama3/)')
st._bottom.write('Created by Jon Gaudu (https://github.com/jongaudu)')
