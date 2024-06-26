import streamlit as st
import replicate
import os
from transformers import AutoTokenizer
import time

# Set assistant icon to Snowflake logo
icons = {"assistant": "./Snowflake_Logomark_blue.svg", "user": "⛷️", "user_anon": "🕵️‍♂️"}

# App title
st.set_page_config(page_title="Snowflake Arctic")

# Replicate Credentials
with st.sidebar:
    st.title('Snowflake Arctic')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your Replicate API token.', icon='⚠️')
            st.markdown("**Don't have an API token?** Head over to [Replicate](https://replicate.com) to sign up for one.")

    os.environ['REPLICATE_API_TOKEN'] = replicate_api
    
    chat_token = st.text_input('Enter Your Chat Token:', type='password')
    if chat_token:
        if chat_token == "penguin":
            st.success('Token accepted.')
        else:
            st.warning('Invalid Chat Token.', icon='⚠️')
            st.markdown("**Forgot your token?** Hint: penguin")
    else:
        st.warning('Please enter your Chat Token.', icon='⚠️')
        
    st.subheader("Adjust model parameters")
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.3, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    chat_sessions = st.sidebar.slider('Chat Sessions Saved', min_value=1, max_value=25, value=15, step=1)

# Store LLM-generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hi. I'm Arctic, a new, efficient, intelligent, and truly open language model created by Snowflake AI Research. Ask me anything."}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=icons[message["role"]]):
        st.write(message["content"])

# Function to clear chat history
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hi. I'm Arctic, a new, efficient, intelligent, and truly open language model created by Snowflake AI Research. Ask me anything."}]
    st.session_state.chat_aborted = False


def private_mode():
    if "private_mode" not in st.session_state:
        st.session_state.private_mode = False
        
    if not st.session_state.private_mode:
        st.toast('Privacy mode Enabled!', icon='🛡️')
        st.session_state.private_mode = True
    else:
        st.toast('Privacy mode Disabled!', icon='🔓')
        st.session_state.private_mode = False
        
def Anon_mode():
    if "Anon_mode" not in st.session_state:
        st.session_state.Anon_mode = False
        
    if not st.session_state.Anon_mode:
        st.toast('Anonymous mode Enabled!', icon='🕵️‍♂️')
        st.session_state.Anon_mode = True
    else:
        st.toast('Anonymous mode Disabled!', icon='⛷️')
        st.session_state.Anon_mode = False
        
if "Anon_mode" not in st.session_state:
    st.session_state.Anon_mode = False

st.sidebar.button('Clear chat history', on_click=clear_chat_history)
st.sidebar.toggle('Privacy', on_change=private_mode)
st.sidebar.toggle('Anonymous User', on_change=Anon_mode)

@st.cache_resource(show_spinner=False)
def get_tokenizer():
    return AutoTokenizer.from_pretrained("huggyllama/llama-7b")

def get_num_tokens(prompt):
    """Get the number of tokens in a given prompt"""
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(prompt)
    return len(tokens)


# Function for generating Snowflake Arctic response
def generate_arctic_response():
    prompt = []
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            prompt.append("<|im_start|>user\n" + dict_message["content"] + "<|im_end|>")
        else:
            prompt.append("<|im_start|>assistant\n" + dict_message["content"] + "<|im_end|>")
    
    prompt.append("<|im_start|>assistant")
    prompt.append("")
    prompt_str = "\n".join(prompt)
    
    if get_num_tokens(prompt_str) >= 3072:
        st.error("Conversation length too long. Please keep it under 3072 tokens.")
        st.button('Clear chat history', on_click=clear_chat_history, key="clear_chat_history")
        st.stop()

    for event in replicate.stream("snowflake/snowflake-arctic-instruct",
                           input={"prompt": prompt_str,
                                  "prompt_template": r"{prompt}",
                                  "temperature": temperature,
                                  "top_p": top_p,
                                  }):
        yield str(event)

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    if st.session_state.Anon_mode:
        st.session_state.messages.append({"role": "user_anon", "content": prompt})
    else:   
        st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.Anon_mode:
        with st.chat_message("user", avatar=icons["user_anon"]):
            st.write(prompt)
    else:   
        with st.chat_message("user", avatar=icons["user"]):
            st.write(prompt)
        
# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] not in ["assistant", "gamemaster"]:
    with st.chat_message("assistant", avatar="./Snowflake_Logomark_blue.svg"):
        response = generate_arctic_response()
        full_response = st.write_stream(response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
    
