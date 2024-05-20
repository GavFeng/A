import streamlit as st
import replicate
import os
from transformers import AutoTokenizer
import time

# Set assistant icon to Snowflake logo
icons = {"assistant": "./Snowflake_Logomark_blue.svg", "user": "⛷️", "user_anon": "🕵️‍♂️"}

# Timeout threshold in seconds
TIMEOUT_THRESHOLD = 10

# Initialize last activity time
if "last_activity_time" not in st.session_state:
    st.session_state.last_activity_time = time.time()

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
    st.subheader("Adjust model parameters")
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.3, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)

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
        st.session_state.messages.append({"role": "assistant", "content": "I am now in privacy mode."})
        st.session_state.private_mode = True
    else:
        st.session_state.messages.append({"role": "assistant", "content": "Privacy mode deactivated."})
        st.session_state.private_mode = False
        
def Anon_mode():
    if "Anon_mode" not in st.session_state:
        st.session_state.Anon_mode = False
        
    if not st.session_state.Anon_mode:
        st.session_state.messages.append({"role": "assistant", "content": "I am now in Anonymous mode."})
        st.session_state.Anon_mode = True
    else:
        st.session_state.messages.append({"role": "assistant", "content": "Privacy mode deactivated."})
        st.session_state.Anon_mode = False
        
def timeout_mode():
    if "timeout_mode" not in st.session_state:
        st.session_state.timeout_mode = False
        
    if not st.session_state.timeout_mode:
        st.session_state.messages.append({"role": "assistant", "content": "Automatic timeout mode enabled."})
        st.session_state.timeout_mode = True
    else:
        st.session_state.messages.append({"role": "assistant", "content": "Automatic timeout mode disabled."})
        st.session_state.timeout_mode = False


st.sidebar.button('Clear chat history', on_click=clear_chat_history)
st.sidebar.toggle('Privacy', on_change=private_mode)
st.sidebar.toggle('Anonymous User', on_change=Anon_mode)
st.sidebar.toggle('Timeout', on_change=timeout_mode)

# Function to check for inactivity and perform timeout action
def check_inactivity():
    # Get the current time
    current_time = time.time()
    
    # Calculate the elapsed time since the last activity
    elapsed_time = current_time - st.session_state.last_activity_time
    
    # Update the last activity time
    st.session_state.last_activity_time = current_time
    
    # Check if timeout mode is enabled and elapsed time exceeds the timeout threshold
    if st.session_state.timeout_mode and elapsed_time > TIMEOUT_THRESHOLD:
        # Perform timeout action
        st.session_state.messages.append({"role": "assistant", "content": "Session timed out due to inactivity."})

    # Trigger the script to rerun after a short delay
    st.experimental_set_query_params(_timeout_refreshed=current_time)



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
    
