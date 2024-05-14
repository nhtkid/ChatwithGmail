# The following python code was built in VSCode
# Have your api key saved in a file called .env and your key should look like OPENAI_API_KEY = SK-xxxx
# This app is built with Streamlit. To run the app, do "Streamlit Run path/to/file" in the VSCode terminal
# To use the Gmail toolkit from Langchain, follow https://developers.google.com/gmail/api/quickstart/python#authorize_credentials_for_a_desktop_application to download the OAuth Client.

pip install -U google-api-python-client google-auth-httplib2 google-auth-oauthlib beautifulsoup4
pip install -U langchain langchain-community langchain-openai langchainhub python-dotenv streamlit

import os
import streamlit as st
from langchain_community.agent_toolkits import GmailToolkit
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Instantiate GmailToolkit
toolkit = GmailToolkit()

# Get tools
tools = toolkit.get_tools()

# Set up the LLM and agent
instructions = "You are an assistant and you are very good at managing my emails."
base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)
agent = create_openai_functions_agent(llm, tools, prompt)

# Initialize Streamlit app
st.title("Chat to your Gmail")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask your question:"):
    # Append user input to the messages and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Prepare the input for the agent executor
    agent_input = {
        "input": prompt
    }
    
    # Invoke agent executor and get response
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
    )
    response = agent_executor.invoke(agent_input)
    
    # Append the response to the messages and display it
    st.session_state.messages.append({"role": "assistant", "content": response['output']})
    with st.chat_message("assistant"):
        st.markdown(response['output'])
