#the idea is to create a advanced chat bot which retaines memory based on sesion and revomes memory after a certain amount of time.
#We will use chatGroq from langchain ,because it gives beeter result campared to huggingface models.

#all the required libariries are mentioned in requirements.txt file

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()
API_KEY = os.getenv("Api_key")

chatbot = ChatGroq(temperature=0,
    model="llama3-70b-8192",
    api_key=API_KEY
    )

from langchain_core.messages import HumanMessage
messagesToTheChatbot = [
    HumanMessage(content="My favourite color is steel blue."),
]

def extract_message(message)->str:
    components = {}
    for meta,text in message:
        components[meta] = text
    return components['content']


#print(extract_message(chatbot.invoke(messagesToTheChatbot)))

# till now our chatbot only behaves like a simple model it dosent retain memory

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

chatbotMemory ={}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chatbotMemory:
        chatbotMemory[session_id] = ChatMessageHistory()
    return chatbotMemory[session_id]


chatbot_with_message_history = RunnableWithMessageHistory(
    chatbot, 
    get_session_history
)

session1 = {"configurable": {"session_id": "001"}}

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="My favorite color is steel blue.")],
    config=session1,
)

#print(responseFromChatbot.content)

responseFromChatbot_1 = chatbot_with_message_history.invoke(
    [HumanMessage(content="I dont have any experience with it ,I just simply like it the way it looks")],
    config= session1)

#print(responseFromChatbot_1.content)

print(chatbotMemory)

