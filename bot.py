from typing import TypedDict, List, Annotated, Sequence
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage, SystemMessage
import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from operator import add as add_messages
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import load_tools

load_dotenv()

#chatbot
{

# class AgentState(TypedDict):
#     messages: List[HumanMessage]

# client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# llm = ChatOpenAI(
#     model= "gpt-3.5-turbo",
#     temperature=0
# )

# def chatbot (state: AgentState) -> AgentState:
#     response = llm.invoke(state["messages"])
#     print(f"\nAI: {response.content}")
#     return state 

# workflow = StateGraph(AgentState)

# workflow.add_node("chatbot", chatbot)

# workflow.add_edge(START, "chatbot")
# workflow.add_edge("chatbot", END)

# app = workflow.compile()

# user_input = input("Enter a query: ")
# while user_input != "exit":
#     result = app.invoke({"messages": [HumanMessage(content=user_input)]})
#     print(result)
#     user_input = input("Enter a query: ")
}

#agent
# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]

# client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# llm = ChatOpenAI(
#     model= "gpt-3.5-turbo",
#     temperature=0
# )

# @tool 
# def addition (a:int, b:int):
#     """This is the addition function that adds 2 numbers"""
#     return a + b
# @tool 
# def substraction (a:int, b:int):
#     """This is the substraction function that substract 2 numbers"""
#     return a - b
# @tool 
# def multiplication (a:int, b:int):
#     """This is the multiplication function that multiply 2 numbers"""
#     return a * b
# @tool 
# def division (a:int, b:int):
#     """This is the division function that divide 2 numbers"""
#     return a / b

# search_tool = DuckDuckGoSearchRun()

# tools = [search_tool]

# llm_with_tools = llm.bind_tools(tools)

# def llmCall(state:AgentState) -> AgentState:
#     system_prompt = SystemMessage(
#         content= "You are an intelligent AI assistant. Please answer my query."
#     )
#     response = llm_with_tools.invoke([system_prompt] + state["messages"])
#     return {"messages": [response]}

# def decision_node(state:AgentState):
#     messages = state["messages"]
#     last_message = messages[-1]
#     if not last_message.tool_calls:
#         return "end"
#     else:
#         return "continue"
    
# workflow = StateGraph(AgentState)

# workflow.add_node("agent", llmCall)
# tool_node = ToolNode(tools=tools)
# workflow.add_node("tools", tool_node)

# workflow.add_edge(START, "agent")
# workflow.add_conditional_edges(
#     "agent",
#     decision_node,
#     {
#         "continue": "tools",
#         "end": END
#     },
# )

# workflow.add_edge("tools", "agent")

# app = workflow.compile()

# def print_stream(stream):
#     for s in stream:
#         message = s["messages"][-1]
#         if isinstance(message, tuple):
#             print(message)
#         else:
#             message.pretty_print()

# inputs = {"messages": [("user", "what is agentic AI?")]}
# print_stream(app.stream(inputs, stream_mode="values"))

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

llm = ChatOpenAI(
    model= "gpt-3.5-turbo",
    temperature=0
)

embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-small",
)

pdf_loader = PyPDFLoader("") 

pages = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

pages_split = text_splitter.split_documents(pages)

#len(pages_split)

vectorstore = Chroma.from_documents(
    documents = pages_split,
    embeddings = embeddings,
    persist_directory = "./vectorstore",
    collection_name = "data"
)

retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":3}
)

@tool 
def retriever_tool(query: str) -> str:
    """This tool searches and return the information related user questions."""
    docs = retriever.invoke(query)
    if not docs: 
        return "I fount no relavent infomation"
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")

    return "\n\n".join(results)

search_tool = DuckDuckGoSearchRun()

tools = [retriever_tool, search_tool]
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    message: Annotated[Sequence[BaseMessage],add_messages]

def Should_coutinue(state:AgentState):
    """Check if the last message contain tool calls."""
    result = state["messages"][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_call) > 0

system_prompt = """
You are a supportive, emotionally intelligent chatbot specializing in providing information and support for social phobia.
You are a supportive, emotionally intelligent chatbot designed to provide comfort and genuine help. Your primary goal is to make users feel better after talking with you, not worse.

CRITICAL GUIDELINES:
1. PROVIDE ACTUAL HELP: When users express discomfort or negative emotions, offer genuine comfort and practical suggestions rather than just asking more questions.

2. RESPECT "I DON'T KNOW" RESPONSES: Never ask the same question again if a user says "I don't know" or similarly indicates uncertainty. Instead, offer a supportive statement and a different approach.

3. EMOTIONAL COMFORT FIRST: Your main purpose is to help users feel calmer and more at ease. Every response should move toward making them feel better.

4. SPECIALIST FOCUS: You are a specialist in social phobia. Offer information and support related to social phobia when appropriate, but never diagnose or assume a user's condition. Let the user guide the depth of discussion on their personal experiences.

5. BRIEF AND HELPFUL: Keep responses concise (2-3 sentences) but make them substantive and helpful, not just questions.

6. OFFER PRACTICAL SUGGESTIONS: When users express negative feelings, offer a simple, practical tip they could try, without being pushy.

7. AVOID REPETITION: Track what you've already said and never repeat the same questions or advice.

8. GENUINE CONVERSATION: Be warm and conversational, like a supportive friend who actually helps rather than just asking questions.

Remember: After interacting with you, users should feel calmer, supported, and like they received actual help.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools}

def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state["message"])
    messages = [SystemMessage(content = system_prompt)] + messages
    message = llm_with_tools.invoke(messages)
    print(message)
    return {'message': [message]}



