---
layout: notebook
title: "Memory Management"
project: agentic-patterns
order: 8
---

## Memory Management

### Overview
In agentic systems, memory refers to an agent's ability to retrain and utilize information from past interactions, observations, and learning experiences. This capability allows agents to make informed decisions, maintain conversational context, and improve over time. Agent memory is generally categorized into two main types:

- **Short-Term Memory (Contextual Memory)**: Similar to working memory, this holds information currently being processed or recently accessed. For agents using LLMs, short-term memory primarily exists within the context window. This window contains recent messages, agent replies, tool usage results, and agent reflections from the current interaction, all of which inform the LLM's subsequent responses and actions. The context window has a limited capacity, restricting the amount of recent information an agent can directly access. Efficient short-term memory management involves keeping the most relevant information within this limited space, possibly through techniques like summarizing older conversation segments or emphasizing key details. The advent of models within "long context" windows simply expands the size of this short-term memory, allowing more information to be held within a single interaction. However, this context is still ephemeral and is lost once the session concludes, and it can be costly and inefficient to process every time. Consequently, agents require separate memory types to achieve true persistence, recall information from past interactions, and build a lasting knowledge base.

- **Long-Term Memory (Persistent Memory)**: This acts as a repository for information agents need to retain across various interactions, tasks, or extended periods, akin to long-term knowledge bases. Data is typically stored outside the agent's immediate processing environment, often in databases, knowledge graphs, or vector databases. In vector databases, information is converted into numerical vectors and stored, enabling agents to retrieve data based on semantic similarity rather than exact keyword matches, a process known as semantic search. When an agent needs information from long-term memory, it queries the external storage, retrieves relecant data, and integrates it into the short-term context for immediate use, thus combining prior knowledge with the current interaction.

### Practical Applications & Use-Cases
- **Chatbots and Conversational AI**: Maintaining conversation flow relies on short-term memory. Chatbots require remembering prior user inputs to provide coherent responses. Long-term memory enables chatbots to recall user preferences, past issues, or prior discussions, offering personalized and continuous interactions.

- **Task-Oriented Agents**: Agents managing multi-step tasks need short-term memory to track previous steps, current progress, and overall goals. This information might reside in the task's contexxt or temporary storage. Long-term memory is crucial for accessing specific user' related data not in the immediate context.

- **Personalized Experiences**: Agents offering tailored interactions utilize long-term memory to stoer and retrieve user preferences, past behaviors, and personalized information. This allows agents to adapt their responses and suggestions.

- **Learning and Improvement**: Agents can refine their performance by learning from past interactions. Successful strategies, mistakes, and new information are stored in long-term memory, facilitating future adaptations. Reinforcement learning agents store learned strategies or knowledge in this way.

- **Information Retrieval (RAG)**: Agents designed for answering questions access a knowledge base, their long-term memory, often implemented within Retrieval Augmented Generation (RAG). The agent retrieves relevant documentatios or data to inform its responses.

- **Autonomous Systems**: Robots or self-driving cars require memory for maps, routes, object locations, and learned behaviors. This involves short-term memory for immediate surroundings and long-term memory for general environmental knowledge.

### Hands-On Code Example

Before getting started with the code, let's cover some technicalities on Memory Management with LangChain and LangGraph:
- **Short-Term Memory**: This is thread-scoped, meaning it tracks the ongoing converstaion within a single session or thread. It provides immediate context, but a full history can challenge an LLM's context window, potentially leading to errors or poor performance. LangGraph manages short-term memory as part of the agent's state, which is persisted via a checkpointer, allowing a thread to be resumed at any time.

- **Long-Term Memory**: This stores user-specific or application-level data across sessions and is shared between conversational threads. It is saved in custom "namespaces" and can be recalled at any time in any thread. LangGraph provides stores to save and recall long-term memories, enabling agents to retain knowledge indefinitely.

**ChatMessageHistory** -- Manual Memory Management. For direct and simple control over a conversation's history outside of a formal chain, the `ChatMessageHistory` class is ideal. It allows for the manual tracking of dialogue exchanges.

```python
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory

# Init the history object
history = ChatMessageHistory()

# Add user and AI messages
history.add_user_message("I'm heading to Samarkand next week.")
history.add_ai_message("Great!. It's a fantastic historical city.")

# Access the list of messagse
print(history.messages)
```

```
[HumanMessage(content="I'm heading to Samarkan next week.", additional_kwargs={}, response_metadata={}), AIMessage(content="Great!. It's a fantastic historical city.", additional_kwargs={}, response_metadata={}, tool_calls=[], invalid_tool_calls=[])]
```

**ConversationBufferMemory** -- Automated Memory for Chains. For integrating memory directly into chains, `ConversationalBufferMemory` is a common choice. It holds a buffer of the conversation and makes it available to your prompt. Its behavior can be customized with two key parameters:
- `memory_key`: A string that specifies the variable name in your prompt that will hold the chat history. It defaults to "history".
- `return_messages`: A boolean that dictates the format of the history.
    - If `False` (the default), it returns a single formatted string, which is ideal for standard LLMs.
    - If `True`, it returns a list of message objects, which is the recommended format for Chat Models.

```python
from langchain_classic.memory import ConversationBufferMemory

# Init memory
memory = ConversationBufferMemory()

# Save a conversation turn
memory.save_context(
    {"input": "What's the weather like?"},
    {"output": "It's sunny today."}
)

# Load the memory as a string
print(memory.load_memory_variables({}))
```

```
{'history': "Human: What's the weather like?\nAI: It's sunny today."}
```

```python
from dotenv import get_key

from langchain_openai import OpenAI
from langchain_classic.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferMemory


# --- Define LLM and prompt
API_KEY = get_key("../.env", "OPENAI_API_KEY")
llm = OpenAI(model="gpt-5.4-nano", api_key=API_KEY)

template = """You are a helpful travel agent.
Previous conversation: {history}
New question: {question}
Response:
"""

prompt = PromptTemplate.from_template(template)


# --- Configure Memory
# the memory_key "history" matches the variable in the prompt
memory = ConversationBufferMemory(memory_key="history")


# --- Build the Chain
conversation = LLMChain(llm=llm, prompt=prompt, memory=memory)


# --- Run the Conversation
response = conversation.predict(question="I want to book a flight.")
print(response)

response = conversation.predict(question="My name is Ikrom, by the way.")
print(response)

response = conversation.predict(question="What was my name again?")
print(response)
```

```python
"""
For improved effectiveness with chat models, it is recommended to use a structured 
list of message objects by setting `return_messages=True`
"""

from langchain_openai import ChatOpenAI
from langchain_classic.chains.llm import LLMChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from dotenv import get_key

# --- Define chat model and prompt
API_KEY = get_key("../.env", "OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-5.4-nano", api_key=API_KEY)
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a friendly assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{question}")
])


# --- Configure memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# --- Build the chain
conversation = LLMChain(llm=llm, prompt=prompt, memory=memory)


# --- Run the Conversation
response = conversation.predict(question="Hi, I'm Jane.")
print(response)

response = conversation.predict(question="Do you remember my name?")
print(response)
```

```
Hi Jane! 👋 Nice to meet you. How can I help you today?
Yes—your name is Jane.
```

### Short-Term Memory: LangChain Docs

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

agent = create_agent(model=llm, checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "user-ikromshi-session-1"}}

agent.invoke({"messages": {"role": "user", "content": "Hi my name is Ikrom Numonov"}}, config)
result = agent.invoke({"messages": {"role": "user", "content": "What's my full name?"}}, config)

print(result["messages"][-1].content)


# can also connect to a PG database to persist memory
```

```
You told me your name is **Ikrom Numonov**. I don’t have any additional information (like a middle name) beyond that.
```

