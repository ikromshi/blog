---
layout: notebook
title: "Tool Use"
project: agentic-patterns
order: 5
---

## Tool Use
### Overview
For agents to be truly useful and interact with the real world or external systems, they need the ability to use Tools. The Tool Use pattern, often implemented through a mechanism called Function Calling, enables an agent to interact with external APIs, databases, services, or even execute code. It allows the LLM at the core of the agent to decide when and how to use a specific internal function based on the user's request or the current state of the task.

The process typically involves:
1. **Tool Definition**: External functions or capabilities are defined and described to the LLM. This description includes the function's purpose, its name, and the parameters it accepts, along with their types and descriptions.

2. The **LLM Decision**: The LLM receives the user's request and the available tool definitions. Based on its understanding of the request and the tools, the LLM decides if calling one or more tools is necessary to fulfill the request.

3. Function **Call Generation**: If the LLM decides to use a tool, it generates a structured output (often a JSON object) that specifies the name of the tool to call and the arguments (parameters) to pass to it, extracted from the user's request.

4. **Tool Execution**: The agentic framework or orchestration layer intercepts this structured output. It identifies the requested tool and executes the actual external function with the provided arguments.

5. Observation**/Result**: The output or result from the tool execution is returned to the agent.

6. LLM Processing (Optional but **common)**: The LLM receives the tool's output as context and uses it to formulate a final response to the user or decide on the next step in the workflow (which might involve calling another tool, reflecting, or providing a final answer).

This pattern is fundamental because it rbeaks the limitations of the LLM's training data and allows it to access up-to-date information, perform calculations it can't do itnernally, interact with user-specific data, or trigger real-world actions.


### Practical Applications and Use Cases
The Tool Use pattern is applicable in virually any scenario where an agent needs to go beyond generaitng text to perform an action or retrieve specific, dynamic information:

#### 1. Information Retrieval from External Sources
**Use Case**: A weather agent.
- **Tool:** A weather API that takes a location and returns the current weather conditions.
- **Agent Flow**: User asks, "What's the weather in London?", the LLM identifies the need for the weather tool, calls teh tool with "London", and tool returns data, LLM formats the data into a user-friendly response.

#### 2. Interacting with Databases and APIs
**Use Case**: An e-commerce agent.
- **Tool:** API calls to check product inventory, get order status, or process payments.
- **Agent Flow**: User asks: "Is product X in stock?", the LLM calls the inventory API, tool returns stock count, LLM teslls the user the stock status.

#### 3. Performing Calculations and Data Analysis
**Use Case**: A financial agent.
- **Tool:** A calculator function, a stock market data API, a spreadsheet tool.
- **Agent Flow**: User asks "What's the current price of AAPL and calculate the potential profit if bought 100 shares at $150?", LLM calls stock API, gets current price, then calls calculator tool, gets result, formats response.

#### 4. Sending Communications
**Use Case**: A personal assistant agent.
- **Tool:** An email sending API.
- **Agent Flow**: User says, "Send an email to John about the meeting tomorrow.", LLM calls an email tool with the recipient, subject, and body extracted from the request.

#### 5. Executing Code
**Use Case**: A coding assistant agent.
- **Tool:** A code interpreter.
- **Agent Flow**: User provides a Python snippet and asks, "What does this code do?", LLM uses the interpreter tool to run the code and analyze its output.

#### 6. Controlling Other Systems or Devices
**Use Case**: A smart home agent.
- **Tool:** An API to control smart lights.
- **Agent Flow**: User says, "Turn off the living rooms lights." LLM calls teh samrt home tool with the command and target device.

### Hands-On Code Example
The implementation of tool sue within LangChain is a two-stage process. Initially, or or more tools are defined, typically by encapsulating existing Python functions or other runnable components. Subsequently, these tools are bound to a language model, thereby granting the model the capability to generate a structured tool-use request when it determines that an external function call is required to fulfill a user's query.

The following example shows this principle by first defining a simple funtion to simulate an information retrieval tool. Following this, an agent will be constructed and configured to leverage this tool in response to user input.

```python
import asyncio
import nest_asyncio
from dotenv import get_key

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool as langchain_tool
from langchain.agents import create_agent


# --- Configuration
GEM_API = get_key("../.env", "GOOGLE_API_KEY")

try:
    # a model with function/tool calling capabilities
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=GEM_API)
    print(f"Language model initialized: {llm.model}")
except Exception as e:
    print(f"Error initializing language model: {e}")
    llm = None


# --- Define a Tool
@langchain_tool
def search_information(query: str) -> str:
    """
    Provides factual information on a given topic. Use this tool to find answers to phrases
    like 'capital of France' or 'weather in London?'.
    """
    print(f"\n--- Tool Called: search_infromation with query: '{query}' ---")

    # Simulate a search tool with a dictionary of predefined results.
    simulated_results = {
        "weather in london": "The weather in London is currently cloudy with a temperature of 15C.",
        "capital of france": "The capital of France is Paris.",
        "population of earth": "The estimated population of Earth is around 8 billion people.",
        "tallest mountain": "Mount Everest is the talles mountain above sea level.",
        "default": f"Simulated search result for '{query}': No specific information found, but the topic seems interesting."
    }
    result = simulated_results.get(query.lower(), simulated_results["default"])
    print(f"--- TOOL RESULT: {result} ---")
    return result


tools = [search_information]


# --- Create a Tool-Calling Agent
if llm:
    # Create the agent, binding the LLM, tools, and prompt together.
    agent = create_agent(model=llm, tools=tools, system_prompt="You are a helpful agent.")


async def run_agent_with_tool(query: str):
    """Invokes the agent executor with a query and prints the final response."""
    print(f"\n--- Running agent with query: '{query}' ---")
    try:
        response = await agent.ainvoke({"messages": [("human", query)]})
        final_message = response["messages"][-1]
        print("\n--- Final Agent Response ---")
        print(final_message.content)
    except Exception as e:
        print(f"\nAn error occurred during agent execution: {e}")


async def main():
    """Runs all agent queries concurrently"""
    tasks = [
        run_agent_with_tool("What is the capital of France?"),
        run_agent_with_tool("What's the weather like in London?"),
        run_agent_with_tool("Tell me something about dogs."), # should trigger the default tool response
    ]

    await asyncio.gather(*tasks)


# nest_asyncio.apply()
asyncio.run(main())
```

```
Language model initialized: gemini-2.5-flash

--- Running agent with query: 'What is the capital of France?' ---

--- Running agent with query: 'What's the weather like in London?' ---

--- Running agent with query: 'Tell me something about dogs.' ---

--- Tool Called: search_infromation with query: 'capital of France' ---
--- Tool Called: search_infromation with query: 'weather in London' ---
--- TOOL RESULT: The weather in London is currently cloudy with a temperature of 15C. ---

--- TOOL RESULT: The capital of France is Paris. ---

--- Tool Called: search_infromation with query: 'dogs' ---
--- TOOL RESULT: Simulated search result for 'dogs': No specific information found, but the topic seems interesting. ---

--- Final Agent Response ---
The weather in London is currently cloudy with a temperature of 15C.

--- Final Agent Response ---
[{'type': 'text', 'text': "I couldn't find specific information about dogs. Would you like me to search for something more specific, like a particular breed or aspect of dogs?", 'extras': {'signature': 'Cv0CAQw51sdywO9Cd79jcOk56JrJ/Vf+ORC4I996ZOBsn7y5A0d9HN2jhaoDzzQ1y+7dL3+uZO/ENOOjwJjqgGuJlDSljsRUlEXOOVJoSQfmfqmV3I4DaICMoVfTHbp8pPCp92kzfDdDTtI+/Iyohr5cvP4m8s5q/xdiHsDZ0fOKRSCjggdB2aTyywcpUshPks/c/fxiW96yU6VKONmzREOyLH+YK0SaNPLfVTamkqw9fSgpvB8GIkHz7r1gqHTBqsUuOkXTXErx03wK1TTfrFD0mj9npZkr+9Y0MuU+BPSWG/VZ/Rs9NM8WZ8BsSwm7nEIM9AmUzfOEDH1d99EAapjbrJYqyPCE6xb1VMO7AS+lLKc6fHKxfrz4ngd65pgK4sQ7dsHDP16f7yzpZdjbQMHeT3jidNbgL+Ai3VYDyoMFjQXk6UDyuANtFLBRFFy8DsP7I+z26igtD2fm6jqhP0fa7BrVXy5kuC4H9Re+rsA+CIsH3sDCGYtiDWUAANqp'}}]

--- Final Agent Response ---
The capital of France is Paris.
```

