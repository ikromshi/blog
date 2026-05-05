---
layout: notebook
title: "Model Context Protocol"
project: agentic-patterns
order: 10
---

## Model Context Protocol

### Overview
The Model Context Protocol is a universal adapter that allows any LLM to plug into any external system, database, or tool without a custom integration for each one. It's an open standard designed to standardize how various LLM models communicate with external applications, data sources, and tools. It's a universal connection mechanizm that simplifies how LLMs obtain context, execute actions, and interact with various systems.

MCP operates on a client-server architecture. It defines how different elements - data (referred to as sources), interactive templates (which are essentially prompts), and actionable functions (known as tools) - are exposed my an MCP server. These are then consumed by an MCP client, which could be an LLM host application or an AI agent itsef.

MCP is a contract for an "agentic interface", and its effectiveness depends heavily on the design of the underlying APIs it exposes. There is a risk that developers simply wrap pre-existing, legaxy APIs without modification, which can be suboptimal for an agent. For example, if a ticketing system's API only allows retrieving full ticket details one by one, an agent asked to summarize high-priority tickets will be slow and inaccurate like at high voluments. To be truly effective, the underlying API should be improved with deterministic features like filtering and sorting to help the non-deterministic agent work efficiently. This highlights that agents don't magically replace deterministic workflows; they often require stronger deterministic support to succeed.

Furthermore, MCP can wrap an API whose input or output is still not inherently understandable by the agent. An API is only useful if its data format is agent-friendly, a guarantee that MCP itself does not enforce. For instance, creating an MCP server for a document store that returns files as PDFs is mostly useless if the consuming agent cannot parse PDF content. The better approach would be to first create an API that returns a textual version of the document (e.g., MD) which the agent can actually read and process. This means that devs must consider not just the connection, but he nature of the data being exchanged to ensure true compatibility.


### MCP vs. Tool Function Calling
While both MCP and Tool Calling serve to extend LLM capabilities beyond text generation, they differ in their approach and level of abstraction.

Tool calling can be thought of as a direct request from an LLM to a specific, pre-defined tool or function. Note that in this context we use "tool" and "function" interchangeably. This interaction is characterized by a oe-to-one communication model, where the LLM formats a request based on its understanding of a user's intent requiring external action. The application code the executes this request and returns the result to the LLM. This process is often proprietary and varies across different LLM providers.

In contrast, MCP operates as a standardized interface for LLMs to discover, communicate with, and utilize external capabilities. It functions as an open protocol that facilitates interaction with a wide range of tools and systems, aiming to establish an ecosystem where any compliant tool can be accessed by the complian LLM. This fosters interoperability, composability and reusability across different systems and implementations. This strat allows us to bring disparate and legacy services into a modern ecosystem simply by wrapping them in an MCP-compliant interface.

Here's a breakdown of the distinctions:

<div align="center">

| Feature | Tool Function Calling | MCP |
| :--------: | :-------- | :-------- |
| **Standardization** | Proprietary and vendor-specific. The format and implementation differ across LLM provers. | An open, standardized protocol, promoting interoperability between different LLMs and tools. |
| **Scope** | A direct mechanism for an LLM to request the execution of a specific, predefined function. | A broader framework for how LLMs and external tools discover and communicate with each other. |
| **Architecture** | A one-to-one interaction between the LLM and the application's tool-handling logic. | A client-server architecture where LLM-powered applications (clients) can conntect to an utilize various MCP servers (tools) |
| **Discovery** | The LLM is explicitly told which tools are available within the context of a specific conversation. | Enables dynamic discovery of available tools. An MCP client can query a server to see what capabilities it offers. |
| **Reusability** | Tool integrations are often tightly coupled with the specific application and LLM being userd. | Promotes the development of reusable, standalone "MCP servers" that can be accessed by any compliant application. |

</div >


The MCP interaction flow is as follows:
1. Discovery: The MCP Client, on behalf of the LLM, queries an MCP Server to ask what capabilities if offers. The server responds with a manifest listing its available tools (e.g., `send_email`), resources (e.g., `customer_database`), and prompts.

2. Request Formulation: The LML determiens that it needs to use of the discovered tools. For instance, it decides to send an email. It formulates a request, specifying the tool to use (`send_email`) and the necessary parameters (recipient, subject, body).

3. Client Communication: The MCP Client takes the LLM's formulated request and sends it as a standardized call to the appropriate MCP Server.

4. Server Execution: The MCP Server receives the request. It authenticates the client, validates the request, and then executes the specified action by interfacing with the underlying software (e.g., calling the `send()` function of an email API).

5. Response and Context Update: After execution, the MCP Server sends a standardized response back to the MCP Client. This response indicates whether the action was successful and includes any relevant output (e.g., a confirmation ID for the sent email). The client then passes this result back to the LLM, updating its context and enabling it to proceed with the next step of its task.


### Practical Applications & Use Cases
- Database Integration: MCP allows LLMs and agents to seamlessly access and interact with structured data in databases. For instance, using the MCP Toolbox for Databases, an agent can query Google BigQuery datasets to retrieve real-time infromation, generate reports or update records, all driven by natural language commands.

- Generate Media Orchestration: MCP enables agents to integrate with advanced generative media services. Through MCP Tools for Genmedia Services, an agent can orchestrate workflows involving Google's Imagen for image generation, Google's Veo for video creation, Google's Chirp 3 HD for realistic voices, or Google's Lyria for music composition, allowing for dynamic content creation within AI applications.

- External API Interaction: MCP provides a standardized way for LLMs to call and receive responses from any external API. This means an agent can fetch live weather data, pull stock prices, send emails, or interact with CRM systems, extending its capabilities for bayond its core language model.

- Reasoning-Based Information Extraction: Leveraging an LLM's strong reasoning skills, MCP enables effective, query-dependent information extraction that surpasses conventional search and retrieval systems. Instead of a traditional search tool returning an entire document, an agent can analyze the text and extract the precise clause, figure, or statement that directly answers a user's complex question.

- Custom Tool Development: Developers can build custom tools and expose them via an MCP server (e.g., using FastMCP). This allows specialized internal functions or proprietary systems to be made available to LLMs and other agents in a standardized, easily ocnsumable format, without needing to modify the LLM directly.

- Standardized LLM-to-Application Communication: MCP ensures a consistent communication layer between LLMs and the application they interact with. This reduces integration overhead, promotes interoperability between different LLM providers and host applications, and simplifies the development of complex agentic systems.

- Complex Workflow Orchestration: By combining various MCP-exposed tools and data sources, agents can orchestrate highly complex, multi-step workflows. A agent could, for example, retrieve customer data from a database, generate a personalized marketing image, draft a trailored email, and then send it, all by interacting with different MCP services.

- IoT Device Control: MCP can facilitate LLM interaction with IoT devices. An agent could use MCP to send commands to smart home appliances, industrial sensors, or robotics, enabling natural language control and automation of physical systems.

- Financial Services Automation: In financial services, MCP could enable LLMs to interact with various financial data sources, trading platforms, or compliance systems. An agent might analyze market data, execute trades, generate personalized finanfial advice, or automate regulatory reporting, all while maintaining secure and standardized communication.


In short, MCP enables agents to access real-time information from DBs, APIs, and web resources. It also allows agents to perform actions like sending emails, updating records, controlling devices, and executing complex tasks by integrating and processing data from various sources.

### Hands-On Code Example

```python
import json
from dotenv import get_key
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

API_KEY = get_key("../.env", "OPENAI_API_KEY")

async def main():
    # --- Model Configuration
    model = ChatOpenAI(
        model="gpt-5.4",
        temperature=0.1,
        api_key=API_KEY
    )

    
    # --- MCP Connections
    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "stdio", # Local subprocess communication
                "command": "python",
                # Absolute path you math_server.py file
                "args": ["/Users/ikromshi/coding/ml/agentic-design-patterns/chapter-2/01-mcp/01-math-mcp-server.py"]
            },
            "weather": {
                "transport": "http", # HTTP-based remote server
                # Need to start weather server on port 8000
                "url": "http://localhost:8000/mcp"
            }
        }
    )

    tools = await client.get_tools()
    agent = create_agent(
        model,
        tools,
    )


    # --- Execution through MCP
    math_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what is (3 + 5) * 12?"}]}
    )
    weather_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what's the weather in nyc?"}]}
    )

    math_final_message = math_response["messages"][-1]
    weather_final_message = weather_response["messages"][-1]

    print(f"math_response: {math_final_message.content}")
    print(f"weather_final_message: {weather_final_message.content}")
    
    # pretty_json = json.dumps(weather_response, indent=4, default=str)
    # print(pretty_json)


if __name__ == "__main__":
    await main()
```

```
math_response: 96
weather_final_message: It’s always sunny in New York.
```

