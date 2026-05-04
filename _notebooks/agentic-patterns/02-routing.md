---
layout: notebook
title: "Routing"
project: agentic-patterns
order: 2
---

## Routing
### Overview
While sequential processing with Prompt Chaining allows for deterministic, linear workflows, oftentimes, there is a need for adaptive responses. Routing refers to the capacity for dynamic decision-making, which governs the flow of control to different specialized functions, tools or sub-processes.

For instance, an agent designed for customer inquiries, when equipped with a arouting function, can first classify an incoming query to determine th euser's intent. Based on this classification, it can then direct the query to a specialized agent for direct question-asnwering, a database retrieval tool for account information, or an escalation procedure for complex issues, rather than defaulting to a single, predetermined response pathway. Therefore, a more sophisticated agent using routing could:
1. Analyze the user's query.

2. Route the query based on its _intent_:
    * If the intent is "check order status", route to a sub-agent or tool chain that interacts with the order database.
    * If the intent is "product information", route to a sub-agent or chain that searches the product catalog.
    If the intent is "technical support", route to a different chian that accesses troubleshooting guides or escalates to a human.
    * If the intent is unclear, route to a clarification sub-agent or prompt chain.


### Routing Methods
* **LLM-Based Routing**: The language model itself can be prompted to analyze the input and output a specific identifier or instruction that indicates the next step or destination.

* **Embedded-based Routing**: The input query can be converted into a vector embedding. This embedding is then compared to embeddings representing different routes or capabilities. The query is routed to the route whose embedding is most similar. This is useful for semantic routing.

* **Rule-based Routing**: This involves involves using predefined rules or log (e.g., if-else statements, switch cases) based on keywords, patterns, or structured data extracted from the input. This can be faster and more deterministic than LLM-based routing, but is less flexible for handling nuanced or novel inputs.

* **ML Model-based Routing**: Employs a discriminative model, such as a classifier, that has been specifically trained on a small corpus of labeled data to perform a routing task. This is distinct from LLM-based routing because the decision-making component is not a generative model executing a prompt at inference time. Instead the routing logic is encoded within the fine-tuned model's learned weights.

### Hands-On Code Example
The following example shows a simple agent-like system using LangChain and Google's Generative AI. It sets up a "coordinator" that routes user requests to different simulated "sub-agent" handlers based ont he requests intent (booking, information, or unclear). The system uses a language model to classify the request and then delegates it to the appropriate handler function, simulating a basic delegation pattern often seen in multi-agent architectures.

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch

from dotenv import get_key

# set GOOGLE_API_KEY
GOOGLE_KEY = get_key("../.env", "GOOGLE_API_KEY")


# ---- Configuration
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0, api_key=GOOGLE_KEY)
    print(f"Language model initialized: {llm.model}")
except Exception as e:
    print(f"Error initializing language mode: {e}")
    llm = None


# ---- Simulated Sub-Agent Handlers
def booking_handler(request: str) -> str:
    """Sumulates the Booking Agent handling a request."""
    print("\n--- DELEGATING TO BOOKING HANDLER ---")
    return f"Booking Handler processed request: '{request}'. Result: Simulated booking action."


def info_handler(request: str) -> str:
    """Simulates the Info Agent handling a request."""
    print("\n--- DELEGATING TO INFO HANDLER ---")
    return f"Info Handler processed request: '{request}'. Result: Simulated information retrieval."


def unclear_handler(request: str) -> str:
    print("\n--- HANDLING UNCLEAR REQUEST")
    return f"Coordinator could not delegate request: '{request}'. Please clarify."


# ---- Coordinator Router Chain: decides which handler to delegate to.
coordinator_router_prompt = ChatPromptTemplate.from_messages([
    (
        "system", 
            """Analyze the user's request and determine which specialist handler should process it.
            - If the request is related to booking flights or hotels,
            output 'booker'.
            - If the request asks for general information, output 'info'.
            - If the request is unclear or doesn't fit either category,
            output 'unclear'.
            ONLY output one word: 'booker', 'info', or 'unclear'."""
    ),
    ("user", "{request}")
])

if llm:
    coordinator_router_chain = coordinator_router_prompt | llm | StrOutputParser()


# ---- Delegation Logic: use RunnableBranch to route based on th erouter chain's output.
branches = {
    "booker": RunnablePassthrough.assign(
        output=lambda x: booking_handler(x['request']['request'])
    ),
    "info": RunnablePassthrough.assign(
        output=lambda x: info_handler(x['request']['request'])
    ),
    "unclear": RunnablePassthrough.assign(
        output=lambda x: unclear_handler(x['request']['request'])
    )
}


# Create a RunnableBranch. It takes the output of the router chain and 
# routes the original input ('request') to the corresponding handler.
delegation_branch = RunnableBranch(
    (lambda x: x["decision"].strip() == "booker", branches["booker"]),
    (lambda x: x["decision"].strip() == "info", branches["info"]),
    branches["unclear"]
)


# Combine the router chain and the delegation branch into a single runnable.
# The router's chain's output ('decision') is passed along with the original input ('request')
# to the delegation_branch
coordinator_agent = {
    "decision": coordinator_router_chain,
    "request": RunnablePassthrough()
} | delegation_branch | (lambda x: x['output']) # get the final output



# ---- Example Usage
def main():
    if not llm:
        print("\nSkipping execution login due to LLM initialization failure.")
        return

    print("--- Running with a booking agent ---")
    request_a = "Book me a flight to London."
    result_a = coordinator_agent.invoke({"request": request_a})
    print(f"Final Result A: {result_a}")

    print("--- Running with an info request ---")
    request_b = "What is the capital of Italy?"
    result_b = coordinator_agent.invoke({"request": request_b})
    print(f"Final Result B: {result_b}")

    print("--- Running with an unclear request ---")
    request_c = "Cat"
    result_c = coordinator_agent.invoke({"request": request_c})
    print(f"Final Result C: {result_c}")

if __name__ == "__main__":
    main()
```

```
Language model initialized: gemini-2.5-flash-lite
--- Running with a booking agent ---

--- DELEGATING TO BOOKING HANDLER ---
Final Result A: Booking Handler processed request: 'Book me a flight to London.'. Result: Simulated booking action.
--- Running with an info request ---

--- DELEGATING TO INFO HANDLER ---
Final Result B: Info Handler processed request: 'What is the capital of Italy?'. Result: Simulated information retrieval.
--- Running with an unclear request ---

--- HANDLING UNCLEAR REQUEST
Final Result C: Coordinator could not delegate request: 'Cat'. Please clarify.
```

### Takeaways
* Routing enables agenst to make dynamic decisions about th enext step in a workflow based on conditions.
* It allows agents to handle diverse inputs and adapt their behavior, moving beyond linear execution.
* Routing logic can be implemented using LLMs, rule-based systems, or embedding similarity.
* Frameworks like LangGraph and Google ADK provide structured ways to define and manage routing within agent workflows, albeit with different architectural approaches.

