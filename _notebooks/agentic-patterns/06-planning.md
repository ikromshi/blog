---
layout: notebook
title: "Planning"
project: agentic-patterns
order: 6
---

## Planning
### Overview
At its core, Planning is the ability for an agent or a system of agents to formulate a sequence of actions to move from an initial state towards a goal state. In the context of AI, it's helpful to think of a planning agent as a specialist to whom you delegate a complex goal. The agent's core task is to break down the "how" of a task when you give it the "what". It must first understand the initial state of the task (e.g., budget, number of participants, desired dates) and teh goal state (a successfully booked offsite), and then discover the optimal sequence of actions to connect them. The plan is not known in advance -- it is created in response to the request.

A crucial characteristic of this process is adaptability.An initial plan is just a starting point, not a rigid plan. The agent's real power is its ability to incorporate new information and steer the project around obstacles. When something unexpected comes up, it should register the new constraint, re-evaluate its options, and formulate a new plan, perhaps by suggesting alternative courses of action.

### Practical Applications and Use Cases
The Planning pattern is a core computational process in autonomous systems, enabling an agent to sythesize a sequence of actions to achieve a specified goal, particularly within dynamic or complex environments. The pattern is commonly applied in the following cases:

- Domains such as Procedural Task Automation, where planning is used to orchestrate complex workflows (onboarding a new employee).

- Robotics and autonomous navigation where planning is fundamental for state-space traversal.

- Structured Information Synthesis. When tasked with generating a complex output like a research report, an agent can formulate a plan that includes distinct phrases for information gathering, data summarization, content structuring, and iterative refinement.

### Hands-On Code
The following example shows an agent formulating a multi-step plan to address a complex query and then executing that plan sequentially.

```python
import os
from typing import List, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# --- Define the state of our agent
class AgentState(TypedDict):
    topic: str
    plan: str
    summary: str

# --- Initialize the LLM
llm = ChatOpenAI(model="gpt-4-turbo")

# --- Define the Planning Node
def planner_node(state: AgentState):
    prompt = f"Create a bullet-point plan for a summary on: '{state['topic']}'."
    response = llm.invoke([SystemMessage(content="You are a technical planner."), HumanMessage(content=prompt)])
    return {"plan": response.content}

# 3. Define the Writing Node
def writer_node(state: AgentState):
    prompt = (
        f"Based on this plan: {state['plan']}\n\n"
        f"Write a concise summary on '{state['topic']}' (approx 200 words)."
    )
    response = llm.invoke([SystemMessage(content="You are a technical writer."), HumanMessage(content=prompt)])
    return {"summary": response.content}

# 4. Build the Graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("planner", planner_node)
workflow.add_node("writer", writer_node)

# Set the flow
workflow.set_entry_point("planner")
workflow.add_edge("planner", "writer")
workflow.add_edge("writer", END)

# Compile
app = workflow.compile()

# 5. Execute
print("## Running the planning and writing task ##")
initial_state = {"topic": "The importance of Reinforcement Learning in AI"}
result = app.invoke(initial_state)

print("\n### Plan\n" + result['plan'])
print("\n### Summary\n" + result['summary'])
```

```
## Running the planning and writing task ##

### Plan
- **Introduction to AI and Reinforcement Learning (RL)**
  - Define Artificial Intelligence (AI) and its goal of simulating human intelligence.
  - Introduce Reinforcement Learning as a critical subset of machine learning.
  - Explain the basic concept of RL: learning to make decisions by receiving rewards or penalties.

- **Core Concepts of Reinforcement Learning**
  - Outline key components: agent, environment, action, rewards, state, and policy.
  - Describe how RL differs from other machine learning techniques (supervised and unsupervised learning).

- **Applications of Reinforcement Learning**
  - Highlight diverse applications:
    - Video games and robotic control for adaptive and real-time decision making.
    - Business strategy development such as dynamic pricing and inventory management.
    - Autonomous vehicles in navigating and adjusting to new environments.
    - Healthcare for personalized treatment recommendations.
  
- **Advantages of Reinforcement Learning**
  - Adaptability: Ability to learn optimal actions through trial and error.
  - Decision Making: Helps in sequential decision-making problems where information unfolds over time.
  - Scalability: Can be scaled with the advancement of computational resources and complex problems.

- **Challenges and Limitations**
  - Data efficiency: High demand for data and interaction with the environment.
  - Stability and convergence: Difficulty in achieving stable and reliable learning results.
  - Exploration vs. exploitation: Balancing between exploring new strategies and exploiting known strategies.

- **Recent Advances in RL**
  - Discuss breakthroughs like AlphaGo and OpenAI's agents.
  - Mention improvements in algorithms and computational power, enhancing learning efficiency and performance.
  - Highlight integration with other forms of AI (e.g., deep learning) to enhance capabilities (Deep Reinforcement Learning).

- **Future Outlook**
  - Predict increasing adoption in various sectors as technology and understanding of RL improves.
  - Address potential ethical concerns and need for regulatory frameworks as RL systems become more autonomous.
  - Emphasize continuous research and development needed to overcome existing challenges.

- **Conclusion**
  - Summarize the pivotal role of RL in pushing the boundaries of AI.
  - Reinforce the notion that despite challenges, RL's potential across various industries is immense.
  - Call for more focus on interdisciplinary approaches to enhance RL application and efficiency.

### Summary
Reinforcement Learning (RL) is an essential branch of Artificial Intelligence (AI) that focuses on enabling machines to learn from their interactions with the environment by using a system of rewards and penalties. Unlike other machine learning methods, RL is distinct because it does not require labeled input/output pairs and instead learns to make a sequence of decisions by experiencing the consequences of its actions.

RL is highly adaptable and scalable, making it suitable for a variety of applications including video games, robotics, business strategies like dynamic pricing, autonomous vehicles, and personalized healthcare treatments. This adaptability is rooted in RL's ability to continuously learn optimal actions through trial and error, facilitating complex decision-making processes that unfold over time.

However, RL faces challenges such as demands for extensive data, balancing exploration of new strategies with exploitation of known ones, and achieving stability in learning outcomes. Despite these hurdles, advances in computational power and algorithms have led to significant breakthroughs like AlphaGo and various OpenAI projects, integrating RL with other AI techniques such as deep learning to enhance learning efficiency and performance.

As the technology evolves, RL’s potential applications across various sectors are expected to expand significantly, necessitating ongoing research, ethical considerations, and regulatory measures. Despite its challenges, the role of RL in advancing AI capabilities remains pivotal, promising immense benefits across diverse industries.
```

