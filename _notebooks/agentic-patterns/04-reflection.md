---
layout: notebook
title: "Reflection"
project: agentic-patterns
order: 4
---

## Reflection
### Overview
Previous patterns (Chaining, Routing, Parallelization) enable agents to perform complex tasks more efficiently and flexibly. However, even with sophisticated workflows, an agent's initial output or plan might not be optimal, accurate, or complete. This is where the Reflection pattern comes into play.

The Relflection pattern involves an agent evaluating its own work, output, or internal state and using that evaluation to improve its performance or refine its response. It's a form of self-improvement, allowing the agent to iteratively refine its output or adjust its approach based on feedback, internal critique, or comparison against desired criteria. Reflection can occasionally be facilitated by a separate agent whose specific role is to analyze the output of an initial agent.

The process typically involves:
1. **Execution**: The agent performs a task or generates an initial output.

2. **Evaluation/Critique**: The agent (often using anotehr LLM call or a set of rules) analyzes the result from the previous step. The evaluation might check for factual accuracy, coherence, stule, completeness, adherence to instructions, or other relevant criteria.

3. **Reflection/Refinement**: Based on the critique, the agent determines how to improve. This might involve generating a refined output, adjusting parameters for a subsequent step, or even modifying teh overall plan.

4. **Iteration (optional but common)**: The refined output or adjusted approach can then be executed, and the reflection process can repeat until a satisfactory result is achieved or a sotpping condition is met.

---

### Generator-Critic
An effective implementation of this pattern is to separate the process into two distinct logical roles: a Producer and a Critic. This is often the "Generator-Critic" or "Producer-Reviewer" model. While a single agent can perform self-reflection, using two specialized agents (or two separate LLM calls with distinct system prompts) often yields more robust an unbiased results.

This separation of concerns is powerful because it prevents the "cognitive bias" of an agent reviewing its own work. The Critic agent approaches the output with a fresh perspective, dedicated entirely to finding errrors and areas for improvement.

Furthermore, the effectiveness of the Reflection pattern is significantly enhanced weh the LLM keeps a memory of the conversation. This conversation history provides crucial context for the evaluation phase, which allows the agent to assess its output in isolation, also against previous interactions, use feedback, and evolving goals.

---

### Practical Applications and Use Cases
The Reflection pattern is valuable in scenarios where output quality, accuracy, or adherence to complex constraints is critical.

#### 1. Creative Writing and Content Generation
Use Case: An agent writing a blog post.
* Reflection: Generate a draft, critique it for flow, tone, and clarity, then rewrite it based on the critique. Repeat until the post meets quality standards.
* Benefit: Procides more polished and effective content.

#### 2. Code Generation and Debugging
Use Case: An agent writing a Python function.
* Reflection: Write initial code, run tests or static analysis, identify errors or inefficiencies, then modify the code based on the findings.
* Benefit: Generates more robust and functional code.

#### 3. Complex Problem Solving
Use Case: An agent solving a logic puzzle.
* Reflection: Propose a step, evaluate if it leads closer to the solution or introduces contradictions, backtrack or choose a different step if needed.
* Benefit: Improves the agent's ability to navigate complex spaces.

#### 4. Summarization and Information Synthesis
Use Case: An agent summarizing a long document.
* Reflection: Generate an initial summary, compare it against key points in the original document, refine the summary to include missing information or improve accuracy.
* Benefit: Creates more accurate and comprehensive summaries.

#### 5. Planning and Strategy
Use Case: An agent planning a series of actions to achieve a goal.
* Reflection: Generate a plan, simulate its execution or evaluate its feasibility against constraints, revise the plan based on the evaluation.
* Benefit: Develops more effective and realistic plans.

#### 6. Conversational Agents
Use Case: A customer support agent.
* Reflection: After a user response, review the conversation history and th elast generated message to ensure coherence and address the user's latest input accurately.
* Benefit: Leads to more natual and effective conversations.

### Hands-On Code Example
This example shows a reflection loop to iteratively generate and refine a Python function that calculates the factorial of a number. The process starts with a task prompt, generates initial code, and then repeatedly reflects on the code based on critiques from a simulated senior software engineer role, refining the code in each iteration until the critique stage determines the code is perfect or a maximum number of iterations is reached.

```python
from dotenv import get_key
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage


# ---- Configuration
OAI_KEY = get_key("../.env", "OPENAI_API_KEY")

if not OAI_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file. Please add it.")

# Initialize the LLM
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0.1)


def run_reflection_loop():
    """
    Demonstrates a multi-step AI reflection loop to progressively improve a Python function.
    """
    # --- The Core Task
    task_prompt = """
    Your task is to create a Python function named `calculate_factorial`.
    This function should do the following:
    1. Accept a single integer `n` as input.
    2. Calculate its factorial (n!).
    3. Include a clear docstring explaining what the function does.
    4. Handle edge cases: The factorial of 0 is 1.
    5. Handle invalid input: Raise a ValueError if the input is a negative number.
    """

    # --- The Reflection Loop
    max_iterations = 3
    current_code = ""

    # Conversation history to provide context in each step.
    message_history = [HumanMessage(content=task_prompt)]

    for i in range(max_iterations):
        print("\n" + "=" * 25 + f" REFLECTION LOOP: ITERATION {i + 1} " + "=" * 25)

        # --- 1. Generate / Refine Stage
        # In the first iteration, it generates. In subsequent iterations, it refines.
        if i == 0:
            print("\n>>> STAGE 1: GENERATING initial code...")
            # The first message is the task prompt
            response = llm.invoke(message_history)
            current_code = response.content
        else:
            print("\n>>> STAGE 1: REFINING code based on previous critique...")
            # The message history now contains the task,
            # the last code, and the last critique.
            # We instruct the model to apply the critiques.
            message_history.append(HumanMessage(content="Please refine the code using the critiques provided."))
            response = llm.invoke(message_history)
            current_code = response.content

        print("\n--- Generated code (v" + str(i+1) + ") ---\n" + current_code)
        message_history.append(response) # add the generated code to history


        # --- 2. Reflect Stage
        print("\n>>> STAGE 2: REFLECTING on the generated code...")
        # Create a specific prompt for the reflector agent.
        # This asks the model to act as a senior code reviewer.
        reflector_prompt = [
            SystemMessage(content="""
            You are a senior software engineer and an expert in Python.
            Your role is to perform a meticulous code review.
            Critically evaluate the provided Python code based on the original task requirements.
            Look for bugs, style issues, missing edge cases, and areas for improvement.
            If the code is perfect and meets all requirements,
            respond with the single phrase 'CODE_IS_PERFECT'.
            Otherwise, provide a bulleted list of your critiques.
            """
            ),
            HumanMessage(content=f"Original Task:\n{task_prompt}\n\nCode to Review:\n{current_code}")
        ]

        critique_response = llm.invoke(reflector_prompt)
        critique = critique_response.content

        # --- 3. STOPPING CONDITION
        if "CODE_IS_PERFECT" in critique:
            print("\n--- Critique ---\nNo further critiques found. The code is satisfactory.")
            break

        print("\n--- Critique ---\n" + critique)
        # Add the critique to the history for the next refinement loop.
        message_history.append(HumanMessage(content=f"Critique of the previous code:\n{critique}"))

    print("\n" + "=" * 30 + " FINAL RESULT " + "=" * 30)
    print("\nFinal refined code after the reflection process:\n")
    print(current_code)

if __name__ == "__main__":
    run_reflection_loop()
```

```

========================= REFLECTION LOOP: ITERATION 1 =========================

>>> STAGE 1: GENERATING initial code...

--- Generated code (v1) ---
```python
def calculate_factorial(n: int) -> int:
    """
    Calculate the factorial of a non-negative integer.

    The factorial of n (written as n!) is the product of all positive integers
    from 1 to n. By definition, 0! is 1.

    Args:
        n (int): A non-negative integer.

    Returns:
        int: The factorial of n.

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    if n == 0:
        return 1

    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
```

>>> STAGE 2: REFLECTING on the generated code...

--- Critique ---
No further critiques found. The code is satisfactory.

============================== FINAL RESULT ==============================

Final refined code after the reflection process:

```python
def calculate_factorial(n: int) -> int:
    """
    Calculate the factorial of a non-negative integer.

    The factorial of n (written as n!) is the product of all positive integers
    from 1 to n. By definition, 0! is 1.

    Args:
        n (int): A non-negative integer.

    Returns:
        int: The factorial of n.

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    if n == 0:
        return 1

    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
```
```

