---
layout: notebook
title: "Parallelization"
project: agentic-patterns
order: 3
---

## Parallelization
### Overview
In addition to Prompt Chaining and Routing, many complex agentic tasks involve multiple sub-tasks that can be executed _simultaneously_ rather than one after another. This is where Parallelization comes in.

Parallelization involves executing multiple compoennts, such as LLM calls, tool usages, or even entire sub-agents concurrently. The core idea is to identify parts of the workflow that don't depend on the output of the other parts and execute them in parallel. This is particularly effective when dealing with external services (like APIs or DBs) that have latency, as you can issue multiple requests concurrently.

### Practical Applications and Use Cases
#### 1. Information Gathering and Research
Use Case: An agent researching a company.
- Prallel Tasks: Run sentiment analysis, extract keywords, categorize feedback, and identify urgent issues simultaneously across a batch of feedback entries.
- Benefit: Provides a multi-faceted analysis quickly.

#### 2. Data Processing and Analytics
Use Case: An agent analyzing customer feedback.
- Parallel Tasks: Run sentiment analysis, extract keywords, categorize feedback, and identify urgent issues simultenously across a batch of feedback entries.
- Benefit: Provides a multi-faceted analysis quickly.

#### 3. Multi-API or Tool Interaction
Use Case: A travel planning agent.
- Parallel Tasks: Check flight prices, search for hotel availability, look up local events, and find restaurant recommendations concurrently.
- Benefit: Presents a complete travel plan faster.

#### 4. Content Generation with Mulitple Components
Use Case: An agent creating a marketing email.
- Parallel Tasks: Generate a subject line, draft the email body, find a relevant image, and create a call-to-action button text simultaneously.
- Benefit: Assembles the final email more efficiently.

#### 5. Validation and Verification
Use Case: An agent verifuing user input.
- Parallel Tasks: Check email format, validate phone number, verify address against a database, and check for profanity simultaneously.
- Benefit: Provides faster feedback on input validity.

#### 6. Multi-Modal Processing
Processing different modalities (text, image, audio) of the same input concurrently.
Use Case: An agent analyzing a social media post with text and an image.
- Parallel Tasks: Analyze the text for sentiment and keywords and analyze the image for objects and scene description simultaneously.
- Benefit: Integrates insights from different modalities more quickly.

#### 7. A/B Testing or Multiple Options Generation
Use Case: An agent generating different creative text options.
- Parallel Tasks: Generate three different headlines for an article simultaneously using slightly different prompts or models.
- Benefit: Allows for quick comparison and selection of the best option.

### Hands-On Code Example
Parallel execution within LangChain is fascilitated by LCEL (LangChain Expression Language). The primary method involves structuring multiple runnable components within a dictionary or list contruct. When this collection is passed as input to a subsequent component in the chain, the LCEL runtime executes the contained runnables concurrently.

The following example workflow is designed to execute two independent operations concurrently in response to a single user query. These parallel processes are instantiated as distinct chains or functions, annd their respective outputs are subsequently aggregated into a unified result.

```python
import asyncio
from typing import Optional
from dotenv import get_key

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough


# ---- Configuration
OAI_KEY = get_key("../.env", "OPENAI_API_KEY")
try:
    llm: Optional[ChatOpenAI] = ChatOpenAI(model="gpt-5.4-nano", temperature=0.7)
except Exception as e:
    print(f"Error initializing language model: {e}")
    llm = None


# ---- Define Independent Chains
# These three chains represent distinct tasks that can be executed in parallel.
summarize_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "Summarize the following topic concisely:"),
        ("user", "{topic}")
    ])
    | llm
    | StrOutputParser()
)

questions_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "Generate three interesting questions about the following topic:"),
        ("user", "{topic}")
    ])
    | llm
    | StrOutputParser()
)

terms_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "Identify 5-10 key terms from the following topic, separated by commas:"),
        ("user", "{topic}")
    ])
    | llm
    | StrOutputParser()
)


# ---- Parallel + Synthesis Chain
# 1. Define the block of tasks to run in paralle. The results of these, 
#    along with the original topic will be fed into the next step.
map_chain = RunnableParallel(
    {
        "summary": summarize_chain,
        "questions": questions_chain,
        "key_terms": terms_chain,
        "topic": RunnablePassthrough(), # pass the original topic through
    }
)

# 2. Define the final synthesis prompt which will combine the parallel results.
synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", 
        """Based on the following information:
        Summary: {summary}
        Related Questions: {questions}
        Key Terms: {key_terms}
        Synthesize a comprehensive answer."""
    ),
    ("user", "Original topic: {topic}")
])

# 3. Construct the full chain by piping the parallel results directly
#    into the synthesis prompt, followed by the LLM and output parser.
full_parallel_chain = map_chain | synthesis_prompt | llm | StrOutputParser()



# --- Run the Chain
async def run_parallel_example(topic: str) -> None:
    """
    Asynchronously invokes the parallel processing chain with a specific topic
    and prints the synthesized result.

    Args:
        topic: The input topic to be processed by the LangChain chains.
    """
    if not llm:
        print("LLM is not instantiated. Cannot run example.")
        return

    print(f"\n--- Running Parallel Langchain Example for Topic: '{topic}' ---")
    try:
        # The input to `ainvoke` is the single 'topic' string,
        # then passed to each runnable in the `map_chain`.
        response = await full_parallel_chain.ainvoke(topic)
        print(response)
    except Exception as e:
        print(f"\nAn error occurred during chain execution: {e}")

if __name__ == "__main__":
    test_topic = "The history of space exploration"
    # asyncio.run(run_parallel_example(test_topic))
    await run_parallel_example(test_topic)

```

```

--- Running Parallel Langchain Example for Topic: 'The history of space exploration' ---
The history of space exploration is a story of expanding ambition—made possible by breakthroughs in technology, shaped by global politics, and continually redirected by scientific discoveries.

## 1) From skywatching to rockets: the foundation of the space age
Humanity’s earliest “space exploration” was observation. Over centuries, improved astronomy made it possible to understand orbits, predict celestial events, and eventually imagine reaching space deliberately. The key turning point came with advances in rocketry and propulsion. When rockets became capable of carrying payloads to high altitudes and beyond, exploration shifted from watching the sky to traveling through it.

## 2) The first satellites and the start of the Space Race
A major milestone was **Sputnik 1** in 1957, the first artificial satellite. It proved satellites could be placed into orbit, changing the goal of space activity from theoretical possibility to routine capability. Soon after, **Yuri Gagarin** in 1961 demonstrated that humans could survive and operate in orbit—turning “space” into a realm both for science and for national prestige.

These achievements accelerated into the **Space Race**, a competition strongly influenced by Cold War politics. The race drove rapid investment, rapid engineering iteration, and dramatic public commitment to high-visibility missions.

## 3) Crewed missions and the Moon: Apollo and the peak of early ambition
The **Apollo program** built on earlier rockets, spacecraft systems, and tracking networks to attempt something unprecedented: crewed **Moon landings**. Reaching the Moon reshaped what was considered achievable—not just technologically, but conceptually. It demonstrated that long-duration space travel, precision navigation, lunar descent, and surface operations could all be executed by engineered systems and trained crews.

## 4) Space stations and reusable systems: staying in space longer
After the Moon landings, exploration goals expanded toward endurance and research in orbit. This era saw major development of **space stations**, culminating in the **International Space Station (ISS)**—a symbol of how cooperation increasingly complemented competition.

Meanwhile, the development of systems like the **Space Shuttle** and other advances in reusable technology began to change the economics and logistics of reaching space. Instead of space travel being rare and extremely costly, the long-term vision shifted toward more frequent missions and a broader range of scientific and industrial activities.

## 5) Today’s focus: interplanetary travel, commercial space, and renewed Moon/Mars work
In more recent decades, exploration goals have broadened from Earth orbit to the solar system. **Mars rovers** and planetary probes influenced priorities by revealing how challenging, dynamic, and scientifically valuable other worlds are. These results help shape the next steps—such as refining landing techniques, studying past habitability, and preparing for human exploration.

At the same time, **commercialization of space** has become a major theme. Private companies increasingly contribute launch capabilities, satellites, and mission services, which lowers barriers and enables more rapid mission planning. The renewed focus on the **Moon** (as a stepping stone) and **Mars** (as a primary long-term target) reflects both scientific interest and strategic planning for future deep-space missions.

---

### Big-picture takeaway
Over time, space exploration evolved through a chain of enabling advances:
- **Rockets and propulsion** made travel possible and expanded mission distance.
- **Satellites and orbital spacecraft** transformed exploration into ongoing, data-rich observation.
- **The Space Race and Cold War urgency** accelerated development and public momentum.
- **Apollo and Moon landing capability** proved crewed missions could reach another world.
- **Space stations and reusable systems** supported longer-term presence and research.
- **Modern discoveries (Moon/planet/Mars missions)** and **commercial innovation** increasingly shape what comes next: interplanetary exploration and sustainable activity beyond Earth.

If you want, I can also organize this history into a timeline with the specific missions/years for each key term (Sputnik 1, Gagarin, Apollo, Shuttle, ISS, Mars rovers).
```

---
As a rule of thumb, use this pattern when a workflow contains multiple independent operations that can run simultaneously, such as fetchin from several APIs, processing different chunks of data, or generating multiple pieces of content for later synthesis

