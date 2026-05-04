---
layout: notebook
title: "Multi-Agent Collaboration"
project: agentic-patterns
order: 7
---

## Multi-Agent Collaboration
### Overview
The Multi-Agent Collaboration pattern involves designing systems where multiple independent or semi-independent agents work together to achieve a common goal. Each agent typically has a defined role, specific goals aligned with the overall objective, and potentially access to different tools or knowledge bases. Unlike the Routing/Branching, the power of this pattern lies in the interaction and synergy between these agents.

Collaboration can take various forms:

- **Sequential Handoffs**: One agent completes a task and passes its output to another agent for the next step in a pipeline (similar to the Planning pattern, but explicitly involving different agents).

- **Parallel Processing**: Multiple agents work on different parts of a problem simultaneously, and their results are later combined.

- **Debate and Consensus**: Multi-Agent Collaboration where Agents with varied perspectives and information sources engage in discussions to evaluate options, ultimately reaching a consensus or a more informed decision.

- **Hierarchical Structures**: A manager agent might delegate tasks to worker agents dynamically based on their tool access or plugin capabilities and synthesize their results. Each agent can also handle relevant groups of tools, rather than a single agent handling all the tools.

- **Expert Teams**: Agents with specialized knowledge in different domains (e.g., a researcher, a writer, an editor) collaborate to produce a complex output.

- **Critic-Reviewer**: Agents create initial outputs such as plans, drafts, or answers. A second groups of agents then critically assesses this output for adherence to policies, security, compliance, correctness, quality, and alignment with organizational objectives. The original creator or a final agent revises the output based on this feedback.


### Practical Applications & Use Cases
MAC is applicable across numerous domains:

- **Complex Research and Analysis**: A team of agents could collaborate on a research project. One agent might specialize in searching academic databases, another in summarizing findings, a third in identifying trends, and a fourth in synthesizing the information into a report. This mirrows how a human research team might operate.

- **Software Development**: Agents collborating on building software. Same structure as above.

- **Creative Content Generation**: Creating a marketing campaign could involve a market research agent, a copywriter agent, a graphic design agent (using image gen tools), and a social media scheduling agent, all working together.

- **Financial Analysis**: A multi-agent system could analyze financial markets. Agents might specialize in fetching stock data, analyzing news sentiment, perfroming technical analysis, and generating investment recommendations.

- **Customer Support Escalation**: A front-line support agent could handle initial queries, escalating complex issues to a specialist agent (e.g., technical expert or a billing specialist) when needed, demonstrating a sequential handoff based on problem complexity.

- **Supply Chain Optimization**: Agents could represent different nodes in a supply chain (suppliers, manufacturers, distributors) and collaborate to optimize inventory levels, logistics, and scheduling in response to changing demand or disruptions.

### Hands-On Code
The code below defines AI Agents to generate a blog post about AI trends. The core of the application involves defining two agents: a researcher to find and summarize AI trends, and a writer to create a blog post based on the research.

Two tasks are defined accordingly: one for researching the trends and another for writing the blog post, with the writing task depending on the output of the research task. These agents and tasks are then assembled into a group, specifying a sequential process where tasks are executed in order.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import get_key

# --- Configuration
KEY = get_key("../.env", "OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-5.4-nano", temperature=0.5, api_key=KEY)

if not llm:
    raise Exception("Something went wrong while initializing the LLM model.")


# --- Prompts for agents
researcher_prompt = ChatPromptTemplate.from_messages([
    ("system", 
        """ 
            You are an experienced Senior Research Analyst with a knack for identifying key trends and synthesizing information. 
            Your goal is to find and summarize the latest trends in AI.
        """
    ),
    ("human", "{task}")
])

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", 
        """
            You are a skilled Technical Content Writer who can translate complex techincal topics into accessible content.
            Your goal is to write a clear and engagin blog post based on research findings.
        """
    ),
    ("human", 
        """
            Your task is as follows: {task}.
            
            The research content is: {research}.
        """
    )
])


# --- Agentic Execution Chain
research_chain = researcher_prompt | llm | StrOutputParser()
writer_prompt = writer_prompt | llm | StrOutputParser()


def main():
    task_1 = "Research top 3 emerging trends in AI in 2025-2026; focus on practical applications and potential impact. Output a detail summary of the top 3 trends, including key points and sources."
    task_2 = "Write a 500-word blog post based on the research findings. The post should be engaging and easy for a general audience to understand."

    research_output = research_chain.invoke({"task": task_1})
    blog_post = writer_prompt.invoke({"task": task_2, "research": research_output})

    print(blog_post)


if __name__ == "__main__":
    main()
```

```
## The 3 Biggest AI Trends for 2025–2026 (and How They’ll Actually Change Work)

AI headlines often focus on smarter chatbots. But the real momentum from 2025 to 2026 is about something more practical: systems that can *do work*, understand *real-world inputs* like documents and audio, and run in businesses with *governance and cost control*. Here are the top three emerging AI trends—and what impact you can realistically expect.

---

### 1) Agentic AI in production: from “chat” to “action”
The biggest shift is that AI is moving from answering questions to executing tasks end-to-end. Instead of generating text, “agentic” systems plan steps, call tools (APIs, databases, CRMs, helpdesks), check results, and iterate until a goal is reached—often with guardrails and human approvals for risky moves.

**Where it’s useful now**
- **Customer support & operations:** An agent can triage tickets, pull order/account details, draft a response, and open or close cases.
- **Software engineering:** Agents can generate code, run tests, open pull requests, and respond to CI failures—within sandboxed environments.
- **Enterprise back-office work:** Invoice processing, procurement steps, payroll exception handling, and compliance documentation.
- **Sales and RevOps:** Updating pipelines, generating quotes, cleaning CRM data, and triggering follow-ups based on events.

**Likely impact**
- More productivity through **automation of execution**, not just writing.
- Lower costs by reducing manual triage and repetitive operations.
- New risks—because agents can take actions—so demand for **permissions, audit logs, monitoring, and evaluation** will grow quickly.

**What to watch**
The winning implementations won’t be flashy demos. They’ll be built for reliability: planning + verification loops, grounding in trusted data, and continuous testing.

---

### 2) Multimodal AI goes operational: vision, audio, and documents that produce structured outputs
From 2025–2026, multimodal AI is becoming less of a “cool demo” and more of a core workflow engine. Models increasingly process images, audio, and complex documents (scanned PDFs, forms, diagrams) and return outputs that businesses can use directly—like extracted fields, classifications, and structured decision results.

**Where it shows up**
- **Document intelligence at scale:** Extract invoice/contract/claim fields, detect missing information, and flag anomalies.
- **Manufacturing & field service:** Interpret equipment photos, classify defects, and generate maintenance guidance.
- **Healthcare operations:** Summarize clinician notes, route cases, and assist with coding/billing support (within compliance constraints).
- **Contact centers/media:** Real-time call transcription plus key decision extraction and CRM updates.
- **Retail operations:** Vision-based compliance checks for shelves and planograms.

**Likely impact**
- Faster cycles for knowledge work that currently requires manual reading and typing.
- Better accuracy when systems use **retrieval-augmented approaches** and document-aware extraction (tables/forms, not just text).
- More traceability, because modern systems can provide structured evidence tied to what they extracted.

---

### 3) Governance + efficiency engineering: making AI safe and affordable enough to scale
The third trend is less exciting—but essential: organizations are investing heavily in the engineering needed to deploy AI responsibly. That includes governance (auditing, permissions, data handling), evaluation (offline and live testing), and efficiency improvements (smaller models, caching, quantization, smarter routing).

**Practical applications**
- Policy-controlled copilots and agents (role-based access, tool permissions, secure retrieval).
- Compliance automation: audit trails and evidence logs for regulated workflows.
- Cost optimization: routing simpler requests to cheaper models and batching predictable jobs.
- Privacy protection: PII redaction, retention controls, and secure connectors.

**Likely impact**
- Faster, safer deployment because risks are managed systematically.
- Lower total cost of ownership through efficiency engineering.
- More trustworthy outputs via continuous evaluation and monitoring.

---

### The takeaway
From 2025 to 2026, AI adoption will be driven less by “bigger models” and more by three capabilities: **agents that take action**, **multimodal systems that understand real documents and media**, and **governance + efficiency that make deployment practical**.  

If you want to, tell me your industry (healthcare, banking, retail, manufacturing, etc.) and I’ll suggest high-value pilot projects for each trend with measurable KPIs.
```

