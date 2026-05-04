---
layout: notebook
title: "Prompt Chaining"
project: agentic-patterns
order: 1
---

## Prompt Chaining
### Overview
Prompt Chaining allows us to break down monolithic tasks into sub-tasks and use one's output as another's input. This is crucial to solve complex problems in small, manageable steps.

Furthermore, Prompt Chaining enables the integration of external knowledge tools at each point of the chain, making them specialized in their sub-tasks and therefore making it easier to accomplish them. This also reduces _Contextual Drift_.

### Practical Applications and Use Cases
The core utility of Prompt Chaining lies in breaking down complex problems into sequential, manageable steps. Here are several practical applications and use cases:

1. Information Processing Workflows: summarizing a document, extracting key entities, and then using those entities to query a database or generate a report.

2. Complex Query Answering: answering complex questions like "What were the main causes of the stock market crash in 1929, and how did th egovernment policy respond?"

3. Data Extraction and Transformation: the conversion of unstructured text into a structured format is typically ahieved through an iterative process, requiring sequential modifications to improve the accuracy and completeness of the output.

4. Content Generation Workflows: composition of complex content consisting of distinct phases, including initial ideation, structural outlining, drafting, and subsequent revision.

5. Conversational Agents with State: Prompt Chaining provides a foundational mechanism for preserving converstaional continuity.

6. Code Generation and Refinement: same as above.

### Hands-On Code Example
This is an example of a two-step prompt chain that functions as a data processing pipeline. The chain extracts and transforms an input string into a structured data format.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

# Initialize the language model
llm = ChatOpenAI(temperature=0)

# ---- Prompt 1: Extract Information
prompt_extract = ChatPromptTemplate.from_template(
    "Extract the technical specifications from the following text:\n\n{text_input}"
)

# ---- Prompt 2: Transform to JSON
prompt_transform = ChatPromptTemplate.from_template(
    "Transform the following specifications into a JSON object with 'cpu', 'memory', and 'storage' as keys:\n\n{specifications}"
)

# ---- Build the Chain using LCEL
# StrOutputParser() converts the LLM's message output to a simple string.
extraction_chain = prompt_extract | llm | StrOutputParser()


# The full chain passes the output of the extraction chain into 'specifications' variable for the transformation prompt.
full_chain = (
    {"specifications": extraction_chain}
    |
    prompt_transform
    |
    llm
    |
    StrOutputParser()
)


# ---- Run the chain
input_text = "The new laptop model features a 3.5GHz octa-core processor, 16GB of RAM, and a 1TB NVMe SSD."

# Execute the chain with the input text dictionary.
final_result = full_chain.invoke({"text_input": input_text})
print("\n--- Final JSON Output ---")
print(final_result)
```

```

--- Final JSON Output ---
{
    "cpu": "3.5GHz octa-core",
    "memory": "16GB",
    "storage": "1TB NVMe SSD"
}
```

### Context Engineering vs Prompt Engineering
Context Engineering is the systematic discipline of designing, constructing, and delivering a complete informational environment to an AI model prior to token generation. This methodoly asserts that the quality of a model's output is less dependent on the model's architecture itself and more on the richness of the context provided.

Context Engineering represents a significant evolution from traditional prompt engineering, which focuses primarily on optimizing the phrasing of a user's immediate query.

