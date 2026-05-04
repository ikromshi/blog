---
layout: notebook
title: "Exception Handling and Recovery"
project: agentic-patterns
order: 12
---

## Exception Handling and Recovery

### Overview
For AI anents to operate reliably, they must be able to manage unforseen situations, errors, and malfunctions. Intelligent agents need robust systems to detect problems, initiate recovery procesdures, or at least ensure controlled failure.

This pattern may sometimes be used with reflection. For example, if an initial attempt fails and raises an exception, a reflective process can analyze the failure and re-attempt the task with a refined approach, such as an improved prompt, to resolve the error.

This pattern involves anticipating potential issues, such as tool errors or service unavailability, and developing strategies to mitigate them. These strategies may include error handling, retries, fallbacks, graceful degredation, and notifications. Additionally, the pattern emphasizes recovery mechanisms like state rollback, diagnosis, self-correction, adn escalation, to restore agents to stable operation. 

Key components of exceoption handling and recovery for AI agents are:
- **Error Detection**: This involves meticulously identifying operational issues as they arise. This could manifest as invalid or malformed tool outputs, specific API errors such as 404 or 500 codes, unusually long response times from services or APIs, or incoherent and nonsensical responses that deviate from expected formats. 

- **Error Handling**: Once an error is detected, a carefully thought-out response plan is essential. This includes recording error details meticulously in logs for later debugging and analysis (logging). Retrying the action or request, sometimes with slightly adjusted parameters, may be a viable strategy, especially for transient errors (retries). Utilizing alternative strategies or methods (fallbacks) can ensure that some functionality is maintained. Where complete recovery is not immediately possible, the agent can maintain partial functionality to provide at least some value (graceful degredetaion). Finally, alerting human operators or other agents might be crucial for situations that require human intervention or collaboration (notification).

- **Recovery**: This stage is about restoring agent or system to a stable and operational state after an error. It could involve reversing recent changes or transactions to undo the effects of the error (state rollback). A thorough investigation into the cause of the error is vital for preventing recurrence. Adjusting the agent's logic, or parameters through a self-correction mechanism or replanning process may be needed to avoid the same error in the future. In complex or severe cases, deleting the issue to a human operator or a high-level system (escalation) might be the best course of action.

### Practical Applications & Use Cases
Exception Handling and Recovery is critical for any agent deployed in a real-world scenario where perfect conditions cannot be guaranteed.

- **Customer Service Chatbots**: If a chatbot tries to access a customer database and the database is temporarily down, it shouldn't crash. Instead, it should detect the API error, inform the user about the temporary issue, perhaps suggest trying again later, or escalate the query to a human agent.

- **Automated Financial Trading**: A trading bot attempting to execute a trade might encounter an "insufficient funds" error or a "market closed" error. It needs to handle these exceptions by logging the error, not repeatedly trying the same invalid trade, and potentially notifying the user or adjusting its strategy.

- **Smart Home Automation**: An agent controlling smart lights might fail to turn on a light due to a network issue or a device malfunction. It should detect this failure, perhaps retry, and if still unsuccessful, notify the user that the light could not be turned on and suggest manual intervention.

- **Data Processing Agents**: An agent tasked with prpocessing a batch of documents might encounter a corrupted file. It should skip the corrupted file, log the error, continue processing other files, and report the skipped files at the end rather than halting the entire process.

- **Web Scraping Agents**: When a web-scraping agent encounters a CAPTCHA, a changed website strucutre, or a server error, it needs to handle these gracefully. This could involve pausing, using a proxy, or reporting the specific URL that failed.

### Hands-On Code Example:
This example demonstrates Exception Handling when a tool-use fails.

```python
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import get_key

API_KEY = get_key("../.env", "GOOGLE_API_KEY")

if not API_KEY:
    raise Exception("API KEY absent.")


try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=API_KEY)
    print(f"Language model initialized: {llm.model}")
except Exception as e:
    print(f"Error initializing language model: {e}")
    llm = None



@tool
def get_precise_location_info(address: str) -> str:
    """Given a person's address, computes and returns the person's precise location in coordinates."""
    print("--- get_precise_location_info() invoked")
    # return "23.1222793, -82.3873173"
    return None

@tool
def get_general_area_info(address: str) -> str:
    """Given a person's address, computes and returns the peron's general location (city/country)."""
    print("--- get_general_area_info() invoked")
    return "Tallinn"


system_prompt = """
You are a location analyst. Your job is to get precise location information given a person's general address.
If the precise location doens't exist or isn't available, you must get the general location associated with the person.
"""

location_handler = create_agent(model=llm, system_prompt=system_prompt, tools=[get_precise_location_info, get_general_area_info])


messages_out = location_handler.invoke({"messages": [{"role": "user", "content": "Hello, my name is Ikrom, and I live in 953 Danby Rd."}]})
response = messages_out["messages"][-1]
print(response.content)
```

```
Language model initialized: gemini-2.5-flash
--- get_precise_location_info() invoked
--- get_general_area_info() invoked
[{'type': 'text', 'text': 'I cannot find the precise location of 953 Danby Rd. However, I can tell you that the general area is Tallinn.', 'extras': {'signature': 'Co4EAQw51sfF91mYS/+Kk4bYmg3d/Wf/6KGBxQQSKYglxDvTX2SqjCTMF/cx+Jc+TCWqUo5vk4t3PPC5oEdV6yvE2odQ4YVZKqq6qrK+aeWe00mmc9oySGngij0Ap5JEOPuRILbvJDdYF9QVvgmf7Ihwg8bVT55f3g2p6mPGztNAln44A9ZQXJIeq2a5nQbbhPErQOy0Nxgqz5Tb8rieIw5+ziv3WbT/tSwW3632+CNNnKh5NKKBcsfnRnEagGur/eaken6VoGrr+6ytsdUKocUgXG3hM9iG9BJWxpRtgasy6FddPMrVyc7FHNyGgl2w8BeWUIyjNqBb0K8BZR4hOhULYgXbzhhadKbcBCbKdCmun7UIbNEFFlb/JvX6xQWrdkYOrFDLIAEAh0b9h2JOngFKCrUXUv5SOvqyE5DuTCkQ11KA2+vR+yzqg6DnXz9cxolCTRvu2JJOWUbWd32pkouMRW6G6OAGtIRDVNUszdM2ZztXT53OvUpzUG9BAa5zNOwtiegmbLqDBBS2Tx1L1xXuNkOr32HjbzvQ5bnfQDFDj4s2X3WD4DewCpmktmrHl/w2sgtozH5t/82YAFFuVPajESR0hg/LZSKk6ZDAIGflTWsBpIhko/Ls9M83FYT2fuDuV6Yrkn/UmDHQkFLZ9aca84CWKVxI0zyTxeIzPIXGD2t78nlTs8ilHlLAuIex/g=='}}]
```

```python
import json

pretty_json = json.dumps(messages_out, indent=4, default=str)
print(pretty_json)
print(messages_out["messages"][-1].content[0]["text"])
```

```
{
    "messages": [
        "content='Hello, my name is Ikrom, and I live in 953 Danby Rd.' additional_kwargs={} response_metadata={} id='54521895-270c-4f0a-867c-7f33eceef488'",
        "content='' additional_kwargs={'function_call': {'name': 'get_precise_location_info', 'arguments': '{\"address\": \"953 Danby Rd.\"}'}, '__gemini_function_call_thought_signatures__': {'e4165932-b248-4fab-b75d-059a4b398e03': 'CpICAQw51sdtwmix/755sOm3sn5sw+qpOzMgstx3k1PaHKE+INYOWxO4xpicvlZ85C6lmsomVqHB5GErCGQ8QNyni/VOeprfm9qwTluKMlQAqzVVGtQmcLbDsHzbAzSHGRyVSofUvuxrP/AWe74Tdq500Bw1ZWq22OGqGTrBs/3yM2jBygs1Lowlz8fzMvHQSzR78uvGiSlAo+Kj+jLgodxGYSZB2r8Kr8AxEK0zleB6eemoOUUp4mosE1aFUltt2FfWJPBVMDFcxyPjWLM3emQZnRVqQ6IK2jX4vTVYxTIaeqKhza6Tja5aOk/G+2XcQQnlqA1sqm1Y4le/2GZQwpSXTprVCPjbOs3aLtJxStxHTYtTgA=='}} response_metadata={'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 'safety_ratings': [], 'model_provider': 'google_genai'} id='lc_run--019df1ae-a257-7cf0-8779-4fd9fca90d0e-0' tool_calls=[{'name': 'get_precise_location_info', 'args': {'address': '953 Danby Rd.'}, 'id': 'e4165932-b248-4fab-b75d-059a4b398e03', 'type': 'tool_call'}] invalid_tool_calls=[] usage_metadata={'input_tokens': 185, 'output_tokens': 78, 'total_tokens': 263, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 54}}",
        "content='null' name='get_precise_location_info' id='5e7a6379-c12b-40f1-875a-267528cf65ac' tool_call_id='e4165932-b248-4fab-b75d-059a4b398e03'",
        "content='' additional_kwargs={'function_call': {'name': 'get_general_area_info', 'arguments': '{\"address\": \"953 Danby Rd.\"}'}, '__gemini_function_call_thought_signatures__': {'00b79bed-170d-49e5-b625-1aaf74dcbfca': 'CpoDAQw51scrWw4wLcWBnY1yHAJsPrMGhMpgrDfaYl4lGJPvGQVJNxKTNp403UzaVg4KsRCVl8BabIPsOEGNkMICp9JMa3ddxlL4+w9b3tce42FRRYO/LcW0lwDte0aIYI0joQI7/vgj0GErrIF2sPJlLCS9aL2c5tHEsI1gDgJ3HVnfAiWo7rDsDKYdIYuEENfOutR4tK2C0N/L71gBHA++mRKRSOaituppBKYwVPIsjEWPZsM8usOtpgB2CGjWtbHIpBmprxi9Jfp68Z/mUKOF0Ar3Vi9n8PYdQv2GRptcCiC0B9E5rbN7Y9UrY53GY9Lus3PynF78rp39OCerOxnYwQVFLfCdofdgrM5ba55mEHqcW9+ck2o/haqZ0U5a9sjcTgS7iR6NZOQgs5Wisn8GO1ODRTk3HbvI7kDPP/7v38pDObtZiMtB1uEmuPNEI8a1OXKuOq4wt9PUxzKVhPoEhN5nXFQChgNMjVKQBr85uYq8aSJIOY6eurgaQduxIJBb/bSPzE1VHDlOZDcYU5cSWeZLLh9WOmrodt0='}} response_metadata={'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 'safety_ratings': [], 'model_provider': 'google_genai'} id='lc_run--019df1ae-a8b3-73c2-97ea-80371d5578dd-0' tool_calls=[{'name': 'get_general_area_info', 'args': {'address': '953 Danby Rd.'}, 'id': '00b79bed-170d-49e5-b625-1aaf74dcbfca', 'type': 'tool_call'}] invalid_tool_calls=[] usage_metadata={'input_tokens': 227, 'output_tokens': 113, 'total_tokens': 340, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 89}}",
        "content='Tallinn' name='get_general_area_info' id='28c0a0da-5213-431a-baff-0bfd36328caf' tool_call_id='00b79bed-170d-49e5-b625-1aaf74dcbfca'",
        "content=[{'type': 'text', 'text': 'I cannot find the precise location of 953 Danby Rd. However, I can tell you that the general area is Tallinn.', 'extras': {'signature': 'Co4EAQw51sfF91mYS/+Kk4bYmg3d/Wf/6KGBxQQSKYglxDvTX2SqjCTMF/cx+Jc+TCWqUo5vk4t3PPC5oEdV6yvE2odQ4YVZKqq6qrK+aeWe00mmc9oySGngij0Ap5JEOPuRILbvJDdYF9QVvgmf7Ihwg8bVT55f3g2p6mPGztNAln44A9ZQXJIeq2a5nQbbhPErQOy0Nxgqz5Tb8rieIw5+ziv3WbT/tSwW3632+CNNnKh5NKKBcsfnRnEagGur/eaken6VoGrr+6ytsdUKocUgXG3hM9iG9BJWxpRtgasy6FddPMrVyc7FHNyGgl2w8BeWUIyjNqBb0K8BZR4hOhULYgXbzhhadKbcBCbKdCmun7UIbNEFFlb/JvX6xQWrdkYOrFDLIAEAh0b9h2JOngFKCrUXUv5SOvqyE5DuTCkQ11KA2+vR+yzqg6DnXz9cxolCTRvu2JJOWUbWd32pkouMRW6G6OAGtIRDVNUszdM2ZztXT53OvUpzUG9BAa5zNOwtiegmbLqDBBS2Tx1L1xXuNkOr32HjbzvQ5bnfQDFDj4s2X3WD4DewCpmktmrHl/w2sgtozH5t/82YAFFuVPajESR0hg/LZSKk6ZDAIGflTWsBpIhko/Ls9M83FYT2fuDuV6Yrkn/UmDHQkFLZ9aca84CWKVxI0zyTxeIzPIXGD2t78nlTs8ilHlLAuIex/g=='}}] additional_kwargs={} response_metadata={'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 'safety_ratings': [], 'model_provider': 'google_genai'} id='lc_run--019df1ae-acc1-72b3-a61d-3021fa99170e-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 271, 'output_tokens': 135, 'total_tokens': 406, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 107}}"
    ]
}
I cannot find the precise location of 953 Danby Rd. However, I can tell you that the general area is Tallinn.
```

