---
layout: notebook
title: "Human-in-the-Loop"
project: agentic-patterns
order: 13
---

## Human-in-the-Loop

### Overview
The HITL pattern integrates AI with the human input to enhance agent capabilities.. This approach acknowledges that optimal AI performance frequently requires a combination of automated processing and human insight, especially in scenarios with high complexity and ehtical considerations. 

HITL encompasses several key aspects: 
- ___Human Oversight___, which involves monitoring AI agent performance and output (e.g., via log reviews or real-time dashboards) to ensure adherence to guideleines and prevent undesirable outcomes. 

- ___Intervention and Correction___ occurs when an AI agent encounters errors or ambiguous scenarios and may request human intervention. 

- ___Human Operators___ rectify errors, supply missing data, or guide the agent, which also informs future agent improvements. 

- ___Human Feedback for Learning___ is collected and used to refine AI models, prominently in methodologies like RL and human feedback.  

- ___Decision Augmentation___ is where an AI agent provides analyses and recommendations to a human, who then makes the final decision, enhancing human decision-making through AI-generated insights rather than full autonomy. 

- ___Human-Agent Collaboration___ is a cooperative interaction where humans anad AI agents contribute their respective strenghts; routine data processing may be handled by the agent, while creative problem-solving or complex negotiations are managed by the human. 

- Finally, ___Escalation Policies___ are established protocols that dictate when and how an agent should escalate tasks to human operators, preventing errors in situations beyong the agent's capability.


### Practical Applications & Use Cases
The HITL pattern is important in industries and applications where accuracy, safety, ethics, or nuanced understanding is are paramount:

- Content Moderation: AI agents can rapidly filter vast amounts of online content for violations, but unclear cases are escalated to human moderators for review and final decision.

- Autonomous Driving: Self-driving cars are designed to hand over control to a human driver in complex, undpredictable, or dangerous situations where AI cannot confidently navigate.

- Financial Fraud Detection: High-risk or ambiguous alerts around suspicious transactions are often sent to human analysts for further investigation.

- Legal Document Review: Once agents quickly scan and categorize throusands of legal documents, human legal professionals review the AI's findings for accuracy and related context.

- Customer Support (Complex Queries): If the user's problem is too complex, emotionally charged, or requires empathy the agen't can't provide, the conversation is handed over to a human.

- Data Labeling and Annotation: Humans are needed in the loop to accurately label images, text, or audio to provide the ground truth in annotation.


__"Human-on-the-loop"__ is a variation of this pattern where human experts define the overarching policy, and the AI then handles immediate actions to ensure compliance:
- Automated financial trading system: In this scenario, a human financial expert sets the overarching investment strategy and rules. For instance, the human might define the policy as: "Maintain a portfolio of 70% tech stocks and 30% bonds, do not invest more than 5% in any single compnay, and automatically sell any stock that falls 10% below its purachase price." The AI then monitors the stock market in real-time, executing trades instantly when these predefined conditions are met. The AI is handling the immediate, high-speed actions based on the slower, more strategic policy set by the human operator.

- Modern call center - In this setup, a human manager establishes high-level policies for customer interactions. For instance, the manager might set rules such as "any call mentioning 'service outage' should be immediately routed to a technical support specialist," or "if a customer's tone of voice indicates high frustration, the system should offer to connect them directly to a human agent.". The AI system then handles the initial customer interactions, listening to and interpreting their needs in real-time. It autonomously executes the manager's policies by instantly routing the calls or offering escalations without needing human intervention for each individual case.

### Hands-On Code Example
LangChain provides the HITL pattern as a middleware on top of agent tool calls. It does this by checking each tool call against a configurable policy. If intervention is needed, the middleware issues an __interrupt__ that halts execution.  A human decision then determines what happens next: an action can be approved as-is (`approve`), modified before running (`edit`), rejected with feedback (`reject`), or respnoded to directly (`respond`) for "ask user" style tools.

| Decision Type | Description | Example Use Case |
| --- | --- | --- |
| `approve` | The action is approved as-is and executed without changes. | Send an email draft exactly as written. |
| `edit` | The tool call is executed with modifications. | Change the recipient before sending an email. |
| `reject` | The tool call is rejected, with an explanation added to the conversation. | Reject an email draft and explain how to rewrite it |
| `respond` | Tool execution is skipped; the human;s message becomes the tool result. | Answer the "ask_user" prompt with a direct reply |

With that being said, let's build an agent that can answer questions about a SQL database.

#### Boilerplate

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from dotenv import load_dotenv
import requests, pathlib


# --- Select an LLM and Configure the DB
load_dotenv()
llm = ChatOpenAI(model="gpt-5.4-nano")
url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"


def get_db():
    local_path = pathlib.Path("Chinook.db")
    if local_path.exists():
        print(f"{local_path} already exists, skipping download.")
    else:
        response = requests.get(url)
        if response.status_code == 200:
            local_path.write_bytes(response.content)
            print(f"File downloaded and saved as {local_path}")
        else:
            print(f"Failed to download the file. Status code: {response.status_code}")

    return SQLDatabase.from_uri("sqlite:///Chinook.db")


db = get_db()
print(f"Dialect: {db.dialect}")
print(f"Available tables: {db.get_usable_table_names()}")
print(f"Sample output: {db.run("SELECT * FROM Artist LIMIT 5;")}\n\n")


# --- Add Tools for Database Interactions
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

for tool in tools:
    print(f"{tool.name}: {tool.description}\n")


# --- Use `create_agent`
system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of teh qeury and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only as for the relevant coolumns given the question.

You MUST double check your query before executing it. If you get an error while executing a query, rewrite teh query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP, etc.) to the database.

To start, you should ALWAYS look at the tables in the database to see what you can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(dialect=db.dialect, top_k=5)

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)


# --- Run the Agent
question = "Which genre on average has the longest tracks?"
for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values"
):
    step["messages"][-1].pretty_print()
```

```
Chinook.db already exists, skipping download.
Dialect: sqlite
Available tables: ['Album', 'Artist', 'Customer', 'Employee', 'Genre', 'Invoice', 'InvoiceLine', 'MediaType', 'Playlist', 'PlaylistTrack', 'Track']
Sample output: [(1, 'AC/DC'), (2, 'Accept'), (3, 'Aerosmith'), (4, 'Alanis Morissette'), (5, 'Alice In Chains')]


sql_db_query: Input to this tool is a detailed and correct SQL query, output is a result from the database. If the query is not correct, an error message will be returned. If an error is returned, rewrite the query, check the query, and try again. If you encounter an issue with Unknown column 'xxxx' in 'field list', use sql_db_schema to query the correct table fields.

sql_db_schema: Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables. Be sure that the tables actually exist by calling sql_db_list_tables first! Example Input: table1, table2, table3

sql_db_list_tables: Input is an empty string, output is a comma-separated list of tables in the database.

sql_db_query_checker: Use this tool to double check if your query is correct before executing it. Always use this tool before executing a query with sql_db_query!

================================[1m Human Message [0m=================================

Which genre on average has teh longest trakcs?
==================================[1m Ai Message [0m==================================
Tool Calls:
  sql_db_list_tables (call_HVIZoH2QFZgCP272zMx1puoO)
 Call ID: call_HVIZoH2QFZgCP272zMx1puoO
  Args:
=================================[1m Tool Message [0m=================================
Name: sql_db_list_tables

Album, Artist, Customer, Employee, Genre, Invoice, InvoiceLine, MediaType, Playlist, PlaylistTrack, Track
==================================[1m Ai Message [0m==================================
Tool Calls:
  sql_db_schema (call_5M3WiPLO8yPOo2btY3oYgBzJ)
 Call ID: call_5M3WiPLO8yPOo2btY3oYgBzJ
  Args:
    table_names: Genre, Track
  sql_db_schema (call_S8dKjyNrC75JWuX6RZDZNPVD)
 Call ID: call_S8dKjyNrC75JWuX6RZDZNPVD
  Args:
    table_names: PlaylistTrack, Track
=================================[1m Tool Message [0m=================================
Name: sql_db_schema


CREATE TABLE "PlaylistTrack" (
	"PlaylistId" INTEGER NOT NULL, 
	"TrackId" INTEGER NOT NULL, 
	PRIMARY KEY ("PlaylistId", "TrackId"), 
	FOREIGN KEY("TrackId") REFERENCES "Track" ("TrackId"), 
	FOREIGN KEY("PlaylistId") REFERENCES "Playlist" ("PlaylistId")
)

/*
3 rows from PlaylistTrack table:
PlaylistId	TrackId
1	3402
1	3389
1	3390
*/


CREATE TABLE "Track" (
	"TrackId" INTEGER NOT NULL, 
	"Name" NVARCHAR(200) NOT NULL, 
	"AlbumId" INTEGER, 
	"MediaTypeId" INTEGER NOT NULL, 
	"GenreId" INTEGER, 
	"Composer" NVARCHAR(220), 
	"Milliseconds" INTEGER NOT NULL, 
	"Bytes" INTEGER, 
	"UnitPrice" NUMERIC(10, 2) NOT NULL, 
	PRIMARY KEY ("TrackId"), 
	FOREIGN KEY("MediaTypeId") REFERENCES "MediaType" ("MediaTypeId"), 
	FOREIGN KEY("GenreId") REFERENCES "Genre" ("GenreId"), 
	FOREIGN KEY("AlbumId") REFERENCES "Album" ("AlbumId")
)

/*
3 rows from Track table:
TrackId	Name	AlbumId	MediaTypeId	GenreId	Composer	Milliseconds	Bytes	UnitPrice
1	For Those About To Rock (We Salute You)	1	1	1	Angus Young, Malcolm Young, Brian Johnson	343719	11170334	0.99
2	Balls to the Wall	2	2	1	None	342562	5510424	0.99
3	Fast As a Shark	3	2	1	F. Baltes, S. Kaufman, U. Dirkscneider & W. Hoffman	230619	3990994	0.99
*/
==================================[1m Ai Message [0m==================================
Tool Calls:
  sql_db_query_checker (call_ymQdDbiv161miD70JfwDwzES)
 Call ID: call_ymQdDbiv161miD70JfwDwzES
  Args:
    query: SELECT g.Name AS Genre, AVG(t.Milliseconds)/1000.0 AS AvgTrackSeconds
FROM Genre g
JOIN Track t ON t.GenreId = g.GenreId
GROUP BY g.Name
ORDER BY AvgTrackSeconds DESC
LIMIT 5;
=================================[1m Tool Message [0m=================================
Name: sql_db_query_checker

SELECT g.Name AS Genre, AVG(t.Milliseconds)/1000.0 AS AvgTrackSeconds
FROM Genre g
JOIN Track t ON t.GenreId = g.GenreId
GROUP BY g.Name
ORDER BY AvgTrackSeconds DESC
LIMIT 5;
==================================[1m Ai Message [0m==================================
Tool Calls:
  sql_db_query (call_FSm3fRSlNuNyB7po4iU0uMqx)
 Call ID: call_FSm3fRSlNuNyB7po4iU0uMqx
  Args:
    query: SELECT g.Name AS Genre, AVG(t.Milliseconds)/1000.0 AS AvgTrackSeconds
FROM Genre g
JOIN Track t ON t.GenreId = g.GenreId
GROUP BY g.Name
ORDER BY AvgTrackSeconds DESC
LIMIT 5;
=================================[1m Tool Message [0m=================================
Name: sql_db_query

[('Sci Fi & Fantasy', 2911.7830384615386), ('Science Fiction', 2625.549076923077), ('Drama', 2575.28378125), ('TV Shows', 2145.041021505376), ('Comedy', 1585.263705882353)]
==================================[1m Ai Message [0m==================================

The genre with the **longest average track length** is **Sci Fi & Fantasy**, with an average duration of about **2911.78 seconds** (~48.53 minutes).
```

### Implement Human-in-the-Loop Pattern

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent

# --- Implement Human-in-the-Loop Review
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"sql_db_query": True},
            description_prefix="Tool execution pending approval",
        )
    ],
    checkpointer=InMemorySaver() # allows execution to be paused and resumed;
)

question = "Which genre on average has the longest tracks?"
config = {"configurable": {"thread_id": "1"}}

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    config,
    stream_mode="values"
):
    if "__interrupt__" in step:
        print("INTERRUPTED:")
        interrupt = step["__interrupt__"][0]
        for request in interrupt.value["action_requests"]:
            print(request["description"])
    elif "messages" in step:
        step["messages"][-1].pretty_print()
    else:
        pass
```

```
================================[1m Human Message [0m=================================

Which genre on average has the longest tracks?
==================================[1m Ai Message [0m==================================
Tool Calls:
  sql_db_list_tables (call_aWmYB4Mhu7J5bb2LArwRAPat)
 Call ID: call_aWmYB4Mhu7J5bb2LArwRAPat
  Args:
=================================[1m Tool Message [0m=================================
Name: sql_db_list_tables

Album, Artist, Customer, Employee, Genre, Invoice, InvoiceLine, MediaType, Playlist, PlaylistTrack, Track
==================================[1m Ai Message [0m==================================
Tool Calls:
  sql_db_schema (call_RREjiys9hnaOhrHpJrtf23Ga)
 Call ID: call_RREjiys9hnaOhrHpJrtf23Ga
  Args:
    table_names: Genre, Track
  sql_db_schema (call_s8R3DPZ0uLXPw4QZyIxudSq8)
 Call ID: call_s8R3DPZ0uLXPw4QZyIxudSq8
  Args:
    table_names: Track, Genre
=================================[1m Tool Message [0m=================================
Name: sql_db_schema


CREATE TABLE "Genre" (
	"GenreId" INTEGER NOT NULL, 
	"Name" NVARCHAR(120), 
	PRIMARY KEY ("GenreId")
)

/*
3 rows from Genre table:
GenreId	Name
1	Rock
2	Jazz
3	Metal
*/


CREATE TABLE "Track" (
	"TrackId" INTEGER NOT NULL, 
	"Name" NVARCHAR(200) NOT NULL, 
	"AlbumId" INTEGER, 
	"MediaTypeId" INTEGER NOT NULL, 
	"GenreId" INTEGER, 
	"Composer" NVARCHAR(220), 
	"Milliseconds" INTEGER NOT NULL, 
	"Bytes" INTEGER, 
	"UnitPrice" NUMERIC(10, 2) NOT NULL, 
	PRIMARY KEY ("TrackId"), 
	FOREIGN KEY("MediaTypeId") REFERENCES "MediaType" ("MediaTypeId"), 
	FOREIGN KEY("GenreId") REFERENCES "Genre" ("GenreId"), 
	FOREIGN KEY("AlbumId") REFERENCES "Album" ("AlbumId")
)

/*
3 rows from Track table:
TrackId	Name	AlbumId	MediaTypeId	GenreId	Composer	Milliseconds	Bytes	UnitPrice
1	For Those About To Rock (We Salute You)	1	1	1	Angus Young, Malcolm Young, Brian Johnson	343719	11170334	0.99
2	Balls to the Wall	2	2	1	None	342562	5510424	0.99
3	Fast As a Shark	3	2	1	F. Baltes, S. Kaufman, U. Dirkscneider & W. Hoffman	230619	3990994	0.99
*/
==================================[1m Ai Message [0m==================================
Tool Calls:
  sql_db_query_checker (call_X0GQTSjTlIx8bT4YVZNcf5yW)
 Call ID: call_X0GQTSjTlIx8bT4YVZNcf5yW
  Args:
    query: SELECT g.Name AS Genre, AVG(t.Milliseconds) AS AvgTrackMs, AVG(t.Milliseconds)/60000.0 AS AvgTrackMinutes
FROM Track t
JOIN Genre g ON g.GenreId = t.GenreId
GROUP BY g.Name
ORDER BY AvgTrackMs DESC
LIMIT 5;
=================================[1m Tool Message [0m=================================
Name: sql_db_query_checker

SELECT g.Name AS Genre, AVG(t.Milliseconds) AS AvgTrackMs, AVG(t.Milliseconds)/60000.0 AS AvgTrackMinutes
FROM Track t
JOIN Genre g ON g.GenreId = t.GenreId
GROUP BY g.Name
ORDER BY AvgTrackMs DESC
LIMIT 5;
==================================[1m Ai Message [0m==================================
Tool Calls:
  sql_db_query (call_yoc3dBJ63pUhCrG4jhzxEpH2)
 Call ID: call_yoc3dBJ63pUhCrG4jhzxEpH2
  Args:
    query: SELECT g.Name AS Genre, AVG(t.Milliseconds) AS AvgTrackMs, AVG(t.Milliseconds)/60000.0 AS AvgTrackMinutes
FROM Track t
JOIN Genre g ON g.GenreId = t.GenreId
GROUP BY g.Name
ORDER BY AvgTrackMs DESC
LIMIT 5;
INTERRUPTED:
Tool execution pending approval

Tool: sql_db_query
Args: {'query': 'SELECT g.Name AS Genre, AVG(t.Milliseconds) AS AvgTrackMs, AVG(t.Milliseconds)/60000.0 AS AvgTrackMinutes\nFROM Track t\nJOIN Genre g ON g.GenreId = t.GenreId\nGROUP BY g.Name\nORDER BY AvgTrackMs DESC\nLIMIT 5;'}
```

#### Resume Execution Using `Command`

```python

from langgraph.types import Command 

for step in agent.stream(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config,
    stream_mode="values",
):
    if "messages" in step:
        step["messages"][-1].pretty_print()
    if "__interrupt__" in step:
        print("INTERRUPTED:")
        interrupt = step["__interrupt__"][0]
        for request in interrupt.value["action_requests"]:
            print(request["description"])
    else:
        pass
```

```
==================================[1m Ai Message [0m==================================
Tool Calls:
  sql_db_query (call_yoc3dBJ63pUhCrG4jhzxEpH2)
 Call ID: call_yoc3dBJ63pUhCrG4jhzxEpH2
  Args:
    query: SELECT g.Name AS Genre, AVG(t.Milliseconds) AS AvgTrackMs, AVG(t.Milliseconds)/60000.0 AS AvgTrackMinutes
FROM Track t
JOIN Genre g ON g.GenreId = t.GenreId
GROUP BY g.Name
ORDER BY AvgTrackMs DESC
LIMIT 5;
==================================[1m Ai Message [0m==================================
Tool Calls:
  sql_db_query (call_yoc3dBJ63pUhCrG4jhzxEpH2)
 Call ID: call_yoc3dBJ63pUhCrG4jhzxEpH2
  Args:
    query: SELECT g.Name AS Genre, AVG(t.Milliseconds) AS AvgTrackMs, AVG(t.Milliseconds)/60000.0 AS AvgTrackMinutes
FROM Track t
JOIN Genre g ON g.GenreId = t.GenreId
GROUP BY g.Name
ORDER BY AvgTrackMs DESC
LIMIT 5;
=================================[1m Tool Message [0m=================================
Name: sql_db_query

[('Sci Fi & Fantasy', 2911783.0384615385, 48.52971730769231), ('Science Fiction', 2625549.076923077, 43.759151282051285), ('Drama', 2575283.78125, 42.92139635416667), ('TV Shows', 2145041.0215053763, 35.75068369175627), ('Comedy', 1585263.705882353, 26.421061764705883)]
==================================[1m Ai Message [0m==================================

On average, the **“Sci Fi & Fantasy”** genre has the longest tracks, with an average length of about **2,911,783 ms** (≈ **48.53 minutes**).
```

