---
layout: notebook
title: "Goal Setting and Monitoring"
project: agentic-patterns
order: 11
---

## Goal Setting and Monitoring

### Overview
Goal Setting and Monitoring is about giving agents specific objectives to work towards and equipping them with the means to track their progress and determine if those objectives have been met. Planning typically involves an agent taking a high-level objective and autonomously, or semi-autonomously, generating a series of itermediate steps or sub-goals. These steps can then be executed sequentially or in a more complex flow, potentially involving other patterns like tool use, routing, or multi-agent collaboration. The planning mechanism might involve sophisticated search algorithms, logical reasoning, or increaingly, leveraging the capabilities of LLMs to generate plausible and effective plans based on their training data and understanding of tasks.

### Practical Applications & Use Cases
This pattern is essential for building agents that can operate autonomously and realiably in complex, real-world scenarios, such as:

- Customer Support Automation: An agent's goal might be to "resolve customer's billing inquiry". It monitors the conversation, checks database entries, and uses tools to adjust billing. Success is monitored by confirming the billing change and receiving positive customer feedback. If the issue isn't resolved, it escalates.

- Personalized Learning Systems: A learning agent might have the goal to "improve students' understanding of algebra". It monitors the student's progress on exercises, adapts teaching materials, and tracks performance metrics like accuracy and completion time, adjusting its approach if the student struggles.

- Project Management Assistants: An agent could be tasked with "ensuring project milestone X is completed by Y date". It moinmonitors task statuses, team communications, and resource availability, flagging delays and suggesting corrective actions if the goal is at risk.

- Automated Trading Bots: A trading agent's goal might be to "maximize portfolio gains while staying within risk tolerance". It continuously monitors market data, its current portfolio value, and risk indicators, executing trades when conditions alight with its goals and adjusting strategy of risk thresholds are breached.

- Robotics and Autonomous Vehicles: An autonomous vehicle's primary goal is "safety transport passengers from A to B". It constantly monitors its environment (other vehicles, pedestrians, traffic signals), its own state (speed, fuel), and its progressalong the planned route, adapting its driving behavior to achieve the goal safely and efficiently.

- Content Moderation: An agent's goal could be to "identify and remove harmful content from platform X". It monitors incoming content, applies calssification models, and tracks metrics like false positives/negatives, adjusting its filtering criteria or scalating ambiguous casess to human reviewers.

The pattern is fundamental for agents thatneed to operate reliably, achieve specific outcomes, and adapt to dynamic conditions, providing the necessary framework for intelligent self-management.

### Hands-On Code Example
This scipt shows an autonomous AI agent engineered to generate and refine Python code. Its core function is to produce solutions for specified problems, ensuring adherence to user-defined quality benchmarks.

```python
"""
- Accepts a coding problem (use case) in code or can be as input.
- Accepts a list of goals (e.g., "simple", "tested", "handles edge cases") in code or can be input.
- Uses an LLM to generate and refine Python code until the goals are met (with max 5 iterations).
- LLM answers with True/False to check if goals have been met.
- Saves the final code in a .py file with a clean filename and a header comment.
"""

import os
import random
import re
from dotenv import get_key
from pathlib import Path

from langchain_openai import ChatOpenAI

# --- Configuration
API_KEY = get_key("../.env", "OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0.3,
    api_key=API_KEY
)


# --- Utility Functions
def generate_prompt(use_case: str, goals: list[str], previous_code: str="", feedback: str="") -> str:
    print("-- Constructing prompt for code generation...")
    
    base_prompt = f"""
    You are an AI coding agent. Your job is to write Python code based on the following use case:

    Use Case: {use_case}

    Your goals are: 
    {chr(10).join(f"- {g.strip()}" for g in goals)}
    """

    if previous_code:
        print("Adding previous code to the prompt for refinement.")
        base_prompt += f"\nPreviously generated code: \n{previous_code}"
    if feedback:
        print("Including feedback for revision.")
        base_prompt += f"\nFeedback on the previous version: \n{feedback}\n"

    base_prompt += "\nReturn only the revised Python code. Do not include comments or explanations outside the code."

    return base_prompt


def get_code_feedback(code: str, goals: list[str]) -> str:
    print("Evaluating code against the goals...")
    feedback_prompt = f"""
    You are a Python code reviewer. A code snippet is shown below. Based on the following goals:
    {chr(10).join(f"- {g.strip()}" for g in goals)}
    
    Please critique this code and identify if the goals are met. Mention if improvements are needed for clarity, simplicity, correctness, or test coverage.
    Code:
    {code}
    """
    return llm.invoke(feedback_prompt)


def goals_met(feedback_text: str, goals: list[str]) -> bool:
    """
    Uses the LLM to evaluate whether the goals have been met based on the feedback text.
    Returns True or False (parsed from LLM output).
    """
    review_prompt = f"""
    You are an AI reviewer.

    Here are the goals:
    {chr(10).join(f"- {g.strip()}" for g in goals)}

    Here's the feedback on the code:
    \"\"\"
    {feedback_text}
    \"\"\"

    Based on the feedback above, have the goals been met?
    Respond with only one word: True or False.
    """
    response = llm.invoke(review_prompt).content.strip().lower()
    
    return response == "true"


# test if this gets rid of indentation?
def clean_code_block(code: str) -> str:
    lines = code.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def add_comment_header(code: str, use_case: str) -> str:
    comment = f"# This Python program implements the following use case:\n# {use_case.strip()}\n"
    return comment + "\n" + code


def to_snake_case(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return re.sub(r"\s+", "_", text.strip().lower())


def save_code_to_file(code: str, use_case: str) -> str:
    print("Saving final code to file...")

    summary_prompt = (
        f"Summarize the following use case into a single lowercasre word or phrase, "
        f"no more than 10 characters, suitable for a Python filename:\n\n{use_case}"
    )
    raw_summary = llm.invoke(summary_prompt).content.strip()
    short_name = re.sub(r"^a-zA-Z0-9_", "", raw_summary.replace(" ", "_").lower()[:10])

    random_suffix = str(random.randint(1000, 9999))
    filename = f"{short_name}_{random_suffix}.py"
    filepath = Path.cwd() / filename
    
    with open(filepath, "w") as f:
        f.write(code)

    print(f"Code saved to: {filepath}")
    return str(filepath)


def run_code_agent(use_case: str, goals_input: str, max_iterations: int=5) -> int:
    goals = [g.strip() for g in goals_input.split(",")]

    print(f"\nUse Case: {use_case}")
    print("\nGoals:")
    for g in goals:
        print(f"  - {g}")

    previous_code = ""
    feedback = ""

    for i in range(max_iterations):
        print(f"\n=== Iteration {i + 1} of {max_iterations} ===")
        prompt = generate_prompt(
            use_case,
            goals,
            previous_code,
            feedback if isinstance(feedback, str) else feedback.content
        )

        print("Generating Code...")
        code_response = llm.invoke(prompt)
        raw_code = code_response.content.strip()
        code = clean_code_block(raw_code)
        print("\n Generated Code:\n" + "-" * 50 + f"\n{code}\n" + "-" * 50)

        print("\nSubmitting Code for feedback review...")
        feedback = get_code_feedback(code, goals)
        feedback_text = feedback.content.strip()
        print("\nFeedback Received:\n" + "-" * 50 + f"\n{feedback_text}\n" + "_" * 50)

        if goals_met(feedback_text, goals):
            print("LLM confirms goals are met. Stopping iterations.")
            break

        print("Goals not fully met. Preparing for next iteration...")
        previous_code = code

    final_code = add_comment_header(code, use_case)
    return save_code_to_file(final_code, use_case)


if __name__ == "__main__":
    print("Welcome to the AI Code Generating Agent")

    # Example 1
    use_case_input = "Write code which takes a command line input of a word doc or docx file and opens it and counts the number of words, and characters in it and prints all"
    goals_input = "Code simple to understand, Functionally correct, Handles edge cases"
    run_code_agent(use_case_input, goals_input)
```

```
Welcome to the AI Code Generating Agent

Use Case: Write code which takes a command line input of a word doc or docx file and opens it and counts the number of words, and characters in it and prints all

Goals:
  - Code simple to understand
  - Functionally correct
  - Handles edge cases

=== Iteration 1 of 5 ===
-- Constructing prompt for code generation...
Generating Code...

 Generated Code:
--------------------------------------------------
import sys
import os
import subprocess
import tempfile
import shutil

def extract_text_from_docx(path):
    try:
        from docx import Document
    except Exception:
        raise RuntimeError("python-docx library is required to read .docx files. Install via 'pip install python-docx'.")
    doc = Document(path)
    parts = []
    for para in doc.paragraphs:
        if para.text:
            parts.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text:
                    parts.append(cell.text)
    return "\n".join(parts)

def convert_doc_to_docx(input_path):
    tmpdir = tempfile.mkdtemp()
    try:
        subprocess.run(["soffice", "--headless", "--convert-to", "docx", "--outdir", tmpdir, input_path],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for fname in os.listdir(tmpdir):
            if fname.lower().endswith(".docx"):
                return os.path.join(tmpdir, fname), tmpdir
    except Exception:
        pass
    return None, None

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_doc_or_docx>")
        return

    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print("Error: File not found:", input_path)
        return

    ext = os.path.splitext(input_path)[1].lower()
    docx_path = None
    tmpdir = None

    try:
        if ext == ".docx":
            docx_path = input_path
        elif ext == ".doc":
            converted, tmpdir = convert_doc_to_docx(input_path)
            if not converted:
                print("Error: Could not convert .doc to .docx. Ensure LibreOffice is installed.")
                return
            docx_path = converted
        else:
            print("Error: Unsupported file format. Please provide a .docx or .doc file.")
            return

        text = extract_text_from_docx(docx_path)
        words = len(text.split())
        chars = len(text)

        print("File:", input_path)
        print("Word count:", words)
        print("Character count (including spaces):", chars)
    except RuntimeError as e:
        print("Error:", e)
    except Exception as e:
        print("An unexpected error occurred:", e)
    finally:
        if tmpdir and os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()
--------------------------------------------------

Submitting Code for feedback review...
Evaluating code against the goals...

Feedback Received:
--------------------------------------------------
Overall assessment
- The script achieves its core goal: it can read a .docx (or convert a .doc via LibreOffice), extract text, and report word and character counts. It’s fairly straightforward and easy to follow.
- However, there are several areas where clarity, robustness, and testability could be improved. The current code relies heavily on external tools (LibreOffice’s soffice and the python-docx package) and swallows some errors, which hurts reliability and testability in edge cases.

What works well
- Clear separation of responsibilities:
  - extract_text_from_docx: reads text from a .docx using python-docx (paragraphs and tables).
  - convert_doc_to_docx: attempts to convert a .doc to .docx using soffice in headless mode.
  - main: wires up CLI input, decides whether to use the input as-is or convert, then computes word/character counts.
- Basic error messaging for missing dependencies and unsupported formats.
- Proper cleanup of the temporary directory used for conversion.

What could be improved (clarity, simplicity, correctness, edge cases)
- Dependency handling and error reporting
  - convert_doc_to_docx swallows exceptions and returns (None, None) on any error. This hides the root cause (e.g., soffice not installed, permission issues, invalid input file). It would be better to catch specific exceptions and propagate meaningful error information upward (or log it) for easier debugging.
  - extract_text_from_docx imports python-docx inside the function and raises a RuntimeError if the library isn’t installed. This is fine, but consider using a clearer error type (import-time import vs runtime import) and a more explicit message if the file is not a valid .docx.
- Error handling in main
  - The script prints generic error messages and continues to the next step only on certain conditions. If extract_text_from_docx raises, main prints a generic “An unexpected error occurred” message. It would be nicer to differentiate between user-facing errors (bad input, missing dependencies) and unexpected exceptions, possibly with a short traceback suppressed in CLI tools.
- External dependency assumptions
  - The code assumes soffice (LibreOffice) is installed for .doc conversion. If the user provides a .doc file but soffice is missing or not in PATH, the error messaging could be clearer. Consider a pre-check or a more explicit error message in the conversion path.
- Path handling and readability
  - Use of os.path is fine, but modern Python often benefits from pathlib for readability and easier path manipulations.
  - The code uses a simple text join with newline to combine paragraph and table text. This is fine for many cases but note that it won’t capture text in all Word content areas (headers/footers, text boxes, etc.). If those are important, document this limitation or extend extraction to cover more Word elements.
- Edge cases not fully covered
  - Password-protected or corrupted .docx/.doc files: not handled explicitly.
  - Very large documents: memory usage could be high since it builds a full text string before counting words. This may be acceptable for typical documents but could be improved with streaming or chunked counting if needed.
  - Non-ASCII filenames and paths: generally okay, but you may want to ensure robust encoding handling in all prints and messages.
- Testability
  - Integration tests would require LibreOffice and python-docx installed, making them flaky in CI environments. Consider approaches to improve testability:
    - Add unit tests for extract_text_from_docx using a small, in-repo .docx created with python-docx (skip if python-docx isn’t installed).
    - Add tests that mock convert_doc_to_docx and/or soffice behavior so tests don’t depend on LibreOffice availability.
    - For main, tests can mock dependencies and verify the messaging logic without invoking soffice or reading real documents.

Suggested improvements (concrete ideas)
- Improve error handling and messaging
  - Catch specific exceptions in convert_doc_to_docx (e.g., FileNotFoundError for soffice not found, CalledProcessError for non-zero exit status) and propagate a clear error message up to main.
  - In main, differentiate user-facing errors from unexpected ones. Consider using a small helper to print friendly messages and optionally show a traceback if a verbose flag is set.
- Typing and documentation
  - Add type hints to functions and docstrings to improve readability and tooling support.
  - Example: extract_text_from_docx(path: str) -> str
- Path handling and readability
  - Use pathlib.Path for path manipulations to simplify code and improve readability.
  - Avoid mixing os.path with pathlib; pick one approach.
- Testing strategy (outline)
  - Unit tests for extract_text_from_docx that:
    - Create a small .docx with known content (paragraphs and a simple table) using python-docx (if available).
    - Assert the exact concatenated text or at least the presence/ordering of certain strings.
  - Unit tests for convert_doc_to_docx that:
    - Mock subprocess.run to simulate success and failure, returning a .docx path under tmpdir or None.
  - Integration tests with mocks:
    - Mock convert_doc_to_docx to return a fake docx path and mock extract_text_from_docx to return a known text, then verify the counts and output formatting.
  - Optional: a lightweight CLI test that patches sys.argv and logs outputs without invoking external programs.
- Small stylistic/robustness tweaks
  - Consider adding a short timeout for the soffice call to avoid hanging if the conversion stalls.
  - Consider using a context manager for temporary resources, or at least log the created tmpdir path when debugging.

Minimal patch sketch (conceptual, not full code)
- Replace broad except Exception with targeted error handling in convert_doc_to_docx:
  - Try running soffice; if FileNotFoundError -> raise a clear RuntimeError("LibreOffice (soffice) not found in PATH. Install LibreOffice to enable .doc to .docx conversion.")
  - If subprocess.CalledProcessError -> raise a RuntimeError("Document conversion failed. Ensure the input file is valid and LibreOffice can access it.")
  - On success, locate the produced .docx as before.
- Add lightweight docstring and typing to functions
  - def extract_text_from_docx(path: str) -> str:
- Optional: switch to pathlib
  - from pathlib import Path
  - Use Path for input_path and tmpdir paths, improving readability and robustness.
- Add a basic unit test scaffolding plan (pytest-friendly)
  - Skip tests if python-docx isn’t installed, or mark with pytest.importorskip("docx") as appropriate.
  - Test the file-not-found path and unsupported extension path in main logic by simulating sys.argv and capturing stdout/stderr.

Bottom line
- The code is functional and fairly understandable, and it handles the basic workflow and edge cases like empty documents and unsupported formats.
- To meet higher standards of robustness, testability, and clarity, consider clarifying and hardening error handling (especially around the external conversion step), adopting pathlib, adding docstrings/type hints, and improving test coverage with unit tests that don’t require LibreOffice or python-docx to run in all environments.
- If you expect this to run in various environments or in CI, prioritize adding mocks for the external dependencies and creating tests that don’t rely on soffice or python-docx being installed.
__________________________________________________
Goals not fully met. Preparing for next iteration...

=== Iteration 2 of 5 ===
-- Constructing prompt for code generation...
Adding previous code to the prompt for refinement.
Including feedback for revision.
Generating Code...

 Generated Code:
--------------------------------------------------
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple

def convert_doc_to_docx(input_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    tmpdir = Path(tempfile.mkdtemp())
    try:
        result = subprocess.run(
            ["soffice", "--headless", "--convert-to", "docx", "--outdir", str(tmpdir), str(input_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        raise RuntimeError("LibreOffice (soffice) not found in PATH. Install LibreOffice to enable .doc to .docx conversion.")
    if result.returncode != 0:
        err = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"Document conversion failed: {err or 'Unknown error'}")

    converted = None
    for fname in tmpdir.iterdir():
        if fname.suffix.lower() == ".docx":
            converted = fname
            break

    if converted:
        return converted, tmpdir
    return None, tmpdir

def count_text_in_docx(docx_path: Path) -> Tuple[int, int]:
    try:
        from docx import Document
    except Exception:
        raise RuntimeError("python-docx is required to read .docx files. Install via 'pip install python-docx'.")

    words = 0
    chars = 0
    doc = Document(str(docx_path))

    for para in doc.paragraphs:
        text = para.text
        if text:
            words += len(text.split())
            chars += len(text)

    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                text = cell.text
                if text:
                    words += len(text.split())
                    chars += len(text)

    return words, chars

def main():
    parser = argparse.ArgumentParser(
        description="Count words and characters in a Word document (.doc or .docx)."
    )
    parser.add_argument("path", help="Path to a .docx or .doc file")
    args = parser.parse_args()

    input_path = Path(args.path)
    if not input_path.exists():
        print("Error: File not found:", input_path)
        return

    ext = input_path.suffix.lower()
    docx_path: Optional[Path] = None
    tmpdir: Optional[Path] = None

    try:
        if ext == ".docx":
            docx_path = input_path
        elif ext == ".doc":
            converted, tmpdir = convert_doc_to_docx(input_path)
            if not converted:
                print("Error: Could not convert .doc to .docx. Ensure LibreOffice is installed.")
                return
            docx_path = converted
        else:
            print("Error: Unsupported file format. Please provide a .docx or .doc file.")
            return

        words, chars = count_text_in_docx(docx_path)

        print("File:", input_path)
        print("Word count:", words)
        print("Character count (including spaces):", chars)
    except RuntimeError as e:
        print("Error:", e)
    except Exception as e:
        print("An unexpected error occurred:", e)
    finally:
        if tmpdir and tmpdir.exists():
            shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()
--------------------------------------------------

Submitting Code for feedback review...
Evaluating code against the goals...

Feedback Received:
--------------------------------------------------
Here is a focused code review. Overall, the script is functional and reasonably simple, but there are areas to improve for clarity, robustness, and testability.

What the code does well
- Clear separation of concerns:
  - convert_doc_to_docx handles conversion via LibreOffice (soffice).
  - count_text_in_docx reads a .docx and counts words and characters (including those in tables).
  - main wires the CLI, argument parsing, and overall flow.
- Uses pathlib.Path and type hints, which improves readability and reliability.
- Handles common edge cases:
  - Validates input path existence.
  - Handles .doc input by converting to .docx.
  - Handles missing soffice with a helpful error.
  - Handles missing python-docx with a helpful error.
- Edge-safe behaviors:
  - Case-insensitive handling of extensions.
  - Cleanup of temporary directory in main via finally.

What could be improved (and why)
1) Testability and test coverage
- There are no tests. The code relies on external software (LibreOffice) and an optional Python package (python-docx). If you want reliable tests:
  - Use pytest and mock subprocess.run to simulate soffice behavior for convert_doc_to_docx.
  - Test count_text_in_docx by generating a small docx with python-docx (if available) or by mocking docx.Document.
  - Test the missing-python-docx path by simulating ImportError when importing docx.
- Suggested tests to add:
  - test_convert_doc_to_docx_success: mock subprocess.run to return returncode 0 and create a fake .docx in tmpdir; verify the function returns the path to that .docx and the tmpdir path.
  - test_convert_doc_to_docx_failure: mock subprocess.run to return non-zero; verify a RuntimeError with a meaningful message.
  - test_count_text_in_docx_counts_correctly: create a tiny .docx with a couple of paragraphs and a small table, verify (words, chars) matches expectation.
  - test_count_text_in_docx_missing_dependency: simulate missing python-docx and verify a RuntimeError is raised.
- Practical note: Some environments won’t have LibreOffice or python-docx, so tests should be designed to run with mocks/patches and not require those dependencies.

2) Robustness and reliability
- Temporary directory handling:
  - convert_doc_to_docx creates a tmpdir and returns it to the caller for cleanup. This is fine, but relies on the caller to always cleanup (which main does). If convert_doc_to_docx is used elsewhere, the caller might forget to cleanup. Consider documenting this contract clearly, or refactor to use an explicit cleanup function or a context manager in a higher-level function.
- Multiple .docx in tmpdir:
  - The code picks the first .docx file found. In practice, soffice will place exactly one output, but it’s worth documenting that this code assumes a single new docx in tmpdir. If you ever extend it, you might want to verify there’s exactly one .docx or use a deterministic naming convention.
- Error handling consistency:
  - In main, only RuntimeError is caught with a clean message; other exceptions are reported as “An unexpected error occurred” which is fine for a CLI, but consider distinguishing user-facing errors from programming errors (e.g., by logging details or exiting with non-zero status).

3) Clarity and API surface
- Docstrings and small comments would improve readability:
  - Convert functions currently lack docstrings. A brief description of inputs/outputs, and the error semantics, would help future maintainers.
- Return type semantics:
  - convert_doc_to_docx returns a Tuple[Optional[Path], Optional[Path]]. It’s a bit opaque. A small NamedTuple or dataclass (e.g., ConversionResult with fields converted_path and tmpdir) would make usage clearer.

4) Exit codes and CLI ergonomics
- The program prints errors and returns (implicitly exit code 0/None). For a CLI, consider returning non-zero exit codes on error (e.g., return 1) and using sys.exit to reflect success/failure. This makes it easier to automate usage in scripts.
- main currently prints to stdout for normal results and to stdout for errors (via “Error:” prints). Using a single error pathway or logging could make testing easier, but this is not critical.

5) Minor correctness and style improvements
- Cleaning up on early exit:
  - The finally block cleans tmpdir if it exists, which is good. If you add more early-return paths in the future, ensure the cleanup still happens.
- Counting logic:
  - The word count uses text.split(), which is acceptable but has caveats (e.g., handling hyphenated words, non-breaking spaces). For typical Word content, this is fine. If you need stricter counting, you could implement a small tokenizer or rely on python-docx’s text processing more consistently.
- Dependency messaging:
  - The error message for missing python-docx is good, but you could include a suggestion like “pip install python-docx”.

Concrete suggestions (minimal changes you could consider)
- Add small docstrings to the functions:
  - convert_doc_to_docx(input_path: Path) -> Tuple[Optional[Path], Optional[Path]]: description of return semantics.
  - count_text_in_docx(docx_path: Path) -> Tuple[int, int]: description of how text is counted.
- Return a non-zero exit code on errors in main:
  - e.g., if input path not found, print and return 1.
- Introduce a lightweight named tuple for the conversion result to clarify the return values:
  - from typing import NamedTuple
  - class ConversionResult(NamedTuple):
      converted: Optional[Path]
      tmpdir: Optional[Path]
  - And return ConversionResult(converted, tmpdir) for clarity.
- Add one or two targeted tests (as described above) to cover core paths and error paths.

In summary
- The code is reasonably simple and functionally correct for typical use cases.
- It handles the primary edge cases (missing file, missing external tools, missing dependencies).
- It could be improved with docs, explicit exit codes, and better test coverage, plus a small refactor to improve clarity around the conversion result.
- If you plan to maintain this long-term or integrate it into a larger project, add tests and consider a small refactor to make the API surface and error handling more explicit.
__________________________________________________
LLM confirms goals are met. Stopping iterations.
Saving final code to file...
Code saved to: /Users/ikromshi/coding/ml/agentic-design-patterns/chapter-2/wordchars_2691.py
```

### Key Takeaways
- Goal Setting and Monitoring equipts agents with purpose and mechanisms to track progress.

- Goals should be specific, measurable, achievable, relevant, and time-bound (SMART).

- Clearly defining metrics and success criteria is essential for effective monitoring.

- Monitoring involves observing agent actions, environmental states, and tool outputs.

- Feedback loops from monitoring allow agents to adapt, revise plans, or escalate issues.

