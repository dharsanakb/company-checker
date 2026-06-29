"""
agent.py — the actual "agent" in this project.

This is a hand-written agentic loop (no LangChain / LangGraph):
1. We give the model ONE tool: search_web(query)
2. We send the model a system prompt that tells it to behave like a
   skeptical researcher: investigate from multiple angles, look for
   real patterns (not single complaints), and only stop once it has
   enough evidence.
3. The model decides what to search for, how many times, and when
   it's done. We just execute whatever tool calls it asks for and
   feed the results back in — that decision loop is what makes this
   "agentic" rather than a single hardcoded search + summarize.
"""

import json
from groq import Groq, BadRequestError
from tavily import TavilyClient

MODEL = "llama-3.3-70b-versatile"
MAX_TURNS = 10  # safety cap so the agent can't loop forever (includes any retries)

SYSTEM_PROMPT = """You are a careful, skeptical researcher helping a student \
decide whether a company is safe and worthwhile to intern or work at.

You have one tool: search_web(query). Use it to investigate the company from \
multiple angles before forming a verdict. Useful angles include:
- Employee reviews and work culture (Glassdoor, AmbitionBox, Reddit, Quora)
- Stipend/salary payment issues, delays, or unpaid-internship complaints
- Layoffs, lawsuits, scams, or regulatory/legal action
- Recent news, both good and bad
- Basic legitimacy (is it a real, registered company; known IT firm/consultancy; \
any "red flag" mentions on forums)

How you must work:
1. Run SEVERAL distinct searches (not just one) before concluding anything. \
Vary your queries, e.g. "<company> reviews", "<company> stipend delay reddit", \
"<company> scam complaints", "<company> layoffs news", "<company> glassdoor".
2. Call search_web with exactly ONE query string per call. Never pass a list \
of queries, and never request more than one search in the same turn — issue \
one search, look at the results, then decide on the next one.
3. Be skeptical of a single, isolated complaint — one angry review is noise, \
not a pattern. Only call something a real red flag if multiple independent \
sources point to the same issue.
4. Ignore generic SEO/listicle pages with no real substance — don't use them \
as evidence either way.
5. If you find little or nothing concerning, say so honestly. A clean result \
is a useful, valid finding — don't invent problems to seem thorough.
6. Once you've gathered enough evidence (usually 4-7 searches), STOP \
searching and write your final verdict. Do not call the tool again after that.

When you give your final answer, format it in markdown EXACTLY like this:

## Verdict: <Green Flag / Yellow Flag / Red Flag> — <one-line summary>

### What looks good
- bullet points, or "Nothing notable" if none

### What looks concerning
- bullet points, or "Nothing notable" if none

### Reasoning
2-4 sentences on how you weighed the evidence (why this is/isn't a real \
pattern, what's solid vs. uncertain).

### Sources
- [title](url) — one line per source actually used
"""

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for information and return the top results "
                        "(title, url, and a short content snippet for each).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to run.",
                }
            },
            "required": ["query"],
        },
    },
}


class CompanyCheckerAgent:
    def __init__(self, groq_api_key: str, tavily_api_key: str):
        self.client = Groq(api_key=groq_api_key)
        self.tavily = TavilyClient(api_key=tavily_api_key)

    def _search_web(self, query: str) -> str:
        """Executes the tool call the agent asked for, returns text the model can read."""
        results = self.tavily.search(query=query, max_results=5)
        lines = []
        for r in results.get("results", []):
            lines.append(f"- {r.get('title')} ({r.get('url')}): {r.get('content', '')[:400]}")
        return "\n".join(lines) if lines else "No results found."

    def investigate(self, company_name: str):
        """
        Generator that runs the agent loop and yields status events so the UI
        can show what the agent is doing live:
          ("search", query)
          ("done_search", query, num_results)
          ("final", markdown_text)
          ("error", message)
        """
        user_prompt = f"Investigate this company for someone considering an internship or job there: {company_name}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        consecutive_failures = 0

        for turn in range(MAX_TURNS):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=[SEARCH_TOOL],
                    tool_choice="auto",
                    parallel_tool_calls=False,  # ask for one tool call at a time —
                    # this model occasionally tries to batch several queries into
                    # one malformed call otherwise, which the API then rejects.
                )
            except BadRequestError as e:
                # The model generated a malformed tool call. Nudge it and retry
                # instead of crashing the whole investigation.
                consecutive_failures += 1
                yield ("retry", str(e))
                if consecutive_failures >= 3:
                    yield ("error", "The model kept generating malformed tool calls. Try again.")
                    return
                messages.append({
                    "role": "user",
                    "content": "That tool call was malformed. Call search_web again with "
                                "exactly ONE query string, not a list of queries.",
                })
                continue

            consecutive_failures = 0
            msg = response.choices[0].message

            if not msg.tool_calls:
                # No more tool calls -> the model gave its final answer
                yield ("final", msg.content)
                return

            # Record the assistant's tool-call request in the conversation
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })

            # Execute every requested tool call, feed results back
            for tool_call in msg.tool_calls:
                args = json.loads(tool_call.function.arguments)
                query = args.get("query", company_name)
                yield ("search", query)

                result_text = self._search_web(query)

                yield ("done_search", query, result_text.count("\n- ") + (1 if result_text != "No results found." else 0))

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                })

        yield ("error", "Agent hit the max search limit without giving a final verdict. Try again.")