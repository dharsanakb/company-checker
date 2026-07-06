"""
agent.py — the actual "agent" in this project.

This is a hand-written agentic loop (no LangChain / LangGraph):
1. We give the model FOUR tools: a general web search, plus three
   specialized checks (LinkedIn presence, official website, company
   registry/legitimacy).
2. We send the model a system prompt that tells it to behave like a
   skeptical researcher: investigate from multiple angles, run every
   mandatory check regardless of how things look early on, judge
   whether what it finds is a real pattern or just noise, and only
   stop once it has enough evidence.
3. The model decides what to call and in what order — we just execute
   whatever tool calls it asks for and feed the results back in. That
   decision loop is what makes this "agentic" rather than a single
   hardcoded search + summarize.

Note: we don't have dedicated LinkedIn / MCA-registry API keys wired up,
so check_linkedin_presence, check_official_website, and
check_company_registry are implemented as *specialized* Tavily searches —
each asks a narrowly-targeted question instead of a generic one. This
keeps the project to a single search provider (Tavily) while still giving
the agent four distinct, purpose-built tools to reason with.
"""

import json
from groq import Groq, BadRequestError
from tavily import TavilyClient

MODEL = "llama-3.3-70b-versatile"
MAX_TURNS = 14  # safety cap so the agent can't loop forever (includes any retries)

SYSTEM_PROMPT = """You are a careful, skeptical researcher helping a student \
decide whether a company is safe and worthwhile to intern or work at.

You have FOUR tools:
1. check_linkedin_presence(company_name) — checks whether the company has a \
real, active LinkedIn presence (employee count, hiring activity, recent posts).
2. check_official_website(company_name) — checks whether the company has a \
real, professional, active official website.
3. check_company_registry(company_name, country) — checks whether the company \
appears registered / legally traceable (e.g. MCA records for Indian companies, \
OpenCorporates-style registry data for others).
4. search_web(query) — general-purpose web search for reviews, news, stipend \
complaints, layoffs, scams, lawsuits, work culture, etc.

MANDATORY WORKFLOW — you must follow this every time, with NO exceptions:
1. Always call check_linkedin_presence, check_official_website, AND \
check_company_registry at least once each — even if the company is obviously \
a huge well-known name. These are basic legitimacy checks and must always run \
before you conclude anything. Do not skip them just because the company \
"looks legitimate" from its name alone.
2. After the three legitimacy checks, run SEVERAL distinct search_web \
searches (not just one) before concluding anything. Vary your queries, e.g. \
"<company> reviews", "<company> stipend delay reddit", "<company> scam \
complaints", "<company> layoffs news", "<company> glassdoor".
3. Call any tool with exactly ONE call per turn. Never batch multiple calls \
or pass a list of queries — issue one call, look at the result, then decide \
on the next one.
4. Be skeptical of a single, isolated complaint — one angry review is noise, \
not a pattern. Only call something a real red flag if multiple independent \
sources point to the same issue.
5. Ignore generic SEO/listicle pages with no real substance — don't use them \
as evidence either way.
6. A company having a strong LinkedIn/website/registry presence does NOT \
mean you can skip the search_web checks — legitimacy and trustworthiness are \
separate questions. A company can be 100% real and registered and still have \
a pattern of unpaid stipends or toxic culture. Always run both halves of the \
investigation: (a) legitimacy checks, (b) reputation/review searches.
7. If you find little or nothing concerning, say so honestly. A clean result \
is a useful, valid finding — don't invent problems to seem thorough.
8. Once you've run all 3 legitimacy tools AND gathered enough review/news \
evidence (usually 4-7 search_web calls), STOP and write your final verdict. \
Do not call any tool again after that.

When you give your final answer, format it in markdown EXACTLY like this:

## Verdict: <Green Flag / Yellow Flag / Red Flag> — <one-line summary>

### Legitimacy check
- LinkedIn: <finding>
- Official website: <finding>
- Company registry: <finding>

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

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_linkedin_presence",
            "description": "Checks whether the company has a real, active LinkedIn "
                            "page — employee count, hiring activity, recent posts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "The name of the company to check on LinkedIn.",
                    }
                },
                "required": ["company_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_official_website",
            "description": "Fetches and summarizes signals about the company's official "
                            "website to verify it looks real, active, and professional.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "The company name (or its website URL, if known) to check.",
                    }
                },
                "required": ["company_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_company_registry",
            "description": "Checks whether the company appears registered and legally "
                            "traceable (e.g. MCA records in India, OpenCorporates-style "
                            "registry data for other countries).",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "The company name to check in registry records.",
                    },
                    "country": {
                        "type": "string",
                        "description": "Country to check registration in. Default to "
                                        "'India' if unknown or unspecified.",
                    },
                },
                "required": ["company_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "General-purpose web search for reviews, news, stipend "
                            "complaints, layoffs, scams, lawsuits, or work culture. Returns "
                            "the top results (title, url, and a short content snippet).",
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
    },
]


class CompanyCheckerAgent:
    def __init__(self, groq_api_key: str, tavily_api_key: str):
        self.client = Groq(api_key=groq_api_key)
        self.tavily = TavilyClient(api_key=tavily_api_key)

    # ---- low-level search helper, shared by all tools ----
    def _tavily_search(self, query: str, max_results: int = 5) -> str:
        results = self.tavily.search(query=query, max_results=max_results)
        lines = []
        for r in results.get("results", []):
            lines.append(f"- {r.get('title')} ({r.get('url')}): {r.get('content', '')[:400]}")
        return "\n".join(lines) if lines else "No results found."

    # ---- the four tools the agent can call ----
    def _search_web(self, query: str) -> str:
        """General-purpose search: reviews, news, complaints, culture, etc."""
        return self._tavily_search(query)

    def _check_linkedin_presence(self, company_name: str) -> str:
        """Specialized search targeting LinkedIn signals."""
        query = f"{company_name} LinkedIn company page employees hiring"
        return self._tavily_search(query)

    def _check_official_website(self, company_name: str) -> str:
        """Specialized search targeting the company's official website."""
        query = f"{company_name} official website about us careers"
        return self._tavily_search(query)

    def _check_company_registry(self, company_name: str, country: str = "India") -> str:
        """Specialized search targeting legal/registry traceability."""
        if country.strip().lower() in ("india", ""):
            query = f"{company_name} MCA company registration CIN registered company India"
        else:
            query = f"{company_name} company registration OpenCorporates {country}"
        return self._tavily_search(query)

    def _execute_tool(self, name: str, args: dict) -> str:
        """Routes a tool call by name to the right method."""
        if name == "search_web":
            return self._search_web(args.get("query", ""))
        if name == "check_linkedin_presence":
            return self._check_linkedin_presence(args.get("company_name", ""))
        if name == "check_official_website":
            return self._check_official_website(args.get("company_name", ""))
        if name == "check_company_registry":
            return self._check_company_registry(
                args.get("company_name", ""), args.get("country", "India")
            )
        return f"Unknown tool: {name}"

    def investigate(self, company_name: str):
        """
        Generator that runs the agent loop and yields status events so the UI
        can show what the agent is doing live:
          ("search", tool_name, query_or_args)
          ("done_search", tool_name, num_results)
          ("retry", error_message)
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
                    tools=TOOLS,
                    tool_choice="auto",
                    parallel_tool_calls=False,  # ask for one tool call at a time —
                    # this model occasionally tries to batch several calls into
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
                    "content": "That tool call was malformed. Call exactly ONE tool with "
                                "ONE clean set of arguments, not a list.",
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
                tool_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                # human-readable label for the UI
                label = args.get("query") or args.get("company_name") or company_name
                yield ("search", tool_name, label)

                result_text = self._execute_tool(tool_name, args)

                yield ("done_search", tool_name, result_text.count("\n- ") + (1 if result_text != "No results found." else 0))

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                })

        yield ("error", "Agent hit the max search limit without giving a final verdict. Try again.")
