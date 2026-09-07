#  Internship / Company Red-Flag Checker

An agentic AI tool that researches a company before you intern or work there.

You give it a company name. Instead of running one search and summarizing
results, it behaves like a careful researcher — deciding what to look into,
judging whether what it finds is a real pattern or just noise, and giving you
an honest verdict with sources.

---

## What it produces

```
## Verdict: Red Flag — Multiple credible reports of delayed stipends and no mentorship

### Legitimacy check
- LinkedIn presence: Active page, ~120 employees, last post 3 months ago
- Official website: Loads fine, but contact page has no real address
- Registry status: No clear MCA registration found under this exact name

### What looks good
- Active social media presence and professional website

### What looks concerning
- 3 independent Reddit/Quora threads about unpaid stipends (2023–2024)
- No structured learning reported by any former interns
- AmbitionBox rating: 2.1/5 across 40+ reviews
- No verifiable company registration found

### Reasoning
The stipend delay pattern appears across multiple independent sources over
two years, which makes it a real pattern rather than isolated noise. The
unclear registry status adds to the concern.

### Sources
- [Company Reviews on AmbitionBox](https://ambitionbox.com/...)
- [Reddit thread: internship experience](https://reddit.com/...)
```

---

## The agent's tools

The agent has **four tools** it can call, and it decides which ones to use
and in what order based on what it finds:

| Tool | Purpose | How it's implemented |
|---|---|---|
| `search_web(query)` | General search — reviews, complaints, news, scams, layoffs | Tavily search |
| `check_linkedin_presence(company_name)` | Checks for a real, active LinkedIn page and hiring activity | Tavily search scoped to `site:linkedin.com/company` |
| `check_official_website(url_or_company)` | Fetches the company's real website and checks if it looks legitimate | Tavily search (to find the URL) + direct page fetch with `requests`/`BeautifulSoup` |
| `check_company_registry(company_name, country)` | Checks for legal/registration traces (MCA, OpenCorporates, etc.) | Tavily search scoped to registry-type sources |

**Why these are built on Tavily instead of dedicated APIs:** Real LinkedIn,
MCA, and OpenCorporates APIs either require paid access, special approval,
or extra API keys that aren't free/instant to get. Since Tavily search can
reliably surface the *same* information (a company's LinkedIn page, its
MCA/Zaubacorp listing, etc. all show up in normal search results), routing
all four tools through Tavily keeps the project to a single search
dependency — simpler to set up, explain, and maintain, with identical
practical results for this use case.

---

## Project structure

```
company-checker/
├── agent.py          # The agent: LLM + tool + loop. Core of the project.
├── app.py            # Streamlit UI. Just renders what the agent yields.
├── requirements.txt  # Dependencies
├── .env.example      # API key template
└── README.md         # This file
```

---

## Tech stack — what was used and why

###  Groq API (`openai/gpt-oss-120b`)
**What it is:** A cloud inference platform that runs open-source LLMs
on custom-built hardware called LPUs (Language Processing Units), making
them extremely fast.

**Why Groq:**
- Completely free tier — no credit card required.
  (We tried Gemini, which gave a 403 "project denied" error, and OpenAI,
  which requires billing even to start. Groq just works.)
- Supports function/tool calling, which the agentic loop depends on.
- Fast — results come back in seconds even for a multi-step agent run.

**Why `openai/gpt-oss-120b`:**
- OpenAI's flagship open-weight model with 120 billion parameters —
  large enough to follow nuanced system prompt instructions reliably,
  including "don't call something a red flag unless multiple sources agree"
  and "call the tool once per turn, not in batches."
- Replaced `llama-3.3-70b-versatile`, which Groq deprecated on
  June 17, 2026. `openai/gpt-oss-120b` is Groq's recommended replacement
  for the 70B Llama tier and delivers stronger reasoning performance.

---

###  Four tools, all backed by Tavily
**What they are:** The agent has four distinct tools to call — one general
search, and three specialized legitimacy checks:

| Tool | Purpose |
|------|---------|
| `check_linkedin_presence(company_name)` | Real LinkedIn page, employee count, hiring activity, recent posts |
| `check_official_website(company_name)` | A real, professional, active official website |
| `check_company_registry(company_name, country)` | Registered / legally traceable (MCA records for India, OpenCorporates-style data elsewhere) |
| `search_web(query)` | General reviews, news, stipend complaints, layoffs, scams, lawsuits |

**Important note:** there's no dedicated LinkedIn API, MCA API, or
OpenCorporates key wired up here. Each of the three "legitimacy" tools is
implemented as a *specialized Tavily search* with a narrowly-targeted query
(e.g. `check_linkedin_presence` searches `"<company> LinkedIn company page
employees hiring"`). This keeps the project to one search provider while
still giving the agent four distinct, purpose-built tools to reason with.
If you later get real API access (LinkedIn API, MCA portal, OpenCorporates),
you'd only need to swap the implementation inside each `_check_*` method in
`agent.py` — the tool interface the LLM sees stays identical.

**Why legitimacy checks matter separately from reviews:** a company can be
100% real, registered, and have a polished LinkedIn page — and *still* have
a pattern of unpaid stipends or toxic culture. Legitimacy and trustworthiness
are different questions, so the system prompt forces the agent to run **both
halves** of the investigation, every time, regardless of how clean things
look early on (see "Mandatory workflow" below).

---

###  Streamlit
**What it is:** A Python library that turns a plain `.py` script into a
web app — no HTML, CSS, or JavaScript required.

**Why Streamlit:**
- Single file UI (`app.py` is ~35 lines).
- `st.status()` lets us show the agent's live progress — each search
  query appears in real time as the agent runs it, instead of a blank
  screen until the verdict arrives.
- No separate frontend/backend — you just run `streamlit run app.py`
  and it opens in the browser.

---

###  Hand-written agentic loop (no LangChain / LangGraph)
**What it is:** The loop in `agent.py` that decides when to search and
when to stop — written from scratch in ~60 lines of Python.

**Why no framework:**
- Transparency: every line of the loop is readable. In an interview you
  can explain exactly what happens at each step.
- Simplicity: LangChain/LangGraph add hundreds of lines of abstraction
  for something that is, at its core, a while-loop with an if-statement.
- Control: we added `parallel_tool_calls=False` and a
  `try/except BadRequestError` retry handler because open-weight models
  on Groq occasionally batch multiple queries into one malformed tool call.
  A framework would have hidden this behaviour and made it harder to fix.

**How the loop works (the "agentic" part):**
```
1. Send system prompt + user message to the model.
2. Model responds with a tool call, e.g. check_linkedin_presence("iQGateway")
3. We run the corresponding specialized Tavily search, get results back.
4. Append results to the conversation as a "tool" message.
5. Send the updated conversation back to the model.
6. Model reads the results and either:
     a. Calls another tool → go to step 2.
     b. Has enough evidence → writes the final verdict → we stop.
```
The model decides which tools to call, in what order, and how many times.
That autonomy — not a hardcoded "run these 5 queries" pipeline — is what
makes it an agent.

**Mandatory workflow:** the system prompt explicitly requires the agent to
call all three legitimacy tools (`check_linkedin_presence`,
`check_official_website`, `check_company_registry`) at least once each,
*even for an obviously well-known company* — it cannot skip straight to
"this is clearly legitimate" based on name recognition alone. Only after
those three checks does it move on to general `search_web` review/news
digging. This ordering is enforced through hard rules in the prompt, not
left to the model's judgment, since judgment is exactly what we don't want
to rely on for "did you actually check, or did you assume?"

---

###  requests + BeautifulSoup
**What they are:** `requests` fetches raw web pages over HTTP; `BeautifulSoup`
parses HTML so we can pull out readable text.

**Why they're used:** Specifically for `check_official_website`. Tavily is
great for *search*, but to actually load a specific known company website
and check whether it looks real/professional (or broken/empty/shell-like),
we fetch the page directly and strip out scripts, styles, nav, and footer
clutter — leaving just the readable text the model can judge.

---

###  python-dotenv
**What it is:** Loads environment variables from a `.env` file.

**Why:** Keeps API keys out of the source code. You put your keys in `.env`
(which is gitignored), and the app reads them with `os.environ.get(...)`.
Standard practice for any project that uses secrets.

---

## Setup

1. **Get a free Groq API key**
   → [console.groq.com](https://console.groq.com) — sign up with email or Google, no credit card.

2. **Get a free Tavily API key**
   → [tavily.com](https://tavily.com) — 1,000 free searches/month, no credit card.

3. **Create your `.env` file**
   ```bash
   cp .env.example .env
   # then open .env and paste in both keys
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run**
   ```bash
   streamlit run app.py
   ```

---

## Companies to test with

| Company      | Expected verdict | Why                                                   |
|--------------|------------------|-------------------------------------------------------|
| Zoho         | Green Flag       | Profitable, reputable, good intern reviews            |
| Freshworks   | Green Flag       | Listed company, strong employer brand                 |
| BYJU'S       | Red Flag         | Widely reported salary delays, layoffs, governance issues |
| GoMechanic   | Red Flag         | Founders admitted accounting fraud (ET, Entrackr 2023) |
| Trell        | Red Flag / Yellow | Investor fraud FIR filed, co-founder controversy     |

---

## Known limitations

- The model (`openai/gpt-oss-120b`) occasionally generates malformed tool
  calls (tries to batch two queries into one call). This is handled with
  `parallel_tool_calls=False` and an automatic retry with a correction prompt.
  If it fails 3 times in a row, the app surfaces an error message.
- Note: `llama-3.3-70b-versatile` was the original model used in this
  project but was deprecated by Groq on June 17, 2026. The codebase was
  migrated to `openai/gpt-oss-120b` as the recommended replacement.
- Verdicts are only as good as what's publicly available online. A scammy
  company with no online footprint will come back as "no clear signal" —
  which is itself an honest and useful finding.
- `MAX_TURNS = 10` caps the loop so the agent can't run indefinitely.
  In practice it finishes in 4–7 searches.
