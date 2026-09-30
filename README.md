# Internship / Company Red-Flag Checker

> An agentic AI tool that researches a company before you intern or work there.


You type a company name. Instead of a single search and summary, the agent behaves like a careful researcher — running multiple targeted checks, deciding what to look into next, weighing whether findings are real patterns or just noise, and producing a structured **Green / Yellow / Red Flag** verdict with sources.

No LangChain. No LangGraph. Just a clean, hand-written agentic loop you can explain line by line.

---

## Demo output

```
## Verdict: Red Flag — Multiple credible reports of delayed stipends and no mentorship

### Legitimacy check
- LinkedIn: Active page, ~120 employees, last post 3 months ago
- Official website: Loads fine, but contact page has no real address
- Registry: No clear MCA registration found under this exact name

### What looks good
- Active social media presence and a professional-looking website

### What looks concerning
- 3 independent Reddit/Quora threads about unpaid stipends (2023–2024)
- No structured learning or mentorship reported by any former interns
- AmbitionBox rating: 2.1/5 across 40+ reviews
- No verifiable company registration found

### Reasoning
The stipend delay pattern appears across multiple independent sources over
two years, making it a real pattern rather than isolated noise. The unclear
registry status adds to the concern.

### Sources
- [Company Reviews — AmbitionBox](https://ambitionbox.com/...)
- [Reddit thread: internship experience](https://reddit.com/...)
```

---

## How it works

The agent runs a **mandatory 4-tool investigation** on every company — it cannot skip any check, even for well-known names.

| Tool | What it checks |
|------|---------------|
| `check_linkedin_presence(company_name)` | Real LinkedIn page, employee count, hiring activity, recent posts |
| `check_official_website(company_name)` | Whether the company has a real, active, professional website |
| `check_company_registry(company_name, country)` | Legal traceability — MCA records (India) or OpenCorporates-style data |
| `search_web(query)` | Reviews, stipend complaints, layoffs, scams, lawsuits, news |

After the three legitimacy checks, the agent runs several `search_web` queries with different angles (e.g. `"<company> glassdoor"`, `"<company> stipend delay reddit"`, `"<company> scam complaints"`). It decides what to search for next based on what it found — and only stops when it has enough evidence to write a verdict.

### The agentic loop (simplified)

```
1. Send system prompt + company name to the LLM
2. LLM calls a tool → e.g. check_linkedin_presence("Acme Corp")
3. We run the search via Tavily, get results
4. Append results to the conversation
5. Send back to LLM → it either calls another tool or writes the verdict
6. Repeat until verdict
```

The model decides **which tools to call**, **in what order**, and **when it has enough evidence**. That's what makes this an agent rather than a fixed pipeline.

---

## Tech stack

| Tool | Role | Why |
|------|------|-----|
| **Groq** (`openai/gpt-oss-120b`) | LLM / agent brain | Free tier, no credit card, fast, supports tool calling |
| **Tavily** | Web search | Returns clean text snippets — no HTML scraping needed |
| **Streamlit** | Web UI | Single-file UI with live agent progress via `st.status()` |
| **python-dotenv** | Secret management | Keeps API keys out of source code |

> **Note:** The three "legitimacy" tools (`check_linkedin_presence`, `check_official_website`, `check_company_registry`) are all implemented as **specialized Tavily searches** with narrowly targeted queries — no separate LinkedIn, MCA, or OpenCorporates API required. If you later get access to those APIs, only the internals of each `_check_*` method in `agent.py` need to change.

---

## Project structure

```
company-checker/
├── agent.py          # The agent: LLM + 4 tools + agentic loop
├── app.py            # Streamlit UI (~35 lines)
├── requirements.txt
├── .env.example      # API key template
└── README.md
```

---

## Setup

### 1. Get your API keys (both free, no credit card)

- **Groq:** [console.groq.com](https://console.groq.com) — sign up with email or Google
- **Tavily:** [app.tavily.com](https://app.tavily.com) — 1,000 free searches/month

### 2. Clone and configure

```bash
git clone https://github.com/your-username/company-checker.git
cd company-checker

cp .env.example .env
# Open .env and paste in your keys:
# GROQ_API_KEY=your_key_here
# TAVILY_API_KEY=your_key_here
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

---

## Companies to test with

| Company | Expected verdict | Why |
|---------|-----------------|-----|
| Zoho | Green Flag | Profitable, reputable, strong intern reviews |
| Freshworks | Green Flag | Listed company, strong employer brand |
| BYJU'S | Red Flag | Salary delays, mass layoffs, governance issues (widely reported) |
| GoMechanic | Red Flag | Founders admitted accounting fraud (Economic Times, 2023) |
| Trell | Yellow / Red | Investor fraud FIR filed, co-founder controversy |

---

## Key design decisions

**Why no LangChain/LangGraph?**
The agentic loop is just ~60 lines of Python — a while-loop with an if-statement. Adding a framework would introduce hundreds of lines of abstraction with no benefit, and make it harder to explain or debug. Every line here is readable and intentional.

**Why force all 4 tools every time?**
Legitimacy and trustworthiness are separate questions. A company can be 100% registered with a polished LinkedIn page and still have a pattern of unpaid stipends. The system prompt enforces the full investigation with hard rules — not model judgment.

**Why Tavily over Google/Bing?**
Tavily returns clean, structured text snippets ready for the LLM — no HTML parsing, no scraping, no extra processing. One API call, readable results.

---

## Known limitations

- LLMs occasionally generate malformed tool calls (batching multiple queries into one). Handled with `parallel_tool_calls=False` and an automatic retry with a correction prompt (up to 3 retries before surfacing an error).
- Verdicts depend entirely on what's publicly available online. A company with no online footprint will come back as "no clear signal" — which is itself a useful, honest finding.
- `MAX_TURNS = 14` caps the loop to prevent infinite runs. In practice the agent finishes in 6–10 tool calls.

---

## Skills demonstrated

`Python` · `Agentic AI` · `LLM Tool Calling` · `Prompt Engineering` · `Groq API` · `Tavily Search API` · `Streamlit` · `API Integration` · `Error Handling`

---
