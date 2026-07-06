import os
import streamlit as st
from dotenv import load_dotenv
from agent import CompanyCheckerAgent

load_dotenv()

st.set_page_config(page_title="Company Checker")

st.title("Company Red-Flag Checker")
st.caption("An agent that researches a company before you intern or work there — "
           "not a single search, but a careful, multi-step investigation.")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

if not GROQ_API_KEY or not TAVILY_API_KEY:
    st.error(
        "Missing API keys. Create a `.env` file (see `.env.example`) with:\n\n"
        "```\nGROQ_API_KEY=your_key_here\nTAVILY_API_KEY=your_key_here\n```"
    )
    st.stop()

TOOL_LABELS = {
    "search_web": "Web search",
    "check_linkedin_presence": "LinkedIn check",
    "check_official_website": "Website check",
    "check_company_registry": "Registry check",
}

company_name = st.text_input("Company name", placeholder="e.g. Acme Corp")
go = st.button("Investigate", type="primary", disabled=not company_name)

if go:
    agent = CompanyCheckerAgent(GROQ_API_KEY, TAVILY_API_KEY)

    with st.status(f"Investigating **{company_name}**...", expanded=True) as status:
        final_markdown = None
        for event in agent.investigate(company_name):
            if event[0] == "search":
                _, tool_name, arg_desc = event
                label = TOOL_LABELS.get(tool_name, tool_name)
                st.write(f"{label}: *{arg_desc}*")
            elif event[0] == "done_search":
                _, tool_name, n = event
                st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ found {n} result(s)")
            elif event[0] == "retry":
                st.write("")
            elif event[0] == "final":
                final_markdown = event[1]
                status.update(label="Investigation complete", state="complete")
            elif event[0] == "error":
                st.error(event[1])
                status.update(label="Investigation failed", state="error")

    if final_markdown:
        st.markdown("---")
        st.markdown(final_markdown)
