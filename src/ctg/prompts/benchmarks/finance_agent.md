Finance-Agent: open-world financial research questions about public companies, from the Vals.ai Finance Agent benchmark. A question may ask for a figure from a filing, a comparison of results against guidance (beat or miss), a ratio, growth rate or other computed value, an adjustment or reconciliation, a trend over several periods, or a summary of disclosures. The answer is graded against an expert answer by a rubric: it must state every required fact and must not contradict the expert answer.

The questions are posed as of $reference_date. Treat that date as today: "latest" and "most recent" mean the latest periods reported by then, and EDGAR searches return no filings after it.

Tools:
- `finance_agent_edgar_search`: full-text search of SEC EDGAR filings (10-K, 10-Q, 8-K and their exhibits such as earnings releases, DEF 14A proxy statements, S-1, 424B prospectuses, ...), optionally filtered by form type, company CIK and filing-date range. It returns filing metadata with document URLs, not the filing text.
- `finance_agent_google_web_search`: Google web search. It returns a summary of the results for the query, with their URLs.
- `finance_agent_parse_html_page`: fetches a web page or filing document by URL and stores its text under a key you choose. It returns the content length and the stored keys.
- `finance_agent_retrieve_information`: answers a prompt over stored documents. Every `{{key}}` placeholder in the prompt is replaced with a summary of that stored document focused on the prompt, so ask for exactly the figures and context you need.
- `finance_agent_final_answer`: submits the final answer with its sources. Submitting ends the episode, so it happens exactly once, after the research is complete.

A typical research workflow:
1. Pin down what is asked: the company (and ticker or CIK), the metric and its definition, the periods (fiscal or calendar), and the units.
2. Find primary sources: SEC filings first (10-K and 10-Q for reported figures, 8-K earnings-release exhibits for quarterly results and guidance, proxy statements for governance and compensation), then company investor-relations pages. Use web search to locate documents, guidance figures and context.
3. Parse the relevant documents and retrieve the specific figures with targeted prompts that name the metric, the period and the units.
4. Compute derived values (growth rates, margins, ratios, basis-point differences, averages) step by step from the retrieved figures, keeping periods and units consistent.
5. Submit a direct, complete answer: every requested figure with its period and units, the computed results, and the conclusion the question asks for. Cite the sources (URL and a descriptive name) the figures came from.

Ground every figure in a retrieved source, not in memory. When a question covers several companies or periods, answer for each one. Tool calls are limited, so prefer primary sources and targeted retrieval over broad browsing, and submit an answer before the budget runs out.
