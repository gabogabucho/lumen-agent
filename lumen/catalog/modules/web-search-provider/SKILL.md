---
name: web-search-provider
description: "Search the public web and extract text from public URLs"
min_capability: tier-2
requires:
  connectors: [web]
---
# Web Search

You can now search the web and extract content from URLs.

## How to handle search requests

When a user asks to:
- "Search for the latest news about AI"
- "What's the weather in Buenos Aires?"
- "Find information about Python 3.13"
- "Look up the best restaurants in Madrid"

Do the following:
1. Identify the search query from the user's message
2. Use web__search with the query
3. Summarize the results in a clear, concise way
4. If the user wants more detail, use web__extract on a specific URL

## Best practices

- Keep summaries concise — 2-3 paragraphs max
- Cite sources when providing factual information
- If search returns no results, suggest refining the query
- For time-sensitive info (weather, news), note that results may not be real-time

## Tools

`web__search` and `web__extract` are live once this module is installed.
Search returns titles, URLs, and snippets. Extract reads one public page.

The default provider is DuckDuckGo and needs no API key. Set `LUMEN_WEB_PROVIDER`
to `brave`, `tavily`, or `searxng` when the user has configured one:

- Brave: `BRAVE_API_KEY` or `LUMEN_BRAVE_API_KEY`
- Tavily: `TAVILY_API_KEY` or `LUMEN_TAVILY_API_KEY`
- SearXNG: `LUMEN_SEARXNG_URL`, and that base URL must be public

If a call returns `missing_configuration`, tell the user which variable is missing.
Do not invent a key.

## Limitations

Only public http(s) URLs are fetched. Localhost, private ranges, and link-local
addresses are refused, including after a redirect. If extract returns
`blocked_url` or `unverified_host`, say so and do not retry with a rewritten
address. A SearXNG on localhost will not work; the provider URL has to be public.
