# Various RAG tutorials from LangChain/LangGraph

This repo contains various tutorials found on LangChain or LangGraph's website in regards to Retrieval Augmented Generation

# Setup

Make sure to install dependencies

```bash
pnpm install
```

## LangGraph Server

Install the local LangGraph Server

```bash
# Or install globally, will be available as `langgraphjs`
npm install -g @langchain/langgraph-cli
```

Launch the LangGraph Server

```bash
$ langgraphjs dev

          Welcome to

╦ ┌─┐┌┐┌┌─┐╔═╗┬─┐┌─┐┌─┐┬ ┬
║ ├─┤││││ ┬║ ╦├┬┘├─┤├─┘├─┤
╩═╝┴ ┴┘└┘└─┘╚═╝┴└─┴ ┴┴ ┴ ┴.js

- 🚀 API: http://localhost:2024
- 🎨 Studio UI: https://smith.langchain.com/studio?baseUrl=http://localhost:2024
```

A browser should be launched to show the LangGraph studio
