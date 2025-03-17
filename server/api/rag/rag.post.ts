import type { Document } from '@langchain/core/documents'
import { PGVectorStore } from '@langchain/community/vectorstores/pgvector'
import { AIMessage, HumanMessage, SystemMessage, ToolMessage } from '@langchain/core/messages'
import { PromptTemplate } from '@langchain/core/prompts'
import { tool } from '@langchain/core/tools'
import { Annotation, MessagesAnnotation, StateGraph } from '@langchain/langgraph'
import { ToolNode, toolsCondition } from '@langchain/langgraph/prebuilt'
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai'
import consola from 'consola'
import pg from 'pg'
import { z } from 'zod'
import { postgresCheckpointer } from '~/server/util/postgresCheckpointer'

export default defineLazyEventHandler(async () => {
  const inputSchema = z.object({
    question: z.string().min(1),
  })
  const runtimeConfig = useRuntimeConfig()
  const openaiAPIKey = runtimeConfig.openaiAPIKey
  const { Pool } = pg
  const pool = new Pool({
    host: 'localhost',
    user: 'dbuser',
    password: 'dbpassword',
    database: 'nuxtragdb',
    port: 5432,
    max: 20,
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 2000,
  })

  const embeddings = new OpenAIEmbeddings({
    model: 'text-embedding-3-large',
    apiKey: openaiAPIKey,
  })

  const vectorStore = await PGVectorStore.initialize(embeddings, {
    pool,
    tableName: 'rag_vectors',
    dimensions: 3072,
  })

  const llm = new ChatOpenAI({
    model: 'gpt-4o-mini',
    temperature: 0,
    apiKey: openaiAPIKey,
  })

  const retrieveToolSchema = z.object({
    query: z.string(),
  })

  const retrieveTool = tool(
    async ({ query }) => {
      const retrievedDocs = await vectorStore.similaritySearch(query, 2)
      const serialized = retrievedDocs
        .map(
          doc => `Source: ${doc.metadata.source}\nContent: ${doc.pageContent}`,
        )
        .join('\n')
      return [serialized, retrievedDocs]
    },
    {
      name: 'retrieve',
      description: 'Retrieve information related to a query.',
      schema: retrieveToolSchema,
      responseFormat: 'content_and_artifact',
    },
  )

  async function queryOrRespond(state: typeof MessagesAnnotation.State) {
    const llmWithTools = llm.bindTools([retrieveTool])
    const response = await llmWithTools.invoke(state.messages)
    // MessagesState appends messages to state instead of overwriting
    return { messages: [response] }
  }

  const tools = new ToolNode([retrieveTool])

  async function generate(state: typeof MessagesAnnotation.State) {
    // Get generated ToolMessages
    const recentToolMessages = []
    for (let i = state.messages.length - 1; i >= 0; i--) {
      const message = state.messages[i]
      if (message instanceof ToolMessage) {
        recentToolMessages.push(message)
      }
      else {
        break
      }
    }
    const toolMessages = recentToolMessages.reverse()

    // Format into prompt
    const docsContent = toolMessages.map(doc => doc.content).join('\n')
    const systemMessageContent
      = 'You are an assistant for question-answering tasks. '
        + 'Use the following pieces of retrieved context to answer '
        + 'the question. If you don\'t know the answer, say that you '
        + 'don\'t know. Use three sentences maximum and keep the '
        + 'answer concise.'
        + '\n\n'
        + `${docsContent}`

    const conversationMessages = state.messages.filter(
      message =>
        message instanceof HumanMessage
        || message instanceof SystemMessage
        || (message instanceof AIMessage && message.tool_calls?.length === 0),
    )
    const prompt = [
      new SystemMessage(systemMessageContent),
      ...conversationMessages,
    ]

    // Run
    const response = await llm.invoke(prompt)
    return { messages: [response] }
  }

  const graphBuilder = new StateGraph(MessagesAnnotation)
    .addNode('queryOrRespond', queryOrRespond)
    .addNode('tools', tools)
    .addNode('generate', generate)
    .addEdge('__start__', 'queryOrRespond')
    .addConditionalEdges('queryOrRespond', toolsCondition, {
      __end__: '__end__',
      tools: 'tools',
    })
    .addEdge('tools', 'generate')
    .addEdge('generate', '__end__')

  const checkpointer = await postgresCheckpointer()
  const graph = graphBuilder.compile({ checkpointer })

  return defineEventHandler(async (event) => {
    const body = await readBody(event)
    const parsedBody = inputSchema.safeParse(body)
    if (!parsedBody.success) {
      const formattedError = parsedBody.error.flatten()
      consola.error({ tag: 'eventHandler', message: `Invalid input: ${JSON.stringify(formattedError)}` })
      throw createError({
        statusCode: 400,
        statusMessage: 'Bad Request',
        message: JSON.stringify(formattedError) || 'Invalid input',
      })
    }

    const { question } = parsedBody.data
    consola.info({ tag: 'eventHandler', message: `Received question: ${question}` })
    const threadConfig = { configurable: { thread_id: 'abc123' } }
    const input = { messages: [{ role: 'user', content: question }] }
    const result = await graph.invoke(input, threadConfig)
    consola.info({ tag: 'eventHandler', message: `Result: ${JSON.stringify(result)}` })
    return result.messages[result.messages.length - 1].content
  })
})
