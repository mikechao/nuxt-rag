import type { Document } from '@langchain/core/documents'
import { PGVectorStore } from '@langchain/community/vectorstores/pgvector'
import { PromptTemplate } from '@langchain/core/prompts'
import { Annotation, StateGraph } from '@langchain/langgraph'
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai'
import consola from 'consola'
import pg from 'pg'

import { z } from 'zod'

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

  const promptTemplate = PromptTemplate.fromTemplate(`You are an assistant for question-answering tasks. 
  Use the following pieces of retrieved context to answer the question. 
  If you don't know the answer, just say that you don't know. 
  Use three sentences maximum and keep the answer concise.
  Question: {question} 
  Context: {context} `)

  // Define state for application
  const InputStateAnnotation = Annotation.Root({
    question: Annotation<string>,
  })

  const StateAnnotation = Annotation.Root({
    question: Annotation<string>,
    context: Annotation<Document[]>,
    answer: Annotation<string>,
  })

  // Define application steps
  const retrieve = async (state: typeof InputStateAnnotation.State) => {
    const retrievedDocs = await vectorStore.similaritySearch(state.question)
    return { context: retrievedDocs }
  }

  const generate = async (state: typeof StateAnnotation.State) => {
    const docsContent = state.context.map(doc => doc.pageContent).join('\n')
    const messages = await promptTemplate.invoke({ question: state.question, context: docsContent })
    const response = await llm.invoke(messages)
    return { answer: response.content }
  }

  const graph = new StateGraph(StateAnnotation)
    .addNode('retrieve', retrieve, { input: InputStateAnnotation })
    .addNode('generate', generate)
    .addEdge('__start__', 'retrieve')
    .addEdge('retrieve', 'generate')
    .addEdge('generate', '__end__')
    .compile()

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
    const inputs = { question }
    const result = await graph.invoke(inputs)
    consola.info({ tag: 'eventHandler', message: `Result: ${JSON.stringify(result)}` })
    return result.answer
  })
})
