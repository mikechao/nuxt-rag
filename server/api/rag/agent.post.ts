import { tool } from '@langchain/core/tools'
import { createReactAgent } from '@langchain/langgraph/prebuilt'
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai'
import consola from 'consola'
import { z } from 'zod'
import { pgvectorStore } from '~/server/util/pgvectorStore'
import { postgresCheckpointer } from '~/server/util/postgresCheckpointer'

export default defineLazyEventHandler(async () => {
  const runtimeConfig = useRuntimeConfig()
  const openaiAPIKey = runtimeConfig.openaiAPIKey
  const embeddings = new OpenAIEmbeddings({
    model: 'text-embedding-3-large',
    apiKey: openaiAPIKey,
  })

  const vectorStore = await pgvectorStore(embeddings)

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
  const checkpointer = await postgresCheckpointer()
  const agent = createReactAgent({ llm, tools: [retrieveTool], checkpointer })

  const inputSchema = z.object({
    question: z.string().min(1),
  })
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
    const response = await agent.invoke(input, threadConfig)
    return response.messages[response.messages.length - 1].content
  })
})
