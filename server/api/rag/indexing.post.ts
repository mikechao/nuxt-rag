import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio'
import { PGVectorStore } from '@langchain/community/vectorstores/pgvector'
import { OpenAIEmbeddings } from '@langchain/openai'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'
import consola from 'consola'
import pg from 'pg'
import { z } from 'zod'

export default defineLazyEventHandler(async () => {
  const inputSchema = z.object({
    url: z.string().min(1),
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
    const { url } = parsedBody.data
    const runtimeConfig = useRuntimeConfig()
    const openaiAPIKey = runtimeConfig.openaiAPIKey
    consola.info({ tag: 'eventHandler', message: `Received URL: ${url}` })

    const pTagSelector = 'p'
    const cheerioLoader = new CheerioWebBaseLoader(
      url,
      {
        selector: pTagSelector,
      },
    )
    const docs = await cheerioLoader.load()

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    })
    const allSplits = await splitter.splitDocuments(docs)

    const embeddings = new OpenAIEmbeddings({
      model: 'text-embedding-3-large',
      apiKey: openaiAPIKey,
    })

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

    const vectorStore = await PGVectorStore.initialize(embeddings, {
      pool,
      tableName: 'rag_vectors',
      dimensions: 3072,
    })

    await vectorStore.addDocuments(allSplits)
    return `Successfully added ${allSplits.length} documents to the vector store. ${vectorStore.collectionName}`
  })
})
