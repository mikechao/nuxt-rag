import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio'
import { OpenAIEmbeddings } from '@langchain/openai'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'
import consola from 'consola'
import { z } from 'zod'
import { pgvectorStore } from '~/server/util/pgvectorStore'

export default defineLazyEventHandler(async () => {
  const inputSchema = z.object({
    urls: z.array(z.string().min(1)).nonempty(),
  })
  const runtimeConfig = useRuntimeConfig()
  const openaiAPIKey = runtimeConfig.openaiAPIKey
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
    const { urls } = parsedBody.data
    consola.info({ tag: 'eventHandler', message: `Received URL: ${urls}` })

    const before = performance.now()
    const docs = await Promise.all(
      urls.map(url => new CheerioWebBaseLoader(url).load()),
    )
    const after = performance.now()
    consola.info({ tag: 'eventHandler', message: `Loaded ${docs.length} documents in ${after - before}ms` })

    const beforeSplit = performance.now()
    const docsList = docs.flat()
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 50,
    })
    const allSplits = await splitter.splitDocuments(docsList)
    const afterSplit = performance.now()
    consola.info({ tag: 'eventHandler', message: `Split ${allSplits.length} documents in ${afterSplit - beforeSplit}ms` })

    const beforeEmbedding = performance.now()
    const embeddings = new OpenAIEmbeddings({
      model: 'text-embedding-3-large',
      apiKey: openaiAPIKey,
    })
    const vectorStore = await pgvectorStore(embeddings)
    await vectorStore.addDocuments(allSplits)
    const afterEmbedding = performance.now()
    consola.info({ tag: 'eventHandler', message: `Added ${allSplits.length} documents to the vector store in ${afterEmbedding - beforeEmbedding}ms` })

    return `Successfully added ${allSplits.length} documents to the vector store. ${vectorStore.collectionName}`
  })
})
