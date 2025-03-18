import type { Embeddings } from '@langchain/core/embeddings'
import type { RunnableConfig } from '@langchain/core/runnables'
import type { VectorStoreRetriever } from '@langchain/core/vectorstores'
import process from 'node:process'
import { PGVectorStore } from '@langchain/community/vectorstores/pgvector'
import { OpenAIEmbeddings } from '@langchain/openai'
import pg from 'pg'
import { ensureBaseConfiguration } from './configuration.js'
// https://github.com/langchain-ai/rag-research-agent-template-js/blob/main/src/shared/retrieval.ts

async function makePgVectorRetriever(configuration: ReturnType<typeof ensureBaseConfiguration>, embeddingModel: Embeddings): Promise<VectorStoreRetriever> {
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

  const vectorStore = await PGVectorStore.initialize(embeddingModel, {
    pool,
    tableName: 'rag_vectors',
  })
  return vectorStore.asRetriever({ filter: configuration.searchKwargs || {} })
}

function makeTextEmbeddings(modelName: string): Embeddings {
  /**
   * Connect to the configured text encoder.
   */
  const index = modelName.indexOf('/')
  let provider, model
  if (index === -1) {
    model = modelName
    provider = 'openai' // Assume openai if no provider included
  }
  else {
    provider = modelName.slice(0, index)
    model = modelName.slice(index + 1)
  }
  // can't useRuntimeConfig here not inside a defineEventHandler
  // const runtimeConfig = useRuntimeConfig()
  const openaiAPIKey = process.env.NUXT_OPENAI_API_KEY
  switch (provider) {
    case 'openai':
      return new OpenAIEmbeddings({ model, apiKey: openaiAPIKey })
    default:
      throw new Error(`Unsupported embedding provider: ${provider}`)
  }
}

export async function makeRetriever(
  config: RunnableConfig,
): Promise<VectorStoreRetriever> {
  const configuration = ensureBaseConfiguration(config)
  const embeddingModel = makeTextEmbeddings(configuration.embeddingModel)
  switch (configuration.retrieverProvider) {
    case 'pgvector':
      return await makePgVectorRetriever(configuration, embeddingModel)
    default:
      throw new Error(
        `Unrecognized retrieverProvider in configuration: ${configuration.retrieverProvider}`,
      )
  }
}
