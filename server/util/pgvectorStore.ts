import type { OpenAIEmbeddings } from '@langchain/openai'
import { PGVectorStore } from '@langchain/community/vectorstores/pgvector'
import consola from 'consola'
import pg from 'pg'

export async function pgvectorStore(embeddings: OpenAIEmbeddings) {
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

  try {
    const vectorStore = await PGVectorStore.initialize(embeddings, {
      pool,
      tableName: 'rag_vectors',
      dimensions: 3072,
    })
    return vectorStore
  }
  catch (error: any) {
    consola.error('Error setting up PGVectorStore:', error)
    if (error.message && error.message.includes('ECONNREFUSED')) {
      console.error(
        'Please make sure your Postgres server is running and that the URL is correct.',
      )
      throw createError({
        statusCode: 503,
        message: 'Unable to connect to Postgres. Please try again later.',
      })
    }
    throw createError({
      statusCode: 500,
      message: 'Error setting up PostgresSaver.',
    })
  }
}
