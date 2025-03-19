import { PGVectorStore } from "@langchain/community/vectorstores/pgvector"
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres"
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai"
import consola from "consola"
import pg from 'pg'

export async function makeEmbeddings() {
  const embeddings = new OpenAIEmbeddings({
    model: 'text-embedding-3-large',
    dimensions: 3072,
  })
  return embeddings
}

export async function makeVectorStore(embeddings: OpenAIEmbeddings) {
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
  return vectorStore
}

export async function makeEmbeddingsAndVectorStore() {
  const embeddings = await makeEmbeddings()
  const vectorStore = await makeVectorStore(embeddings)
  return { embeddings, vectorStore }
}

export async function makeModel() {
  const model = new ChatOpenAI({
    model: 'gpt-4o-mini',
    temperature: 0,
  })
  return model
}

export async function postgresCheckpointer() {
    try {
      const postgresURL = process.env.POSTGRES_URL
      if (!postgresURL) {
        throw new Error('POSTGRES_URL environment variable not set')
      }
      const checkpointer = PostgresSaver.fromConnString(
        postgresURL,
      )
      await checkpointer.setup()
      return checkpointer
    }
    catch (error: any) {
      consola.error('Error setting up PostgresSaver:', error)
      if (error.message && error.message.includes('ECONNREFUSED')) {
        consola.error(
          'Please make sure your Postgres server is running and that the URL is correct.',
        )
        throw new Error('Unable to connect to Postgres. Please try again later.')
      }
      throw new Error('Error setting up PostgresSaver.')
    }
}