import { OpenAIEmbeddings } from '@langchain/openai'
import { pgvectorStore } from './pgvectorStore'

/**
 * Creates and returns both OpenAI embeddings and a PG Vector Store
 * @param options Optional configuration options
 * @param options.apiKey OpenAI API key for authentication
 * @param options.model The name of the embedding model to use
 * @param options.dimensions The dimension size of the embeddings
 * @param options.collectionName The name of the vector store collection
 * @returns An object containing both the embeddings instance and vector store
 */
export async function createEmbeddingsAndVectorStore(options?: {
  apiKey?: string
  model?: string
  dimensions?: number
  collectionName?: string
}) {
  const runtimeConfig = useRuntimeConfig()
  const openaiAPIKey = runtimeConfig.openaiAPIKey
  // Initialize embeddings with provided API key or from runtime config
  const embeddings = new OpenAIEmbeddings({
    model: options?.model || 'text-embedding-3-large',
    apiKey: options?.apiKey || openaiAPIKey,
    dimensions: options?.dimensions || 1536,
  })

  const vectorStore = await pgvectorStore(embeddings)
  return {
    embeddings,
    vectorStore,
  }
}
