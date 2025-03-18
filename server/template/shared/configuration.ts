/**
 * Define the configurable parameters for the agent.
 */

import type { RunnableConfig } from '@langchain/core/runnables'
import { Annotation } from '@langchain/langgraph'

/**
 * typeof ConfigurationAnnotation.State class for indexing and retrieval operations.
 *
 * This annotation defines the parameters needed for configuring the indexing and
 * retrieval processes, including user identification, embedding model selection,
 * retriever provider choice, and search parameters.
 */
export const BaseConfigurationAnnotation = Annotation.Root({
  /**
   * Name of the embedding model to use. Must be a valid embedding model name.
   */
  embeddingModel: Annotation<string>,

  /**
   * The vector store provider to use for retrieval.
   * Options are 'elastic', 'elastic-local', 'pinecone', or 'mongodb'.
   */
  retrieverProvider: Annotation<
    'elastic' | 'elastic-local' | 'pinecone' | 'mongodb' | 'pgvector'
  >,

  /**
   * Additional keyword arguments to pass to the search function of the retriever.
   */
  searchKwargs: Annotation<Record<string, any>>,
})

/**
 * Create an typeof BaseConfigurationAnnotation.State instance from a RunnableConfig object.
 *
 * @param config - The configuration object to use.
 * @returns An instance of typeof BaseConfigurationAnnotation.State with the specified configuration.
 */
export function ensureBaseConfiguration(
  config: RunnableConfig,
): typeof BaseConfigurationAnnotation.State {
  const configurable = (config?.configurable || {}) as Partial<
    typeof BaseConfigurationAnnotation.State
  >
  return {
    embeddingModel:
      configurable.embeddingModel || 'openai/text-embedding-3-small',
    retrieverProvider: configurable.retrieverProvider || 'pgvector',
    searchKwargs: configurable.searchKwargs || {},
  }
}
