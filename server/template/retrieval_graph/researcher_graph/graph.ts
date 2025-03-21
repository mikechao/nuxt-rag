/**
 * Researcher graph used in the conversational retrieval system as a subgraph.
 * This module defines the core structure and functionality of the researcher graph,
 * which is responsible for generating search queries and retrieving relevant documents.
 */

import type { RunnableConfig } from '@langchain/core/runnables'
import type { QueryStateAnnotation } from './state.js'
import { PromptTemplate } from '@langchain/core/prompts'
import { END, Send, START, StateGraph } from '@langchain/langgraph'
import { z } from 'zod'
import { zodToJsonSchema } from 'zod-to-json-schema'
import { makeRetriever } from '../../shared/retrieval.js'
import { loadChatModel } from '../../shared/utils.js'
import { ensureAgentConfiguration } from '../configuration.js'
import { ResearcherStateAnnotation } from './state.js'

async function generateQueries(
  state: typeof ResearcherStateAnnotation.State,
  config: RunnableConfig,
): Promise<typeof ResearcherStateAnnotation.Update> {
  const Response = z.object({
    queries: z.array(z.string()),
  })
  const configuration = ensureAgentConfiguration(config)
  const model = (
    await loadChatModel(configuration.queryModel, configuration.useCache)
  ).withStructuredOutput(Response)
  const systemContent = await PromptTemplate.fromTemplate(configuration.generateQueriesSystemPrompt)
    .format({
      output_schema: JSON.stringify(zodToJsonSchema(Response), null, 2),
    })
  const messages: { role: string, content: string }[] = [
    { role: 'system', content: systemContent },
    { role: 'human', content: state.question },
  ]
  const response = await model.invoke(messages)
  return { queries: response.queries }
}

async function retrieveDocuments(
  state: typeof QueryStateAnnotation.State,
  config: RunnableConfig,
): Promise<typeof ResearcherStateAnnotation.Update> {
  const retriever = await makeRetriever(config)
  const response = await retriever.invoke(state.query, config)
  return { documents: response }
}

function retrieveInParallel(
  state: typeof ResearcherStateAnnotation.State,
): Send[] {
  return state.queries.map(
    (query: string) => new Send('retrieveDocuments', { query }),
  )
}

// Define the graph
const builder = new StateGraph({
  stateSchema: ResearcherStateAnnotation,
})
  .addNode('generateQueries', generateQueries)
  .addNode('retrieveDocuments', retrieveDocuments)
  .addEdge(START, 'generateQueries')
  .addConditionalEdges('generateQueries', retrieveInParallel, [
    'retrieveDocuments',
  ])
  .addEdge('retrieveDocuments', END)

// Compile into a graph object that you can invoke and deploy.
export const graph = builder
  .compile()
  .withConfig({ runName: 'ResearcherGraph' })
