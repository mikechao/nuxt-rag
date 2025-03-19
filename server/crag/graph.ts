import type { DocumentInterface } from '@langchain/core/documents'
import type { BaseChatModel } from '@langchain/core/language_models/chat_models'
import type { RunnableConfig } from '@langchain/core/runnables'
import { TavilySearchResults } from '@langchain/community/tools/tavily_search'
import { Document } from '@langchain/core/documents'
import { StringOutputParser } from '@langchain/core/output_parsers'
import { ChatPromptTemplate } from '@langchain/core/prompts'
import { END, START, StateGraph } from '@langchain/langgraph'
import { ChatOpenAI } from '@langchain/openai'
import consola from 'consola'
import { formatDocumentsAsString } from 'langchain/util/document'
import { z } from 'zod'
import { createEmbeddingsAndVectorStore } from '../util/embeddingAndVectorStore'
import { GraphState } from './state'

// https://langchain-ai.github.io/langgraphjs/tutorials/rag/langgraph_crag

async function makeRetriever() {
  const { vectorStore } = await createEmbeddingsAndVectorStore()
  const retriever = vectorStore.asRetriever()
  return retriever
}

async function makeModel(): Promise<BaseChatModel> {
  const model = new ChatOpenAI({
    model: 'gpt-4o-mini',
    temperature: 0,
  })
  return model
}

/**
 * Retrieve documents
 *
 * @param {typeof GraphState.State} state The current state of the graph.
 * @param {RunnableConfig | undefined} _config The configuration object for tracing.
 * @returns {Promise<Partial<typeof GraphState.State>>} The new state object.
 */
async function retrieve(
  state: typeof GraphState.State,
  _config: RunnableConfig | undefined,
): Promise<Partial<typeof GraphState.State>> {
  consola.log('---RETRIEVE---')
  const retriever = await makeRetriever()
  const before = performance.now()
  const documents = await retriever
    .withConfig({ runName: 'FetchRelevantDocuments' })
    .invoke(state.question)
  const after = performance.now()
  consola.info({ tag: 'retrieve', message: `Retrieved ${documents.length} documents in ${after - before}ms` })
  return {
    documents,
  }
}

/**
 * Generate answer
 *
 * @param {typeof GraphState.State} state The current state of the graph.
 * @returns {Promise<Partial<typeof GraphState.State>>} The new state object.
 */
async function generate(
  state: typeof GraphState.State,
): Promise<Partial<typeof GraphState.State>> {
  consola.log('---GENERATE---')
  const model = await makeModel()
  const prompt = ChatPromptTemplate.fromTemplate(`You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:`)
  // Construct the RAG chain by piping the prompt, model, and output parser
  const ragChain = prompt.pipe(model).pipe(new StringOutputParser())

  const before = performance.now()
  const generation = await ragChain.invoke({
    context: formatDocumentsAsString(state.documents),
    question: state.question,
  })
  const after = performance.now()
  consola.info({ tag: 'generate', message: `Generated answer in ${after - before}ms` })

  return {
    generation,
  }
}

/**
 * Determines whether the retrieved documents are relevant to the question.
 *
 * @param {typeof GraphState.State} state The current state of the graph.
 * @returns {Promise<Partial<typeof GraphState.State>>} The new state object.
 */
async function gradeDocuments(
  state: typeof GraphState.State,
): Promise<Partial<typeof GraphState.State>> {
  consola.log('---CHECK RELEVANCE---')
  const model = await makeModel()
  // pass the name & schema to `withStructuredOutput` which will force the model to call this tool.
  const outputSchema = z.object({
    binaryScore: z.enum(['yes', 'no']).describe('Relevance score \'yes\' or \'no\''),
  }).describe('Grade the relevance of the retrieved documents to the question. Either \'yes\' or \'no\'.')
  const llmWithTool = model.withStructuredOutput(
    outputSchema,
    {
      name: 'grade',
    },
  )

  const prompt = ChatPromptTemplate.fromTemplate(
    `You are a grader assessing relevance of a retrieved document to a user question.
  Here is the retrieved document:

  {context}

  Here is the user question: {question}

  If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
  Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.`,
  )

  // Chain
  const chain = prompt.pipe(llmWithTool)
  consola.info({ tag: 'gradeDocuments', message: `Grading ${state.documents.length} documents` })
  const before = performance.now()
  const filteredDocs: Array<DocumentInterface> = []
  for await (const doc of state.documents) {
    const grade = await chain.invoke({
      context: doc.pageContent,
      question: state.question,
    })
    if (grade.binaryScore === 'yes') {
      consola.log('---GRADE: DOCUMENT RELEVANT---')
      filteredDocs.push(doc)
    }
    else {
      consola.log('---GRADE: DOCUMENT NOT RELEVANT---')
    }
  }
  const after = performance.now()
  consola.info({ tag: 'gradeDocuments', message: `Graded ${filteredDocs.length} documents in ${after - before}ms` })
  consola.info({ tag: 'gradeDocuments', message: `Found ${filteredDocs.length} relevant documents` })
  return {
    documents: filteredDocs,
  }
}

/**
 * Transform the query to produce a better question.
 *
 * @param {typeof GraphState.State} state The current state of the graph.
 * @returns {Promise<Partial<typeof GraphState.State>>} The new state object.
 */
async function transformQuery(
  state: typeof GraphState.State,
): Promise<Partial<typeof GraphState.State>> {
  consola.log('---TRANSFORM QUERY---')
  const model = await makeModel()
  // Pull in the prompt
  const prompt = ChatPromptTemplate.fromTemplate(
    `You are generating a question that is well optimized for semantic search retrieval.
  Look at the input and try to reason about the underlying sematic intent / meaning.
  Here is the initial question:
  \n ------- \n
  {question} 
  \n ------- \n
  Formulate an improved question: `,
  )
  consola.info({ tag: 'transformQuery', message: `Transforming query: ${state.question}` })
  // Prompt
  const before = performance.now()
  const chain = prompt.pipe(model).pipe(new StringOutputParser())
  const betterQuestion = await chain.invoke({ question: state.question })
  const after = performance.now()
  consola.info({ tag: 'transformQuery', message: `Transformed query in ${after - before}ms` })
  consola.info({ tag: 'transformQuery', message: `Improved question: ${betterQuestion}` })
  return {
    question: betterQuestion,
  }
}

/**
 * Web search based on the re-phrased question using Tavily API.
 *
 * @param {typeof GraphState.State} state The current state of the graph.
 * @returns {Promise<Partial<typeof GraphState.State>>} The new state object.
 */
async function webSearch(
  state: typeof GraphState.State,
): Promise<Partial<typeof GraphState.State>> {
  consola.log('---WEB SEARCH---')

  const before = performance.now()
  const tool = new TavilySearchResults()
  const docs = await tool.invoke({ input: state.question })
  const webResults = new Document({ pageContent: docs })
  const newDocuments = state.documents.concat(webResults)
  const after = performance.now()
  consola.info({ tag: 'webSearch', message: `Web search completed in ${after - before}ms` })
  consola.info({ tag: 'webSearch', message: `Added 1 document to the graph` })
  return {
    documents: newDocuments,
  }
}

/**
 * Determines whether to generate an answer, or re-generate a question.
 *
 * @param {typeof GraphState.State} state The current state of the graph.
 * @returns {"transformQuery" | "generate"} Next node to call
 */
function decideToGenerate(state: typeof GraphState.State) {
  consola.log('---DECIDE TO GENERATE---')

  const filteredDocs = state.documents
  if (filteredDocs.length === 0) {
    // All documents have been filtered checkRelevance
    // We will re-generate a new query
    consola.log('---DECISION: TRANSFORM QUERY---')
    return 'transformQuery'
  }

  // We have relevant documents, so generate answer
  consola.log('---DECISION: GENERATE---')
  return 'generate'
}

const workflow = new StateGraph(GraphState)
  // Define the nodes
  .addNode('retrieve', retrieve)
  .addNode('gradeDocuments', gradeDocuments)
  .addNode('generate', generate)
  .addNode('transformQuery', transformQuery)
  .addNode('webSearch', webSearch)

workflow.addEdge(START, 'retrieve')
workflow.addEdge('retrieve', 'gradeDocuments')
workflow.addConditionalEdges(
  'gradeDocuments',
  decideToGenerate,
)
workflow.addEdge('transformQuery', 'webSearch')
workflow.addEdge('webSearch', 'generate')
workflow.addEdge('generate', END)

export const graph = workflow.compile().withConfig({ runName: 'CorrectiveRAG' })
