import type { EvaluationResult } from 'langsmith/evaluation'
import type { ExampleCreate } from 'langsmith/schemas'
import { BrowserbaseLoader } from '@langchain/community/document_loaders/web/browserbase'
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'
import consola from 'consola'
import { MemoryVectorStore } from 'langchain/vectorstores/memory'
import { Client } from 'langsmith'
import { evaluate } from 'langsmith/evaluation'
import { traceable } from 'langsmith/traceable'
import { z } from 'zod'
import { Correctness } from '~/server/rag/eval/correctness'
import { Groundedness } from '~/server/rag/eval/groundedness'
import { Relevance } from '~/server/rag/eval/relevance'
import { RetrievalRelevance } from '~/server/rag/eval/retrievalRelevance'

export default defineLazyEventHandler(async () => {
  const runtimeConfig = useRuntimeConfig()
  const openaiAPIKey = runtimeConfig.openaiAPIKey

  // List of URLs to load documents from
  const urls = [
    'https://lilianweng.github.io/posts/2023-06-23-agent/',
    'https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/',
    'https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/',
  ]

  // BrowserbaseLoader only works with @browserbasehq/sdk 1.1.5
  const loader = new BrowserbaseLoader(urls, {
    apiKey: runtimeConfig.browserbaseAPIKey,
    textContent: true,
  })
  const docs = await loader.load()

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  })

  const allSplits = await splitter.splitDocuments(docs)

  const embeddings = new OpenAIEmbeddings({
    model: 'text-embedding-3-large',
    apiKey: openaiAPIKey,
  })

  const vectorStore = new MemoryVectorStore(embeddings)

  // Index chunks
  await vectorStore.addDocuments(allSplits)

  const llm = new ChatOpenAI({
    model: 'gpt-4o-mini',
    temperature: 1,
    apiKey: openaiAPIKey,
  })

  const ragBot = traceable(
    async (question: string) => {
      // LangChain retriever will be automatically traced
      const retrievedDocs = await vectorStore.similaritySearch(question)
      const docsContent = retrievedDocs.map(doc => doc.pageContent).join('')

      const instructions = `You are a helpful assistant who is good at analyzing source information and answering questions.
        Use the following source documents to answer the user's questions.
        If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise.

        Documents:
        ${docsContent}`

      const aiMsg = await llm.invoke([
        {
          role: 'system',
          content: instructions,
        },
        {
          role: 'user',
          content: question,
        },
      ])

      return { answer: aiMsg.content, documents: retrievedDocs }
    },
  )

  // Define the examples for the dataset
  const examples = [
    [
      'How does the ReAct agent use self-reflection? ',
      'ReAct integrates reasoning and acting, performing actions - such tools like Wikipedia search API - and then observing / reasoning about the tool outputs.',
    ],
    [
      'What are the types of biases that can arise with few-shot prompting?',
      'The biases that can arise with few-shot prompting include (1) Majority label bias, (2) Recency bias, and (3) Common token bias.',
    ],
    [
      'What are five types of adversarial attacks?',
      'Five types of adversarial attacks are (1) Token manipulation, (2) Gradient based attack, (3) Jailbreak prompting, (4) Human red-teaming, (5) Model red-teaming.',
    ],
  ]

  const client = new Client({
    apiKey: runtimeConfig.langsmithAPIKey,
  })

  const datasetName = 'Lilian Weng Blogs Q&A'
  const dataset = await client.createDataset(datasetName)
  const examplesArray: ExampleCreate[] = []
  examples.forEach(([input, output]) => {
    // dataset_id or dataset_name, but not both
    examplesArray.push({
      inputs: { input },
      outputs: { outputs: output },
      dataset_id: dataset.id,
    })
  })
  await client.createExamples(examplesArray)

  const correctness = new Correctness(openaiAPIKey)
  const groundedness = new Groundedness(openaiAPIKey)
  const retrievalRelevance = new RetrievalRelevance(openaiAPIKey)
  const relevance = new Relevance(openaiAPIKey)

  const targetFunc = (inputs: Record<string, any>) => {
    return ragBot(inputs.input)
  }

  return defineEventHandler(async (event) => {
    const experimentResults = await evaluate(targetFunc, {
      data: datasetName,
      evaluators: [correctness.correctness, groundedness.grounded, relevance.relevance, retrievalRelevance.retrievalRelevance],
      experimentPrefix: 'rag-doc-relevance',
      metadata: { version: 'LCEL context, gpt-4-0125-preview' },
      client,
    })
    return experimentResults.summaryResults
  })
})
