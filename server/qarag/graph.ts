// https://js.langchain.com/docs/tutorials/qa_chat_history/

import { AIMessage, HumanMessage, SystemMessage, ToolMessage } from '@langchain/core/messages'
import { tool } from '@langchain/core/tools'
import { MessagesAnnotation, StateGraph } from '@langchain/langgraph'
import { ToolNode, toolsCondition } from '@langchain/langgraph/prebuilt'
import { z } from 'zod'
import { makeEmbeddingsAndVectorStore, makeModel, postgresCheckpointer } from '../shared/utils'

const retrieveToolSchema = z.object({
  query: z.string(),
})

const retrieveTool = tool(
  async ({ query }) => {
    const { vectorStore } = await makeEmbeddingsAndVectorStore()
    const retrievedDocs = await vectorStore.similaritySearch(query, 2)
    const serialized = retrievedDocs
      .map(
        doc => `Source: ${doc.metadata.source}\nContent: ${doc.pageContent}`,
      )
      .join('\n')
    return [serialized, retrievedDocs]
  },
  {
    name: 'retrieve',
    description: 'Retrieve information related to a query.',
    schema: retrieveToolSchema,
    responseFormat: 'content_and_artifact',
  },
)

const tools = new ToolNode([retrieveTool])

async function queryOrRespond(state: typeof MessagesAnnotation.State) {
  const llm = await makeModel()
  const llmWithTools = llm.bindTools([retrieveTool])
  const response = await llmWithTools.invoke(state.messages)
  // MessagesState appends messages to state instead of overwriting
  return { messages: [response] }
}

async function generate(state: typeof MessagesAnnotation.State) {
  // Get generated ToolMessages
  const recentToolMessages = []
  for (let i = state.messages.length - 1; i >= 0; i--) {
    const message = state.messages[i]
    if (message instanceof ToolMessage) {
      recentToolMessages.push(message)
    }
    else {
      break
    }
  }
  const toolMessages = recentToolMessages.reverse()

  // Format into prompt
  const docsContent = toolMessages.map(doc => doc.content).join('\n')
  const systemMessageContent
    = 'You are an assistant for question-answering tasks. '
      + 'Use the following pieces of retrieved context to answer '
      + 'the question. If you don\'t know the answer, say that you '
      + 'don\'t know. Use three sentences maximum and keep the '
      + 'answer concise.'
      + '\n\n'
      + `${docsContent}`

  const conversationMessages = state.messages.filter(
    message =>
      message instanceof HumanMessage
      || message instanceof SystemMessage
      || (message instanceof AIMessage && message.tool_calls?.length === 0),
  )
  const prompt = [
    new SystemMessage(systemMessageContent),
    ...conversationMessages,
  ]

  // Run
  const llm = await makeModel()
  const response = await llm.invoke(prompt)
  return { messages: [response] }
}

const graphBuilder = new StateGraph(MessagesAnnotation)
  .addNode('queryOrRespond', queryOrRespond)
  .addNode('tools', tools)
  .addNode('generate', generate)
  .addEdge('__start__', 'queryOrRespond')
  .addConditionalEdges('queryOrRespond', toolsCondition, {
    __end__: '__end__',
    tools: 'tools',
  })
  .addEdge('tools', 'generate')
  .addEdge('generate', '__end__')

// eslint-disable-next-line antfu/no-top-level-await
const checkpointer = await postgresCheckpointer()
export const graph = graphBuilder.compile({ checkpointer }).withConfig({ runName: 'QA Chat History' })
