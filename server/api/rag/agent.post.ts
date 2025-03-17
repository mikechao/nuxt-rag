import { AIMessage, HumanMessage, ToolMessage, type BaseMessage } from '@langchain/core/messages'
import { ChatPromptTemplate, PromptTemplate } from '@langchain/core/prompts'
import { Annotation, END, START, StateGraph } from '@langchain/langgraph'
import { ToolNode } from '@langchain/langgraph/prebuilt'
import { ChatOpenAI } from '@langchain/openai'
import consola from 'consola'
import { createRetrieverTool } from 'langchain/tools/retriever'
import { z } from 'zod'
import { createEmbeddingsAndVectorStore } from '~/server/util/embeddingAndVectorStore'

// https://langchain-ai.github.io/langgraphjs/tutorials/rag/langgraph_agentic_rag/
export default defineLazyEventHandler(async () => {
  const { vectorStore } = await createEmbeddingsAndVectorStore()
  const retriever = vectorStore.asRetriever()
  const runtimeConfig = useRuntimeConfig()
  const openaiAPIKey = runtimeConfig.openaiAPIKey

  const GraphState = Annotation.Root({
    messages: Annotation<BaseMessage[]>({
      reducer: (x, y) => x.concat(y),
      default: () => [],
    }),
    question: Annotation<string>
  })

  const tool = createRetrieverTool(
    retriever,
    {
      name: 'retrieve_blog_posts',
      description:
        'Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.',
    },
  )
  const tools = [tool]
  const toolNode = new ToolNode<typeof GraphState.State>(tools)

  async function retrieve(state: typeof GraphState.State) {
    consola.log('---RETRIEVE---')
    const docs = await tool.invoke({ query: state.question })
    consola.log('---RETRIEVE SUCCESS---')
    return {
      messages: [new ToolMessage(docs)],
    }
  }

  /**
   * Decides whether the agent should retrieve more information or end the process.
   * This function checks the last message in the state for a function call. If a tool call is
   * present, the process continues to retrieve information. Otherwise, it ends the process.
   * @param {typeof GraphState.State} state - The current state of the agent, including all messages.
   * @returns {string} - A decision to either "continue" the retrieval process or "end" it.
   */
  function shouldRetrieve(state: typeof GraphState.State): string {
    const { messages } = state
    consola.log('---DECIDE TO RETRIEVE---')
    const lastMessage = messages[messages.length - 1]

    if ('tool_calls' in lastMessage && Array.isArray(lastMessage.tool_calls) && lastMessage.tool_calls.length) {
      consola.log('---DECISION: RETRIEVE---')
      return 'retrieve'
    }
    // If there are no tool calls then we finish.
    return END
  }

  /**
   * Determines whether the Agent should continue based on the relevance of retrieved documents.
   * This function checks if the last message in the conversation is of type FunctionMessage, indicating
   * that document retrieval has been performed. It then evaluates the relevance of these documents to the user's
   * initial question using a predefined model and output parser. If the documents are relevant, the conversation
   * is considered complete. Otherwise, the retrieval process is continued.
   * @param {typeof GraphState.State} state - The current state of the agent, including all messages.
   * @returns {Promise<Partial<typeof GraphState.State>>} - The updated state with the new message added to the list of messages.
   */
  async function gradeDocuments(state: typeof GraphState.State): Promise<Partial<typeof GraphState.State>> {
    consola.log('---GET RELEVANCE---')

    const { messages } = state
    const tool = {
      name: 'give_relevance_score',
      description: 'Give a relevance score to the retrieved documents.',
      schema: z.object({
        binaryScore: z.string().describe('Relevance score \'yes\' or \'no\''),
      }),
    }

    const prompt = ChatPromptTemplate.fromTemplate(
      `You are a grader assessing relevance of retrieved docs to a user question.
  Here are the retrieved docs:
  \n ------- \n
  {context} 
  \n ------- \n
  Here is the user question: {question}
  If the content of the docs are relevant to the users question, score them as relevant.
  Give a binary score 'yes' or 'no' score to indicate whether the docs are relevant to the question.
  Yes: The docs are relevant to the question.
  No: The docs are not relevant to the question.`,
    )

    const model = new ChatOpenAI({
      model: 'gpt-4o-mini',
      temperature: 0,
      apiKey: openaiAPIKey,
    }).bindTools([tool], {
      tool_choice: tool.name,
    })

    const chain = prompt.pipe(model)

    const lastMessage = messages[messages.length - 1]

    const score = await chain.invoke({
      question: messages[0].content as string,
      context: lastMessage.content as string,
    })

    return {
      messages: [score],
    }
  }

  /**
   * Check the relevance of the previous LLM tool call.
   *
   * @param {typeof GraphState.State} state - The current state of the agent, including all messages.
   * @returns {string} - A directive to either "yes" or "no" based on the relevance of the documents.
   */
  function checkRelevance(state: typeof GraphState.State): string {
    consola.log('---CHECK RELEVANCE---')

    const { messages } = state
    const lastMessage = messages[messages.length - 1]
    if (!('tool_calls' in lastMessage)) {
      throw new Error('The \'checkRelevance\' node requires the most recent message to contain tool calls.')
    }
    const toolCalls = (lastMessage as AIMessage).tool_calls
    if (!toolCalls || !toolCalls.length) {
      throw new Error('Last message was not a function message')
    }

    if (toolCalls[0].args.binaryScore === 'yes') {
      consola.log('---DECISION: DOCS RELEVANT---')
      return 'yes'
    }
    consola.log('---DECISION: DOCS NOT RELEVANT---')
    return 'no'
  }

  /**
 * Invokes the agent model to generate a response based on the current state.
 * This function calls the agent model to generate a response to the current conversation state.
 * The response is added to the state's messages.
 * @param {typeof GraphState.State} state - The current state of the agent, including all messages.
 * @returns {Promise<Partial<typeof GraphState.State>>} - The updated state with the new message added to the list of messages.
 */
async function agent(state: typeof GraphState.State): Promise<Partial<typeof GraphState.State>> {
  consola.log("---CALL AGENT---");

  const { messages } = state;
  // Find the AIMessage which contains the `give_relevance_score` tool call,
  // and remove it if it exists. This is because the agent does not need to know
  // the relevance score.
  const filteredMessages = messages.filter((message) => {
    if ("tool_calls" in message && Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
      return message.tool_calls[0].name !== "give_relevance_score";
    }
    return true;
  });

  const model = new ChatOpenAI({
    model: "gpt-4o-mini",
    temperature: 0,
    apiKey: openaiAPIKey,
  }).bindTools(tools);

  const response = await model.invoke(filteredMessages);
  return {
    messages: [response],
  };
}

/**
 * Transform the query to produce a better question.
 * @param {typeof GraphState.State} state - The current state of the agent, including all messages.
 * @returns {Promise<Partial<typeof GraphState.State>>} - The updated state with the new message added to the list of messages.
 */
async function rewrite(state: typeof GraphState.State): Promise<Partial<typeof GraphState.State>> {
  consola.log("---TRANSFORM QUERY---");

  const { messages } = state;
  const question = messages[0].content as string;
  const prompt = ChatPromptTemplate.fromTemplate(
    `Look at the input and try to reason about the underlying semantic intent / meaning. \n 
Here is the initial question:
\n ------- \n
{question} 
\n ------- \n
Formulate an improved question:`,
  );

  // Grader
  const model = new ChatOpenAI({
    model: "gpt-4o-mini",
    temperature: 0,
    apiKey: openaiAPIKey
  });
  const response = await prompt.pipe(model).invoke({ question });
  return {
    messages: [response],
    question: response.content as string,
  };
}

/**
 * Generate answer
 * @param {typeof GraphState.State} state - The current state of the agent, including all messages.
 * @returns {Promise<Partial<typeof GraphState.State>>} - The updated state with the new message added to the list of messages.
 */
async function generate(state: typeof GraphState.State): Promise<Partial<typeof GraphState.State>> {
  consola.log("---GENERATE---");

  const { messages } = state;
  const question = messages[0].content as string;
  // Extract the most recent ToolMessage
  const lastToolMessage = messages.slice().reverse().find((msg) => msg.getType() === "tool");
  if (!lastToolMessage) {
    throw new Error("No tool message found in the conversation history");
  }

  const docs = lastToolMessage.content as string;

  const prompt = PromptTemplate.fromTemplate(`You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:`)

  const llm = new ChatOpenAI({
    model: "gpt-4o-mini",
    temperature: 0,
    apiKey: openaiAPIKey,
  });

  const ragChain = prompt.pipe(llm);

  const response = await ragChain.invoke({
    context: docs,
    question,
  });

  return {
    messages: [response],
  };
}

// Define the graph
const workflow = new StateGraph(GraphState)
  // Define the nodes which we'll cycle between.
  .addNode("agent", agent)
  .addNode("retrieve", retrieve)
  .addNode("gradeDocuments", gradeDocuments)
  .addNode("rewrite", rewrite)
  .addNode("generate", generate)

  workflow.addEdge(START, "agent")

  workflow.addConditionalEdges(
    "agent",
    // Assess agent decision
    shouldRetrieve,
  )
  workflow.addEdge("retrieve", "gradeDocuments")
  workflow.addConditionalEdges(
    "gradeDocuments",
    // Assess agent decision
    checkRelevance,
    {
      // Call tool node
      yes: "generate",
      no: "rewrite", // placeholder
    },
  )

  workflow.addEdge("generate", END);
  workflow.addEdge("rewrite", "agent")

  const graph = workflow.compile();

  const inputSchema = z.object({
    question: z.string().min(1),
  })
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

    const { question } = parsedBody.data
    consola.info({ tag: 'eventHandler', message: `Received question: ${question}` })
    const inputs = {
      messages: [new HumanMessage(question)],
      question
    }
    const response = await graph.invoke(inputs)
    consola.info({ tag: 'eventHandler', message: `Result: ${JSON.stringify(response.messages[response.messages.length - 1])}` })
    return response.messages[response.messages.length - 1].content
  })
})
