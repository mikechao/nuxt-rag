import consola from 'consola'
import { z } from 'zod'
import { graph } from '~/server/crag/graph'

// https://langchain-ai.github.io/langgraphjs/tutorials/rag/langgraph_crag
export default defineLazyEventHandler(async () => {
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
    const inputs = { question }
    const config = { recursionLimit: 50 }
    const result = await graph.invoke(inputs, config)
    consola.info({ tag: 'eventHandler', message: `Result: ${JSON.stringify(result)}` })
    return result
  })
})
