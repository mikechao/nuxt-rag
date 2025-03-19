import consola from 'consola'
import { z } from 'zod'
import { graph } from '~/server/qarag/graph'

// https://js.langchain.com/docs/tutorials/qa_chat_history/
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
    const threadConfig = { configurable: { thread_id: 'abc123' } }
    const input = { messages: [{ role: 'user', content: question }] }
    const result = await graph.invoke(input, threadConfig)
    consola.info({ tag: 'eventHandler', message: `Result: ${JSON.stringify(result.messages[result.messages.length - 1])}` })
    return result.messages[result.messages.length - 1].content
  })
})
