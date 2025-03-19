import consola from 'consola'
import { z } from 'zod'
import { graph } from '~/server/template/retrieval_graph/graph'

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
    const inputs = { messages: [{ role: 'user', content: question }] }
    const result = await graph.invoke(inputs)
    return result.messages[result.messages.length - 1].content
  })
})
