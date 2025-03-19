import consola from 'consola'
import { z } from 'zod'
import { graph } from '~/server/template/retrieval_graph/graph'

export default defineLazyEventHandler(async () => {
  const inputSchema = z.object({
    question: z.string().min(1),
    queryModel: z.enum(['openai/gpt-4o-mini', 'anthropic/claude-3-haiku-20240307']).optional().default('openai/gpt-4o-mini'),
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

    const { question, queryModel } = parsedBody.data
    const inputs = { messages: [{ role: 'user', content: question }] }
    const config = { configurable: { queryModel: queryModel as string } }
    const result = await graph.withConfig(config).invoke(inputs)
    return result.messages[result.messages.length - 1].content
  })
})
