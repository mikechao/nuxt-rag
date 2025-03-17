import { PostgresSaver } from '@langchain/langgraph-checkpoint-postgres'
import consola from 'consola'

export async function postgresCheckpointer() {
  const runtimeConfig = useRuntimeConfig()

  try {
    const checkpointer = PostgresSaver.fromConnString(
      runtimeConfig.postgresURL,
    )
    await checkpointer.setup()
    return checkpointer
  }
  catch (error: any) {
    consola.error('Error setting up PostgresSaver:', error)
    if (error.message && error.message.includes('ECONNREFUSED')) {
      consola.error(
        'Please make sure your Postgres server is running and that the URL is correct.',
      )
      throw createError({
        statusCode: 503,
        message: 'Unable to connect to Postgres. Please try again later.',
      })
    }
    throw createError({
      statusCode: 500,
      message: 'Error setting up PostgresSaver.',
    })
  }
}
