/**
 *
    Retrieval relevance: Retrieved docs vs input
 
    Goal: Measure "how relevant are my retrieved results for this query"
    Mode: Does not require reference answer, because it will compare the question to the retrieved context
    Evaluator: Use LLM-as-judge to assess relevance
 
 */

import { ChatOpenAI } from '@langchain/openai'
import { z } from 'zod'

export class RetrievalRelevance {
  private retrievalRelevanceLLM: ChatOpenAI
  private retrievalRelevanceOutput = z
    .object({
      explanation: z
        .string()
        .describe('Explain your reasoning for the score'),
      relevant: z
        .boolean()
        .describe('True if the retrieved documents are relevant to the question, False otherwise'),
    })
    .describe('Retrieval relevance score for the retrieved documents v.s. the question.')

  private retrievalRelevanceInstructions = `You are a teacher grading a quiz. 

  You will be given a QUESTION and a set of FACTS provided by the student. 
  
  Here is the grade criteria to follow:
  (1) You goal is to identify FACTS that are completely unrelated to the QUESTION
  (2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
  (3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met
  
  Relevance:
  A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
  A relevance value of False means that the FACTS are completely unrelated to the QUESTION.
  
  Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 
  
  Avoid simply stating the correct answer at the outset.`

  constructor(openApikey: string) {
    this.retrievalRelevanceLLM = new ChatOpenAI({
      model: 'gpt-4o-mini',
      temperature: 0,
      apiKey: openApikey,
    })
  }

  async retrievalRelevance({ inputs, outputs }: { inputs: Record<string, any>, outputs: Record<string, any> }) {
    const docString = outputs.documents.map(doc => doc.pageContent).join('')
    const answer = `FACTS: ${docString} QUESTION: ${inputs.question}`

    const structuredLLM = this.retrievalRelevanceLLM.withStructuredOutput(this.retrievalRelevanceOutput)
    const messages = [{ role: 'system', content: this.retrievalRelevanceInstructions }, { role: 'user', content: answer }]
    const grade = await structuredLLM.invoke(messages)
    return grade.relevant
  }
}
