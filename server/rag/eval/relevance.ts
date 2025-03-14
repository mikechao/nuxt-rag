/**
 *
    Relevance: Response vs input
 
    Goal: Measure "how well does the generated response address the initial user input"
    Mode: Does not require reference answer, because it will compare the answer to the input question
    Evaluator: Use LLM-as-judge to assess answer relevance, helpfulness, etc.
 
 *
 */

import { ChatOpenAI } from '@langchain/openai'
import { z } from 'zod'

export class Relevance {
  private relevanceLLM: ChatOpenAI
  private relevanceOutput = z
    .object({
      explanation: z
        .string()
        .describe('Explain your reasoning for the score'),
      relevant: z
        .boolean()
        .describe('Provide the score on whether the answer addresses the question'),
    })
    .describe('Relevance score for gene')

  private relevanceInstructions = `You are a teacher grading a quiz. 

  You will be given a QUESTION and a STUDENT ANSWER. 
  
  Here is the grade criteria to follow:
  (1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
  (2) Ensure the STUDENT ANSWER helps to answer the QUESTION
  
  Relevance:
  A relevance value of True means that the student's answer meets all of the criteria.
  A relevance value of False means that the student's answer does not meet all of the criteria.
  
  Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 
  
  Avoid simply stating the correct answer at the outset.`

  constructor(openAIKey: string) {
    this.relevanceLLM = new ChatOpenAI({
      model: 'gpt-4o-mini',
      temperature: 0,
      apiKey: openAIKey,
    })
  }

  async relevance({ inputs, outputs }: { inputs: Record<string, any>, outputs: Record<string, any> }) {
    const answer = `QUESTION: ${inputs.input}
    STUDENT ANSWER: ${outputs.answer}`

    const structuredLLM = this.relevanceLLM.withStructuredOutput(this.relevanceOutput)
    const messages = [{ role: 'system', content: this.relevanceInstructions }, { role: 'user', content: answer }]
    const grade = await structuredLLM.invoke(messages)
    return grade.relevant
  }
}
