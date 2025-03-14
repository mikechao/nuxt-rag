/**
    Groundedness: Response vs retrieved docs
 
    Goal: Measure "to what extent does the generated response agree with the retrieved context"
    Mode: Does not require reference answer, because it will compare the answer to the retrieved context
    Evaluator: Use LLM-as-judge to assess faithfulness, hallucinations, etc.
 
 */

import { ChatOpenAI } from '@langchain/openai'
import { z } from 'zod'

export class Groundedness {
  private groundedLLM: ChatOpenAI
  private groundedOutput = z
    .object({
      explanation: z
        .string()
        .describe('Explain your reasoning for the score'),
      grounded: z
        .boolean()
        .describe('Provide the score on if the answer hallucinates from the documents'),
    })
    .describe('Grounded score for the answer from the retrieved documents.')

  private groundedInstructions = `You are a teacher grading a quiz. 

  You will be given FACTS and a STUDENT ANSWER. 
  
  Here is the grade criteria to follow:
  (1) Ensure the STUDENT ANSWER is grounded in the FACTS. 
  (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.
  
  Grounded:
  A grounded value of True means that the student's answer meets all of the criteria.
  A grounded value of False means that the student's answer does not meet all of the criteria.
  
  Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 
  
  Avoid simply stating the correct answer at the outset.`

  constructor(openAIKey: string) {
    this.groundedLLM = new ChatOpenAI({
      model: 'gpt-4o-mini',
      temperature: 0,
      apiKey: openAIKey,
    })
  }

  async grounded({ inputs, outputs }: { inputs: Record<string, any>, outputs: Record<string, any> }) {
    const docString = outputs.documents.map(doc => doc.pageContent).join('')
    const answer = `FACTS: ${docString} STUDENT ANSWER: ${outputs.answer}`

    const structuredLLM = this.groundedLLM.withStructuredOutput(this.groundedOutput)
    const messages = [{ role: 'system', content: this.groundedInstructions }, { role: 'user', content: answer }]
    const grade = await structuredLLM.invoke(messages)
    return grade.grounded
  }
}
