import { ChatOpenAI } from '@langchain/openai'
import { z } from 'zod'

/**
 *
    Correctness: Response vs reference answer
 
    Goal: Measure "how similar/correct is the RAG chain answer, relative to a ground-truth answer"
    Mode: Requires a ground truth (reference) answer supplied through a dataset
    Evaluator: Use LLM-as-judge to assess answer correctness.
 
    https://docs.smith.langchain.com/evaluation/tutorials/rag#correctness-response-vs-reference-answer
 */
export class Correctness {
  private graderLLM: ChatOpenAI
  private graderOutput = z.object({
    explanation: z.string().describe('Explain your reasoning for the score'),
    correct: z.boolean().describe('True if the answer is correct, False otherwise.'),
  }).describe('Correctness score for reference answer v.s. generated answer.')

  private correctnessInstructions = `You are a teacher grading a quiz. 

  You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. 
  
  Here is the grade criteria to follow:
  (1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. 
  (2) Ensure that the student answer does not contain any conflicting statements.
  (3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.
  
  Correctness:
  A correctness value of True means that the student's answer meets all of the criteria.
  A correctness value of False means that the student's answer does not meet all of the criteria.
  
  Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 
  
  Avoid simply stating the correct answer at the outset.`

  constructor(openAIKey: string) {
    this.graderLLM = new ChatOpenAI({
      model: 'gpt-4o-mini',
      temperature: 0,
      apiKey: openAIKey,
    })
  }

  async correctness({ inputs, outputs, referenceOutput }: { inputs: Record<string, any>, outputs: Record<string, any>, referenceOutput: Record<string, any> }) {
    const answer = `QUESTION: ${inputs.input}
    GROUND TRUTH: ${referenceOutput.answer}
    STUDENT ANSWER: ${outputs.answer}
    ${this.correctnessInstructions}`

    const structuredLLM = this.graderLLM.withStructuredOutput(this.graderOutput)
    const messages = [{ role: 'system', content: this.correctnessInstructions }, { role: 'user', content: answer }]
    const grade = await structuredLLM.invoke(messages)
    return grade.correct
  }
}
