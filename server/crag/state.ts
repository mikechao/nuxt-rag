import type { DocumentInterface } from '@langchain/core/documents'
import { Annotation } from '@langchain/langgraph'

export const GraphState = Annotation.Root({
  documents: Annotation<DocumentInterface[]>({
    reducer: (x, y) => y ?? x ?? [],
  }),
  question: Annotation<string>({
    reducer: (x, y) => y ?? x ?? '',
  }),
  generation: Annotation<string>({
    reducer: (x, y) => y ?? x,
  }),
})
