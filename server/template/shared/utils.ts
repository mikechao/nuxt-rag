import type { BaseChatModel } from '@langchain/core/language_models/chat_models'

import type { Document } from 'langchain/document'
import { LocalFileCache } from 'langchain/cache/file_system'
import { initChatModel } from 'langchain/chat_models/universal'

export function formatDoc(doc: Document): string {
  const metadata = doc.metadata || {}
  const meta = Object.entries(metadata)
    .map(([k, v]) => ` ${k}=${v}`)
    .join('')
  const metaStr = meta ? ` ${meta}` : ''

  return `<document${metaStr}>\n${doc.pageContent}\n</document>`
}

export function formatDocs(docs?: Document[]): string {
  /** Format a list of documents as XML. */
  if (!docs || docs.length === 0) {
    return '<documents></documents>'
  }
  const formatted = docs.map(formatDoc).join('\n')
  return `<documents>\n${formatted}\n</documents>`
}

/**
 * Load a chat model from a fully specified name.
 * @param fullySpecifiedName - String in the format 'provider/model' or 'provider/account/provider/model'.
 * @returns A Promise that resolves to a BaseChatModel instance.
 */
export async function loadChatModel(
  fullySpecifiedName: string,
  useLocalCache: boolean = false,
): Promise<BaseChatModel> {
  const index = fullySpecifiedName.indexOf('/')
  const cache = useLocalCache ? await LocalFileCache.create('rag-template-cache') : undefined
  if (index === -1) {
    // If there's no "/", assume it's just the model
    return await initChatModel(fullySpecifiedName, { cache })
  }
  else {
    const provider = fullySpecifiedName.slice(0, index)
    const model = fullySpecifiedName.slice(index + 1)
    return await initChatModel(model, { modelProvider: provider, cache })
  }
}
