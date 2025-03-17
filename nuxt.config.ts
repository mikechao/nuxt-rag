import process from 'node:process'

// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2024-11-01',
  devtools: { enabled: true },
  runtimeConfig: {
    openaiAPIKey: '',
    langsmithAPIKey: '',
    browserbaseAPIKey: '',
    postgresURL: process.env.NUXT_POSTGRES_URL,
  },
})
