/**
 * AI Provider Abstraction Layer
 *
 * Supports multiple AI providers with intelligent routing based on complexity:
 * - OpenAI (GPT-5 nano, GPT-5 mini, GPT-4o-mini)
 * - Google (Gemini 2.5 Flash, Gemini 2.5 Flash Thinking)
 * - Anthropic (Claude Sonnet 4 - legacy/fallback)
 */

export { BaseAIProvider, type AIModelConfig, type ScreeningRequest, type ScreeningResponse } from './base-provider'
export { OpenAIProvider } from './openai-provider'
export { GoogleProvider } from './google-provider'
export { AnthropicProvider } from './anthropic-provider'
export { ModelRouter, modelRouter, AI_MODELS, type ModelName } from './model-router'
