import type { ParsedResume, ScreeningResult } from '../../types'

export interface AIModelConfig {
  provider: 'openai' | 'google' | 'anthropic'
  model: string
  maxTokens: number
  temperature?: number
  costPer1MInput: number  // USD
  costPer1MOutput: number // USD
}

export interface ScreeningRequest {
  resume: ParsedResume
  jobDescription: string
  jobTitle?: string
  industry?: 'healthcare' | 'it' | 'warehouse' | 'general'
}

export interface ScreeningResponse {
  match_score: number
  recommendation: 'strong_yes' | 'yes' | 'maybe' | 'no'
  summary: string
  matched_requirements: Array<{
    requirement: string
    evidence: string
    confidence?: number
  }>
  missing_requirements: string[]
  strengths: string[]
  concerns: string[]
  interview_questions: string[]
}

export abstract class BaseAIProvider {
  protected config: AIModelConfig

  constructor(config: AIModelConfig) {
    this.config = config
  }

  /**
   * Screen a resume against a job description
   */
  abstract screenResume(request: ScreeningRequest): Promise<ScreeningResponse>

  /**
   * Parse a resume from text
   */
  abstract parseResume(resumeText: string): Promise<ParsedResume>

  /**
   * Calculate cost for a screening operation
   */
  calculateCost(inputTokens: number, outputTokens: number): number {
    const inputCost = (inputTokens / 1_000_000) * this.config.costPer1MInput
    const outputCost = (outputTokens / 1_000_000) * this.config.costPer1MOutput
    return inputCost + outputCost
  }

  /**
   * Estimate token count (rough approximation: 1 token ≈ 4 characters)
   */
  protected estimateTokens(text: string): number {
    return Math.ceil(text.length / 4)
  }

  /**
   * Get provider name
   */
  getProviderName(): string {
    return this.config.provider
  }

  /**
   * Get model name
   */
  getModelName(): string {
    return this.config.model
  }

  /**
   * Get cost per resume (estimated for 2500 input + 1000 output tokens)
   */
  getEstimatedCostPerResume(): number {
    return this.calculateCost(2500, 1000)
  }
}
