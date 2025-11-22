import { OpenAIProvider } from './openai-provider'
import { GoogleProvider } from './google-provider'
import { BaseAIProvider, type AIModelConfig } from './base-provider'
import type { ParsedResume } from '../../types'

/**
 * Available AI models with their configurations
 */
export const AI_MODELS = {
  // OpenAI Models
  'gpt-5-nano': {
    provider: 'openai',
    model: 'gpt-5-nano',
    maxTokens: 4000,
    temperature: 0.3,
    costPer1MInput: 0.05,
    costPer1MOutput: 0.40,
    tier: 'budget',
    description: '40x cheaper than Claude, great for standard screening'
  },
  'gpt-5-mini': {
    provider: 'openai',
    model: 'gpt-5-mini',
    maxTokens: 4000,
    temperature: 0.3,
    costPer1MInput: 0.25,
    costPer1MOutput: 2.00,
    tier: 'balanced',
    description: 'Good balance of cost and quality, supports caching'
  },
  'gpt-4o-mini': {
    provider: 'openai',
    model: 'gpt-4o-mini',
    maxTokens: 4000,
    temperature: 0.3,
    costPer1MInput: 0.15,
    costPer1MOutput: 0.60,
    tier: 'budget',
    description: 'Fast and cheap, good structured output'
  },

  // Google Models
  'gemini-2.5-flash': {
    provider: 'google',
    model: 'gemini-2.5-flash',
    maxTokens: 4000,
    temperature: 0.3,
    costPer1MInput: 0.15,
    costPer1MOutput: 0.60,
    tier: 'budget',
    description: '1M context, thinking budget, 21x cheaper'
  },
  'gemini-2.5-flash-thinking': {
    provider: 'google',
    model: 'gemini-2.5-flash',
    maxTokens: 4000,
    temperature: 0.3,
    costPer1MInput: 0.15,
    costPer1MOutput: 3.50,  // With thinking enabled
    tier: 'premium',
    description: 'Advanced reasoning for complex cases'
  },

  // Anthropic (legacy/fallback)
  'claude-sonnet-4': {
    provider: 'anthropic',
    model: 'claude-sonnet-4-20250514',
    maxTokens: 4000,
    temperature: 0.3,
    costPer1MInput: 3.00,
    costPer1MOutput: 15.00,
    tier: 'premium',
    description: 'Highest quality, most expensive'
  }
} as const

export type ModelName = keyof typeof AI_MODELS

/**
 * Model routing logic based on resume/job complexity
 */
export class ModelRouter {
  private defaultModel: ModelName = 'gpt-5-nano'
  private complexModel: ModelName = 'gemini-2.5-flash'
  private premiumModel: ModelName = 'gpt-5-mini'

  /**
   * Select the optimal model based on screening complexity
   */
  selectModelForScreening(
    resume: ParsedResume,
    jobDescription: string
  ): ModelName {
    const complexityScore = this.calculateComplexity(resume, jobDescription)

    // High complexity (20% of cases) - use premium model
    if (complexityScore >= 7) {
      console.log(`Using premium model (complexity: ${complexityScore})`)
      return this.premiumModel
    }

    // Medium complexity (15% of cases) - use balanced model
    if (complexityScore >= 5) {
      console.log(`Using complex model (complexity: ${complexityScore})`)
      return this.complexModel
    }

    // Low complexity (65% of cases) - use budget model
    console.log(`Using default model (complexity: ${complexityScore})`)
    return this.defaultModel
  }

  /**
   * Calculate complexity score (0-10)
   */
  private calculateComplexity(
    resume: ParsedResume,
    jobDescription: string
  ): number {
    let score = 0

    // Resume parsing confidence
    if (resume.confidence_score < 0.85) score += 2
    else if (resume.confidence_score < 0.90) score += 1

    // Job description length
    if (jobDescription.length > 3000) score += 2
    else if (jobDescription.length > 2000) score += 1

    // Seniority indicators
    const seniorityKeywords = ['senior', 'lead', 'director', 'principal', 'vp', 'chief']
    if (seniorityKeywords.some(kw => jobDescription.toLowerCase().includes(kw))) {
      score += 2
    }

    // Years of experience
    if (resume.total_years_experience > 15) score += 2
    else if (resume.total_years_experience > 10) score += 1

    // Number of certifications
    if (resume.certifications.length > 5) score += 1
    if (resume.certifications.length > 10) score += 1

    // Education complexity (advanced degrees)
    const hasAdvancedDegree = resume.education.some(edu =>
      edu.degree.toLowerCase().includes('phd') ||
      edu.degree.toLowerCase().includes('doctorate') ||
      edu.degree.toLowerCase().includes('master')
    )
    if (hasAdvancedDegree) score += 1

    return Math.min(score, 10) // Cap at 10
  }

  /**
   * Get provider instance for a model
   */
  getProvider(modelName: ModelName): BaseAIProvider {
    const config = AI_MODELS[modelName] as AIModelConfig

    switch (config.provider) {
      case 'openai':
        return new OpenAIProvider(config)
      case 'google':
        return new GoogleProvider(config)
      case 'anthropic':
        // Import dynamically to avoid circular dependency
        const { AnthropicProvider } = require('./anthropic-provider')
        return new AnthropicProvider(config)
      default:
        throw new Error(`Unknown provider: ${config.provider}`)
    }
  }

  /**
   * Get cost estimate for a model
   */
  getEstimatedCost(modelName: ModelName): number {
    const config = AI_MODELS[modelName]
    // Estimate: 2500 input tokens + 1000 output tokens
    return ((2500 / 1_000_000) * config.costPer1MInput) +
           ((1000 / 1_000_000) * config.costPer1MOutput)
  }

  /**
   * Get all available models sorted by cost
   */
  getModelsByCost(): Array<{ name: ModelName; cost: number; description: string }> {
    return (Object.keys(AI_MODELS) as ModelName[])
      .map(name => ({
        name,
        cost: this.getEstimatedCost(name),
        description: AI_MODELS[name].description
      }))
      .sort((a, b) => a.cost - b.cost)
  }
}

// Export singleton
export const modelRouter = new ModelRouter()
