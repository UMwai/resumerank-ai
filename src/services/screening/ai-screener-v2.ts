import { modelRouter, type ModelName } from '../../lib/ai-providers/model-router'
import type { ParsedResume, ScreeningResult } from '../../types'
import { nanoid } from 'nanoid'

export interface ScreeningOptions {
  jobDescription: string
  jobTitle?: string
  industry?: 'healthcare' | 'it' | 'warehouse' | 'general'
  forceModel?: ModelName  // Override automatic routing
}

export interface ScreeningMetrics {
  modelUsed: string
  provider: string
  estimatedCost: number
  complexityScore: number
  processingTimeMs: number
}

export class AIScreenerServiceV2 {
  /**
   * Screen a resume against a job description with intelligent model routing
   */
  async screenResume(
    resume: ParsedResume,
    options: ScreeningOptions
  ): Promise<ScreeningResult & { metrics: ScreeningMetrics }> {
    const startTime = Date.now()

    // Select optimal model (or use forced model)
    const modelName = options.forceModel ||
      modelRouter.selectModelForScreening(resume, options.jobDescription)

    // Get provider instance
    const provider = modelRouter.getProvider(modelName)

    console.log(`Screening with ${provider.getProviderName()}/${provider.getModelName()}`)

    // Perform screening
    const result = await provider.screenResume({
      resume,
      jobDescription: options.jobDescription,
      jobTitle: options.jobTitle,
      industry: options.industry
    })

    const processingTime = Date.now() - startTime

    // Calculate metrics
    const metrics: ScreeningMetrics = {
      modelUsed: provider.getModelName(),
      provider: provider.getProviderName(),
      estimatedCost: provider.getEstimatedCostPerResume(),
      complexityScore: this.calculateComplexityScore(resume, options.jobDescription),
      processingTimeMs: processingTime
    }

    return {
      id: `screening_${nanoid()}`,
      ...result,
      processing_time_ms: processingTime,
      ai_model_used: `${provider.getProviderName()}/${provider.getModelName()}`,
      metrics
    }
  }

  /**
   * Batch screen multiple resumes (uses most efficient model distribution)
   */
  async batchScreen(
    resumes: ParsedResume[],
    options: ScreeningOptions
  ): Promise<Array<ScreeningResult & { metrics: ScreeningMetrics }>> {
    // Process in parallel with intelligent routing for each
    const results = await Promise.all(
      resumes.map(resume => this.screenResume(resume, options))
    )

    // Log batch statistics
    const totalCost = results.reduce((sum, r) => sum + r.metrics.estimatedCost, 0)
    const avgCost = totalCost / results.length

    console.log(`Batch screening complete:`)
    console.log(`  - Resumes: ${results.length}`)
    console.log(`  - Total cost: $${totalCost.toFixed(4)}`)
    console.log(`  - Avg cost/resume: $${avgCost.toFixed(4)}`)

    return results
  }

  /**
   * Get model recommendations for a resume/job combination
   */
  getModelRecommendation(
    resume: ParsedResume,
    jobDescription: string
  ): {
    recommended: ModelName
    alternatives: Array<{ model: ModelName; cost: number; description: string }>
    complexityScore: number
  } {
    const recommended = modelRouter.selectModelForScreening(resume, jobDescription)
    const alternatives = modelRouter.getModelsByCost()
    const complexityScore = this.calculateComplexityScore(resume, jobDescription)

    return {
      recommended,
      alternatives,
      complexityScore
    }
  }

  /**
   * Calculate complexity score (0-10)
   */
  private calculateComplexityScore(
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

    // Certifications
    if (resume.certifications.length > 5) score += 1
    if (resume.certifications.length > 10) score += 1

    // Advanced degrees
    const hasAdvancedDegree = resume.education.some(edu =>
      edu.degree.toLowerCase().includes('phd') ||
      edu.degree.toLowerCase().includes('doctorate') ||
      edu.degree.toLowerCase().includes('master')
    )
    if (hasAdvancedDegree) score += 1

    return Math.min(score, 10)
  }

  /**
   * Get cost breakdown by model tier
   */
  getCostProjections(monthlyResumes: number): {
    allBudget: number
    allPremium: number
    intelligent: number
    savings: number
  } {
    const budgetCost = modelRouter.getEstimatedCost('gpt-5-nano')
    const premiumCost = modelRouter.getEstimatedCost('claude-sonnet-4')

    // Intelligent routing: 65% budget, 15% balanced, 20% premium
    const intelligentCost =
      (0.65 * budgetCost) +
      (0.15 * modelRouter.getEstimatedCost('gemini-2.5-flash')) +
      (0.20 * modelRouter.getEstimatedCost('gpt-5-mini'))

    return {
      allBudget: budgetCost * monthlyResumes,
      allPremium: premiumCost * monthlyResumes,
      intelligent: intelligentCost * monthlyResumes,
      savings: (premiumCost * monthlyResumes) - (intelligentCost * monthlyResumes)
    }
  }
}

// Export singleton
export const aiScreenerV2 = new AIScreenerServiceV2()
