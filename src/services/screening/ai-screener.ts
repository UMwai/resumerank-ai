import Anthropic from '@anthropic-ai/sdk'
import type { ParsedResume, ScreeningResult, AIModel } from '../../types'
import { nanoid } from 'nanoid'

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY
})

export interface ScreeningOptions {
  jobDescription: string
  jobTitle?: string
  industry?: 'healthcare' | 'it' | 'warehouse' | 'general'
}

export class AIScreenerService {
  /**
   * Screen a resume against a job description
   */
  async screenResume(
    resume: ParsedResume,
    options: ScreeningOptions
  ): Promise<ScreeningResult> {
    const startTime = Date.now()

    // Select appropriate AI model based on complexity
    const model = this.selectModel(resume, options)

    // Generate screening using Claude
    const result = await this.generateScreening(resume, options, model)

    const processingTime = Date.now() - startTime

    return {
      ...result,
      id: `screening_${nanoid()}`,
      processing_time_ms: processingTime,
      ai_model_used: model
    }
  }

  /**
   * Select the appropriate AI model based on complexity
   */
  private selectModel(resume: ParsedResume, options: ScreeningOptions): AIModel {
    // Use faster/cheaper Haiku model if:
    // 1. High parsing confidence (clear resume)
    // 2. Short job description
    // 3. Not a senior/lead role
    // 4. Standard industry

    const isSimpleCase =
      resume.confidence_score > 0.90 &&
      options.jobDescription.length < 2000 &&
      !options.jobDescription.toLowerCase().includes('senior') &&
      !options.jobDescription.toLowerCase().includes('lead') &&
      !options.jobDescription.toLowerCase().includes('director')

    if (isSimpleCase) {
      return 'claude-haiku-20250301' // Faster, cheaper
    }

    return 'claude-sonnet-4-20250514' // More powerful
  }

  /**
   * Generate screening analysis using Claude
   */
  private async generateScreening(
    resume: ParsedResume,
    options: ScreeningOptions,
    model: AIModel
  ): Promise<Omit<ScreeningResult, 'id' | 'processing_time_ms' | 'ai_model_used'>> {
    const industryContext = this.getIndustryContext(options.industry)

    const prompt = `You are an expert ${options.industry || 'general'} recruiter screening a candidate.

${industryContext}

JOB DESCRIPTION:
${options.jobDescription}

CANDIDATE RESUME:
${JSON.stringify(resume, null, 2)}

Analyze this candidate and return ONLY valid JSON (no markdown, no explanation):

{
  "match_score": number (0-100, how well candidate matches requirements),
  "recommendation": "strong_yes" | "yes" | "maybe" | "no",
  "summary": "string (2-3 sentences explaining overall fit)",
  "matched_requirements": [
    {
      "requirement": "string (specific requirement from job description)",
      "evidence": "string (specific evidence from resume)",
      "confidence": number (0.0-1.0, optional)
    }
  ],
  "missing_requirements": ["string (requirements from job description not met)"],
  "strengths": ["string (unique strengths not explicitly required but valuable)"],
  "concerns": ["string (potential red flags or concerns)"],
  "interview_questions": ["string (3 specific questions to ask this candidate)"]
}

Scoring Guidelines:
- 90-100: Exceptional match, exceeds requirements
- 80-89: Strong match, meets all key requirements
- 70-79: Good match, meets most requirements
- 60-69: Moderate match, missing some key requirements
- Below 60: Poor match, significant gaps

Recommendation Guidelines:
- strong_yes: 85+ score, hire immediately
- yes: 70-84 score, strong candidate worth interviewing
- maybe: 55-69 score, borderline, need more information
- no: <55 score, significant gaps

Be specific: Cite exact years, certifications, companies, or achievements from the resume.
Be objective: Focus on requirements match, not bias.
Be helpful: Interview questions should probe gaps or validate strengths.`

    try {
      const message = await anthropic.messages.create({
        model,
        max_tokens: 3000,
        messages: [{
          role: 'user',
          content: prompt
        }]
      })

      const responseText = message.content[0].type === 'text'
        ? message.content[0].text
        : ''

      const parsed = JSON.parse(responseText)

      return parsed
    } catch (error) {
      console.error('AI screening error:', error)
      throw new Error('Failed to generate screening analysis')
    }
  }

  /**
   * Get industry-specific context for screening
   */
  private getIndustryContext(industry?: string): string {
    switch (industry) {
      case 'healthcare':
        return `Healthcare Staffing Context:
- Prioritize active licenses and certifications (RN, LPN, BLS, ACLS, etc.)
- ICU/critical care experience is more valuable than general med-surg
- Level 1 trauma center experience indicates high-acuity skills
- Charge nurse or leadership experience is a strong plus
- Recent experience (within 2 years) is critical for clinical skills`

      case 'it':
        return `IT Staffing Context:
- Modern tech stacks (React, TypeScript, AWS) are more valuable than legacy (jQuery, PHP)
- Recent projects and continuous learning indicate up-to-date skills
- Open source contributions or side projects show passion
- Leadership/mentorship experience matters for senior roles
- Certifications (AWS, GCP, Azure) validate cloud skills`

      case 'warehouse':
        return `Warehouse/Logistics Staffing Context:
- Forklift certification is often mandatory
- Safety record and attendance are critical
- Experience with WMS (Warehouse Management Systems) is valuable
- Physical demands: ability to lift 50+ lbs often required
- Reliability and consistency matter more than advanced skills`

      default:
        return `General Staffing Context:
- Match candidate experience to role requirements precisely
- Look for career progression and stability
- Relevant industry experience matters
- Culture fit signals: company types, values mentioned
- Red flags: frequent job changes (<1 year), employment gaps (>6 months)`
    }
  }
}

// Export singleton instance
export const aiScreener = new AIScreenerService()
