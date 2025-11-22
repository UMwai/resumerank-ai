import Anthropic from '@anthropic-ai/sdk'
import { BaseAIProvider, type AIModelConfig, type ScreeningRequest, type ScreeningResponse } from './base-provider'
import type { ParsedResume } from '../../types'

export class AnthropicProvider extends BaseAIProvider {
  private client: Anthropic

  constructor(config: AIModelConfig) {
    super(config)
    this.client = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY
    })
  }

  async screenResume(request: ScreeningRequest): Promise<ScreeningResponse> {
    const industryContext = this.getIndustryContext(request.industry)

    const prompt = `You are an expert ${request.industry || 'general'} recruiter screening a candidate.

${industryContext}

JOB DESCRIPTION:
${request.jobDescription}

CANDIDATE RESUME:
${JSON.stringify(request.resume, null, 2)}

Analyze this candidate and return ONLY valid JSON matching this schema:

{
  "match_score": number (0-100),
  "recommendation": "strong_yes" | "yes" | "maybe" | "no",
  "summary": "string (2-3 sentences)",
  "matched_requirements": [
    {
      "requirement": "string",
      "evidence": "string",
      "confidence": number (0.0-1.0)
    }
  ],
  "missing_requirements": ["string"],
  "strengths": ["string"],
  "concerns": ["string"],
  "interview_questions": ["string"]
}

Be specific: Cite exact years, certifications, companies from the resume.`

    try {
      const message = await this.client.messages.create({
        model: this.config.model,
        max_tokens: this.config.maxTokens,
        temperature: this.config.temperature || 0.3,
        messages: [{
          role: 'user',
          content: prompt
        }]
      })

      const responseText = message.content[0].type === 'text'
        ? message.content[0].text
        : ''

      const parsed = JSON.parse(responseText)
      return parsed as ScreeningResponse
    } catch (error) {
      console.error('Anthropic screening error:', error)
      throw new Error(`Anthropic screening failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  async parseResume(resumeText: string): Promise<ParsedResume> {
    const prompt = `Parse this resume and extract structured information.

RESUME TEXT:
${resumeText}

Return ONLY valid JSON matching this schema:

{
  "candidate": {
    "name": "string",
    "email": "string or null",
    "phone": "string or null",
    "location": "string or null",
    "linkedin": "string (URL) or null"
  },
  "experience": [
    {
      "company": "string",
      "title": "string",
      "start_date": "YYYY-MM",
      "end_date": "YYYY-MM or 'present'",
      "duration_months": number,
      "location": "string or null",
      "highlights": ["string"]
    }
  ],
  "education": [
    {
      "institution": "string",
      "degree": "string",
      "field": "string or null",
      "end_date": "YYYY or null",
      "gpa": number or null
    }
  ],
  "skills": ["string"],
  "certifications": [
    {
      "name": "string",
      "issuer": "string or null",
      "issue_date": "YYYY-MM or null",
      "expiry_date": "YYYY-MM or null"
    }
  ],
  "total_years_experience": number,
  "summary": "string or null",
  "confidence_score": number (0.0-1.0)
}`

    try {
      const message = await this.client.messages.create({
        model: this.config.model,
        max_tokens: 4000,
        temperature: 0.1,
        messages: [{
          role: 'user',
          content: prompt
        }]
      })

      const responseText = message.content[0].type === 'text'
        ? message.content[0].text
        : ''

      return JSON.parse(responseText) as ParsedResume
    } catch (error) {
      console.error('Anthropic parsing error:', error)
      throw new Error(`Anthropic parsing failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  private getIndustryContext(industry?: string): string {
    switch (industry) {
      case 'healthcare':
        return `Healthcare Staffing Context:
- Prioritize active licenses and certifications
- ICU/critical care experience > general med-surg
- Recent experience critical for clinical skills`

      case 'it':
        return `IT Staffing Context:
- Modern tech stacks preferred
- Recent projects indicate up-to-date skills
- Certifications validate expertise`

      case 'warehouse':
        return `Warehouse/Logistics Context:
- Forklift certification often mandatory
- Safety record critical
- Reliability matters most`

      default:
        return `General Staffing Context:
- Match experience to requirements
- Look for progression and stability`
    }
  }
}
