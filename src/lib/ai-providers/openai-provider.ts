import OpenAI from 'openai'
import { BaseAIProvider, type AIModelConfig, type ScreeningRequest, type ScreeningResponse } from './base-provider'
import type { ParsedResume } from '../../types'

export class OpenAIProvider extends BaseAIProvider {
  private client: OpenAI

  constructor(config: AIModelConfig) {
    super(config)
    this.client = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY
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

Scoring Guidelines:
- 90-100: Exceptional match, exceeds requirements
- 80-89: Strong match, meets all key requirements
- 70-79: Good match, meets most requirements
- 60-69: Moderate match, missing some key requirements
- Below 60: Poor match, significant gaps

Be specific: Cite exact years, certifications, companies from the resume.`

    try {
      const completion = await this.client.chat.completions.create({
        model: this.config.model,
        messages: [
          {
            role: 'system',
            content: 'You are an expert resume screening assistant. Always respond with valid JSON only.'
          },
          {
            role: 'user',
            content: prompt
          }
        ],
        response_format: { type: 'json_object' },
        max_tokens: this.config.maxTokens,
        temperature: this.config.temperature || 0.3
      })

      const responseText = completion.choices[0].message.content || '{}'
      const result = JSON.parse(responseText)

      return result as ScreeningResponse
    } catch (error) {
      console.error('OpenAI screening error:', error)
      throw new Error(`OpenAI screening failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
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
      const completion = await this.client.chat.completions.create({
        model: this.config.model,
        messages: [
          {
            role: 'system',
            content: 'You are an expert resume parser. Always respond with valid JSON only.'
          },
          {
            role: 'user',
            content: prompt
          }
        ],
        response_format: { type: 'json_object' },
        max_tokens: 4000,
        temperature: 0.1
      })

      const responseText = completion.choices[0].message.content || '{}'
      return JSON.parse(responseText) as ParsedResume
    } catch (error) {
      console.error('OpenAI parsing error:', error)
      throw new Error(`OpenAI parsing failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  private getIndustryContext(industry?: string): string {
    switch (industry) {
      case 'healthcare':
        return `Healthcare Staffing Context:
- Prioritize active licenses and certifications (RN, LPN, BLS, ACLS)
- ICU/critical care experience > general med-surg
- Level 1 trauma center experience indicates high-acuity skills
- Recent experience (within 2 years) critical for clinical skills`

      case 'it':
        return `IT Staffing Context:
- Modern tech stacks (React, TypeScript, AWS) > legacy (jQuery, PHP)
- Recent projects and continuous learning indicate up-to-date skills
- Leadership/mentorship experience matters for senior roles
- Certifications (AWS, GCP, Azure) validate cloud skills`

      case 'warehouse':
        return `Warehouse/Logistics Context:
- Forklift certification often mandatory
- Safety record and attendance critical
- WMS experience valuable
- Reliability matters more than advanced skills`

      default:
        return `General Staffing Context:
- Match candidate experience to role requirements precisely
- Look for career progression and stability
- Relevant industry experience matters`
    }
  }
}
