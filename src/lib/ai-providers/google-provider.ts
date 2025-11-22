import { GoogleGenerativeAI } from '@google/generative-ai'
import { BaseAIProvider, type AIModelConfig, type ScreeningRequest, type ScreeningResponse } from './base-provider'
import type { ParsedResume } from '../../types'

export class GoogleProvider extends BaseAIProvider {
  private client: GoogleGenerativeAI

  constructor(config: AIModelConfig) {
    super(config)
    this.client = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY || '')
  }

  async screenResume(request: ScreeningRequest): Promise<ScreeningResponse> {
    const model = this.client.getGenerativeModel({
      model: this.config.model,
      generationConfig: {
        temperature: this.config.temperature || 0.3,
        maxOutputTokens: this.config.maxTokens,
        responseMimeType: 'application/json'
      }
    })

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
- 90-100: Exceptional match
- 80-89: Strong match, meets all key requirements
- 70-79: Good match
- 60-69: Moderate match
- Below 60: Poor match

Be specific: Cite exact years, certifications, companies from the resume.`

    try {
      const result = await model.generateContent(prompt)
      const responseText = result.response.text()
      const parsed = JSON.parse(responseText)

      return parsed as ScreeningResponse
    } catch (error) {
      console.error('Google AI screening error:', error)
      throw new Error(`Google AI screening failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  async parseResume(resumeText: string): Promise<ParsedResume> {
    const model = this.client.getGenerativeModel({
      model: this.config.model,
      generationConfig: {
        temperature: 0.1,
        maxOutputTokens: 4000,
        responseMimeType: 'application/json'
      }
    })

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
      const result = await model.generateContent(prompt)
      const responseText = result.response.text()
      return JSON.parse(responseText) as ParsedResume
    } catch (error) {
      console.error('Google AI parsing error:', error)
      throw new Error(`Google AI parsing failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  private getIndustryContext(industry?: string): string {
    switch (industry) {
      case 'healthcare':
        return `Healthcare Staffing Context:
- Prioritize active licenses and certifications (RN, LPN, BLS, ACLS)
- ICU/critical care experience > general med-surg
- Level 1 trauma center experience indicates high-acuity skills`

      case 'it':
        return `IT Staffing Context:
- Modern tech stacks (React, TypeScript, AWS) > legacy
- Recent projects indicate up-to-date skills
- Certifications validate cloud skills`

      case 'warehouse':
        return `Warehouse/Logistics Context:
- Forklift certification often mandatory
- Safety record and attendance critical
- Reliability matters most`

      default:
        return `General Staffing Context:
- Match candidate experience to requirements
- Look for career progression and stability`
    }
  }
}
