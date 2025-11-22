import pdf from 'pdf-parse'
import mammoth from 'mammoth'
import Anthropic from '@anthropic-ai/sdk'
import type { ParsedResume } from '../../types'

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY
})

export class ResumeParserService {
  /**
   * Parse a resume file (PDF or DOCX) and extract structured data
   */
  async parseResume(fileBuffer: Buffer, filename: string): Promise<ParsedResume> {
    const startTime = Date.now()

    // Step 1: Extract text from file
    const resumeText = await this.extractText(fileBuffer, filename)

    // Step 2: Use Claude to parse into structured format
    const parsed = await this.parseWithAI(resumeText)

    const processingTime = Date.now() - startTime
    console.log(`Resume parsed in ${processingTime}ms`)

    return parsed
  }

  /**
   * Extract text content from PDF or DOCX file
   */
  private async extractText(fileBuffer: Buffer, filename: string): Promise<string> {
    const extension = filename.toLowerCase().split('.').pop()

    if (extension === 'pdf') {
      return await this.extractFromPDF(fileBuffer)
    } else if (extension === 'docx' || extension === 'doc') {
      return await this.extractFromDOCX(fileBuffer)
    } else if (extension === 'txt') {
      return fileBuffer.toString('utf-8')
    } else {
      throw new Error(`Unsupported file type: ${extension}`)
    }
  }

  /**
   * Extract text from PDF using pdf-parse
   */
  private async extractFromPDF(buffer: Buffer): Promise<string> {
    try {
      const data = await pdf(buffer)
      return data.text
    } catch (error) {
      console.error('PDF parsing error:', error)
      throw new Error('Failed to parse PDF file')
    }
  }

  /**
   * Extract text from DOCX using mammoth
   */
  private async extractFromDOCX(buffer: Buffer): Promise<string> {
    try {
      const result = await mammoth.extractRawText({ buffer })
      return result.value
    } catch (error) {
      console.error('DOCX parsing error:', error)
      throw new Error('Failed to parse DOCX file')
    }
  }

  /**
   * Use Claude AI to parse resume text into structured data
   */
  private async parseWithAI(resumeText: string): Promise<ParsedResume> {
    const prompt = `Parse this resume and extract structured information.

RESUME TEXT:
${resumeText}

Extract the following information and return ONLY valid JSON (no markdown, no explanation):

{
  "candidate": {
    "name": "string",
    "email": "string or null",
    "phone": "string or null",
    "location": "string or null",
    "linkedin": "string (URL) or null",
    "website": "string (URL) or null"
  },
  "experience": [
    {
      "company": "string",
      "title": "string",
      "start_date": "YYYY-MM or YYYY",
      "end_date": "YYYY-MM or YYYY or 'present'",
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
      "start_date": "YYYY or null",
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
      "expiry_date": "YYYY-MM or null",
      "credential_id": "string or null"
    }
  ],
  "total_years_experience": number,
  "summary": "string (2-3 sentence professional summary) or null",
  "confidence_score": number (0.0 to 1.0, how confident you are in the extraction)
}

Rules:
1. Be precise with dates (use YYYY-MM format when month is clear, YYYY otherwise)
2. Calculate duration_months accurately
3. Extract ALL skills mentioned (technical and soft skills)
4. If information is missing or unclear, use null
5. For confidence_score: 1.0 = very clear resume, 0.5 = some ambiguity, 0.2 = very unclear
6. Return ONLY the JSON object, no other text`

    try {
      const message = await anthropic.messages.create({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 4000,
        messages: [{
          role: 'user',
          content: prompt
        }]
      })

      const responseText = message.content[0].type === 'text'
        ? message.content[0].text
        : ''

      // Parse the JSON response
      const parsed = JSON.parse(responseText)

      return parsed as ParsedResume
    } catch (error) {
      console.error('AI parsing error:', error)

      // Fallback: return basic structure
      return this.createFallbackParsedResume(resumeText)
    }
  }

  /**
   * Create a basic parsed resume structure when AI parsing fails
   */
  private createFallbackParsedResume(resumeText: string): ParsedResume {
    // Extract email using regex
    const emailMatch = resumeText.match(/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/)
    const email = emailMatch ? emailMatch[0] : null

    // Extract phone using regex
    const phoneMatch = resumeText.match(/(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}/)
    const phone = phoneMatch ? phoneMatch[0] : null

    // Extract name (first line that looks like a name)
    const lines = resumeText.split('\n').map(l => l.trim()).filter(l => l)
    const name = lines[0] || 'Unknown'

    return {
      candidate: {
        name,
        email: email || undefined,
        phone: phone || undefined,
        location: undefined,
        linkedin: undefined
      },
      experience: [],
      education: [],
      skills: [],
      certifications: [],
      total_years_experience: 0,
      confidence_score: 0.3 // Low confidence fallback
    }
  }
}

// Export singleton instance
export const resumeParser = new ResumeParserService()
