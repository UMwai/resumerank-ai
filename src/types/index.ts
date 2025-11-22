import { z } from 'zod'

// ===== Resume Data Types =====

export const ContactInfoSchema = z.object({
  name: z.string(),
  email: z.string().email().optional(),
  phone: z.string().optional(),
  location: z.string().optional(),
  linkedin: z.string().url().optional(),
  website: z.string().url().optional()
})

export const WorkExperienceSchema = z.object({
  company: z.string(),
  title: z.string(),
  start_date: z.string(), // ISO date or "YYYY-MM"
  end_date: z.string().or(z.literal('present')),
  duration_months: z.number().optional(),
  location: z.string().optional(),
  highlights: z.array(z.string())
})

export const EducationSchema = z.object({
  institution: z.string(),
  degree: z.string(),
  field: z.string().optional(),
  start_date: z.string().optional(),
  end_date: z.string().optional(),
  gpa: z.number().optional()
})

export const CertificationSchema = z.object({
  name: z.string(),
  issuer: z.string().optional(),
  issue_date: z.string().optional(),
  expiry_date: z.string().optional(),
  credential_id: z.string().optional()
})

export const ParsedResumeSchema = z.object({
  candidate: ContactInfoSchema,
  experience: z.array(WorkExperienceSchema),
  education: z.array(EducationSchema),
  skills: z.array(z.string()),
  certifications: z.array(CertificationSchema),
  total_years_experience: z.number(),
  summary: z.string().optional(),
  confidence_score: z.number().min(0).max(1)
})

export type ParsedResume = z.infer<typeof ParsedResumeSchema>

// ===== Screening Types =====

export const MatchedRequirementSchema = z.object({
  requirement: z.string(),
  evidence: z.string(),
  confidence: z.number().min(0).max(1).optional()
})

export const ScreeningResultSchema = z.object({
  id: z.string(),
  match_score: z.number().min(0).max(100),
  recommendation: z.enum(['strong_yes', 'yes', 'maybe', 'no']),
  summary: z.string(),
  matched_requirements: z.array(MatchedRequirementSchema),
  missing_requirements: z.array(z.string()),
  strengths: z.array(z.string()),
  concerns: z.array(z.string()),
  interview_questions: z.array(z.string()),
  processing_time_ms: z.number(),
  ai_model_used: z.string()
})

export type ScreeningResult = z.infer<typeof ScreeningResultSchema>

// ===== API Request/Response Types =====

export const ParseResumeRequestSchema = z.object({
  file: z.string(), // base64 encoded
  filename: z.string()
})

export const ScreenResumeRequestSchema = z.object({
  resume: z.union([ParsedResumeSchema, z.string()]), // Parsed object or raw text
  job_description: z.string().min(50),
  job_title: z.string().optional(),
  industry: z.enum(['healthcare', 'it', 'warehouse', 'general']).optional()
})

export const BatchUploadRequestSchema = z.object({
  job_description: z.string().min(50),
  job_title: z.string().optional(),
  industry: z.enum(['healthcare', 'it', 'warehouse', 'general']).optional(),
  webhook_url: z.string().url().optional()
})

// ===== Database Types =====

export interface User {
  id: string
  email: string
  name: string | null
  company_name: string | null
  industry: string | null
  created_at: Date
  updated_at: Date
}

export interface ApiKey {
  id: string
  user_id: string
  key_hash: string
  key_prefix: string // First 8 chars for display
  name: string | null
  last_used_at: Date | null
  created_at: Date
  revoked_at: Date | null
}

export interface Subscription {
  id: string
  user_id: string
  stripe_customer_id: string | null
  stripe_subscription_id: string | null
  plan: 'free' | 'starter' | 'professional' | 'enterprise'
  status: 'active' | 'canceled' | 'past_due'
  current_period_start: Date | null
  current_period_end: Date | null
  cancel_at: Date | null
  created_at: Date
  updated_at: Date
}

export interface Screening {
  id: string
  user_id: string
  api_key_id: string | null
  resume_filename: string | null
  resume_file_url: string | null
  job_description: string | null
  job_title: string | null
  industry: string | null
  parsed_data: ParsedResume | null
  match_score: number | null
  recommendation: 'strong_yes' | 'yes' | 'maybe' | 'no' | null
  summary: string | null
  matched_requirements: Array<{ requirement: string; evidence: string }> | null
  missing_requirements: string[] | null
  strengths: string[] | null
  concerns: string[] | null
  interview_questions: string[] | null
  processing_time_ms: number | null
  ai_model_used: string | null
  confidence_score: number | null
  user_rating: number | null
  user_feedback: string | null
  created_at: Date
}

export interface BatchJob {
  id: string
  user_id: string
  job_description: string
  industry: string | null
  total_resumes: number
  completed_resumes: number
  failed_resumes: number
  status: 'processing' | 'completed' | 'failed'
  results_file_url: string | null
  webhook_url: string | null
  created_at: Date
  completed_at: Date | null
}

// ===== Config Types =====

export interface AppConfig {
  port: number
  nodeEnv: 'development' | 'production' | 'test'
  database: {
    url: string
  }
  redis: {
    url: string
  }
  anthropic: {
    apiKey: string
  }
  stripe: {
    secretKey: string
    publishableKey: string
    webhookSecret: string
  }
  aws: {
    accessKeyId: string
    secretAccessKey: string
    region: string
    s3Bucket: string
  }
}

// ===== AI Models =====

export type AIModel = 'claude-sonnet-4-20250514' | 'claude-haiku-20250301'

export interface AIModelConfig {
  model: AIModel
  maxTokens: number
  temperature: number
}
