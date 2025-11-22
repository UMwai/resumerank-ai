# Technical Architecture

## System Overview

ResumeRank AI is a cloud-native, API-first SaaS platform built for speed, scalability, and cost efficiency.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CLIENT LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  Web UI          │    REST API      │   Webhooks      │    SDKs             │
│  (Next.js)       │   (Public)       │   (Callbacks)   │   (Python, JS)      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            API GATEWAY                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Authentication (API Keys + JWT)                                          │
│  • Rate Limiting (Redis-based)                                              │
│  • Request Validation                                                       │
│  • CORS handling                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         APPLICATION LAYER                                   │
├──────────────────────────┬──────────────────────┬───────────────────────────┤
│   Resume Parser          │  AI Screening         │  Batch Processor          │
│   Service                │  Service              │  Service                  │
│   ┌────────────────┐     │  ┌────────────────┐   │  ┌────────────────┐      │
│   │ PDF Parse      │     │  │ Claude API     │   │  │ Queue Worker   │      │
│   │ DOCX Parse     │────▶│  │ Prompt Eng     │   │  │ (BullMQ)       │      │
│   │ Text Extract   │     │  │ Response Parse │   │  │ S3 Upload      │      │
│   └────────────────┘     │  └────────────────┘   │  └────────────────┘      │
└──────────────────────────┴──────────────────────┴───────────────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                          │
├──────────────────────────┬──────────────────────┬───────────────────────────┤
│   PostgreSQL             │  Redis Cache          │  S3-Compatible Storage    │
│   (Supabase)             │  (Upstash)            │  (Cloudflare R2)          │
│   ┌────────────────┐     │  ┌────────────────┐   │  ┌────────────────┐      │
│   │ Users          │     │  │ API Keys       │   │  │ Resume Files   │      │
│   │ Subscriptions  │     │  │ Rate Limits    │   │  │ Batch Results  │      │
│   │ API Keys       │     │  │ Job Desc Cache │   │  │ Exports        │      │
│   │ Screenings     │     │  │ Session Data   │   │  │                │      │
│   └────────────────┘     │  └────────────────┘   │  └────────────────┘      │
└──────────────────────────┴──────────────────────┴───────────────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EXTERNAL SERVICES                                      │
├──────────────────────────┬──────────────────────┬───────────────────────────┤
│   Anthropic Claude       │  Stripe              │  PostHog / Analytics      │
│   (AI Screening)         │  (Payments)          │  (Product Analytics)      │
└──────────────────────────┴──────────────────────┴───────────────────────────┘
```

## Technology Stack

### Frontend

| Component | Technology | Reasoning |
|-----------|------------|-----------|
| **Framework** | Next.js 15 (App Router) | React SSR, API routes, built-in optimization |
| **Hosting** | Vercel | Free tier, zero-config deployment, edge functions |
| **Styling** | Tailwind CSS | Rapid development, small bundle size |
| **State** | React Query + Zustand | Server state (React Query) + client state (Zustand) |
| **Forms** | React Hook Form + Zod | Type-safe validation, great DX |
| **File Upload** | react-dropzone | Drag-and-drop, multi-file support |
| **Charts** | Recharts | Lightweight, customizable |

**Cost:** $0/month (Vercel free tier covers MVP)

### Backend

| Component | Technology | Reasoning |
|-----------|------------|-----------|
| **Runtime** | Node.js 22 | Best ecosystem for rapid iteration |
| **Framework** | Hono.js | Fastest Node.js framework, great for APIs |
| **Deployment** | Railway / Fly.io | $5-10/month, better for background jobs than Vercel |
| **ORM** | Drizzle ORM | Type-safe, minimal overhead, SQL-first |
| **Queue** | BullMQ + Upstash Redis | Reliable background jobs, free tier available |
| **Validation** | Zod | Shared with frontend, runtime type safety |

**Cost:** $5-20/month (scales with usage)

### Data & Storage

| Component | Technology | Reasoning |
|-----------|------------|-----------|
| **Database** | Supabase PostgreSQL | Free 500MB, auto-scaling, built-in auth |
| **Cache** | Upstash Redis | Free 10K commands/day, global replication |
| **File Storage** | Cloudflare R2 | Free 10GB, S3-compatible, no egress fees |
| **Search** | PostgreSQL Full-Text | Good enough for MVP, no extra service |

**Cost:** $0-10/month (free tiers cover MVP)

### AI & Processing

| Component | Technology | Reasoning |
|-----------|------------|-----------|
| **Primary AI** | Claude Sonnet 4 | Best balance of speed, cost, accuracy |
| **Fast Path** | Claude Haiku | 10x cheaper for simple resumes |
| **PDF Parsing** | pdf-parse (npm) | Free, reliable, works on Node.js |
| **DOCX Parsing** | mammoth.js | Free, converts to markdown/HTML |
| **OCR (backup)** | Tesseract.js | Free, runs in browser or Node.js |

**Cost:** $0.01-0.05/resume (varies by complexity)

### Infrastructure & DevOps

| Component | Technology | Reasoning |
|-----------|------------|-----------|
| **Hosting** | Railway | Simple, affordable, good DX |
| **DNS** | Cloudflare | Free, fast, great security |
| **Monitoring** | Sentry | Free tier, error tracking |
| **Analytics** | PostHog | Free self-hosted, product analytics |
| **Logs** | Better Stack (Logtail) | Free tier, great search |
| **CI/CD** | GitHub Actions | Free for public repos, simple |

**Cost:** $0-20/month

### Development Tools

| Component | Technology |
|-----------|------------|
| **Language** | TypeScript |
| **Package Manager** | pnpm |
| **Linting** | ESLint + Prettier |
| **Testing** | Vitest + Playwright |
| **API Docs** | Scalar (OpenAPI) |
| **Version Control** | Git + GitHub |

## Database Schema

### Core Tables

```sql
-- Users & Authentication
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email VARCHAR(255) UNIQUE NOT NULL,
  name VARCHAR(255),
  company_name VARCHAR(255),
  industry VARCHAR(100),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- API Keys
CREATE TABLE api_keys (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  key_hash VARCHAR(255) UNIQUE NOT NULL,  -- bcrypt hash
  key_prefix VARCHAR(20) NOT NULL,        -- First 8 chars for display
  name VARCHAR(100),                       -- User-defined name
  last_used_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  revoked_at TIMESTAMPTZ
);

-- Subscriptions
CREATE TABLE subscriptions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  stripe_customer_id VARCHAR(255),
  stripe_subscription_id VARCHAR(255),
  plan VARCHAR(50) NOT NULL,  -- 'free', 'starter', 'professional', 'enterprise'
  status VARCHAR(50) NOT NULL, -- 'active', 'canceled', 'past_due'
  current_period_start TIMESTAMPTZ,
  current_period_end TIMESTAMPTZ,
  cancel_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Resume Screenings (core data)
CREATE TABLE screenings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  api_key_id UUID REFERENCES api_keys(id),

  -- Input
  resume_filename VARCHAR(500),
  resume_file_url TEXT,  -- S3/R2 URL
  job_description TEXT,
  job_title VARCHAR(255),
  industry VARCHAR(100),

  -- Parsed Resume Data (JSONB for flexibility)
  parsed_data JSONB,

  -- Screening Results
  match_score INTEGER,  -- 0-100
  recommendation VARCHAR(20),  -- 'strong_yes', 'yes', 'maybe', 'no'
  summary TEXT,
  matched_requirements JSONB,
  missing_requirements JSONB,
  strengths JSONB,
  concerns JSONB,
  interview_questions JSONB,

  -- Metadata
  processing_time_ms INTEGER,
  ai_model_used VARCHAR(50),  -- 'claude-sonnet-4', 'claude-haiku'
  confidence_score DECIMAL(3,2),  -- 0.00-1.00

  -- User Feedback
  user_rating INTEGER,  -- 1-5 stars
  user_feedback TEXT,

  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Batch Jobs
CREATE TABLE batch_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  job_description TEXT NOT NULL,
  industry VARCHAR(100),
  total_resumes INTEGER DEFAULT 0,
  completed_resumes INTEGER DEFAULT 0,
  failed_resumes INTEGER DEFAULT 0,
  status VARCHAR(50) DEFAULT 'processing',  -- 'processing', 'completed', 'failed'
  results_file_url TEXT,  -- CSV download URL
  webhook_url TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  completed_at TIMESTAMPTZ
);

-- Usage Tracking
CREATE TABLE usage_logs (
  id BIGSERIAL PRIMARY KEY,
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  api_key_id UUID REFERENCES api_keys(id),
  endpoint VARCHAR(255),
  credits_used INTEGER DEFAULT 1,  -- 1 credit = 1 resume screened
  response_time_ms INTEGER,
  status_code INTEGER,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_screenings_user_id ON screenings(user_id);
CREATE INDEX idx_screenings_created_at ON screenings(created_at DESC);
CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX idx_usage_logs_user_id_created ON usage_logs(user_id, created_at DESC);
CREATE INDEX idx_batch_jobs_user_status ON batch_jobs(user_id, status);
```

## API Design

### Authentication

**API Key Format:** `rr_live_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6`
- `rr_` = ResumeRank prefix
- `live_` / `test_` = environment
- 32 random characters (hex)

**Header:**
```
Authorization: Bearer rr_live_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
```

### Endpoints

#### 1. Parse Resume

```http
POST /api/v1/resumes/parse
Content-Type: multipart/form-data

file: <binary>
```

**Response:**
```json
{
  "id": "screening_abc123",
  "parsed_data": { /* full parsed resume */ },
  "confidence": 0.94,
  "processing_time_ms": 1247
}
```

#### 2. Screen Resume

```http
POST /api/v1/resumes/screen
Content-Type: application/json

{
  "resume": { /* parsed resume or raw text */ },
  "job_description": "We're seeking...",
  "industry": "healthcare"
}
```

**Response:**
```json
{
  "id": "screening_abc123",
  "match_score": 87,
  "recommendation": "strong_yes",
  "summary": "Strong candidate with...",
  "matched_requirements": [...],
  "missing_requirements": [...],
  "strengths": [...],
  "concerns": [...],
  "interview_questions": [...],
  "processing_time_ms": 2891
}
```

#### 3. Batch Upload

```http
POST /api/v1/batch
Content-Type: multipart/form-data

files[]: <binary>[]
job_description: <text>
industry: <string>
webhook_url: <url>  (optional)
```

**Response:**
```json
{
  "batch_id": "batch_xyz789",
  "total_resumes": 47,
  "status": "processing",
  "estimated_completion": "2025-11-22T15:30:00Z",
  "status_url": "/api/v1/batch/batch_xyz789/status"
}
```

#### 4. Get Batch Status

```http
GET /api/v1/batch/{batch_id}/status
```

**Response:**
```json
{
  "batch_id": "batch_xyz789",
  "status": "processing",
  "progress": {
    "total": 47,
    "completed": 32,
    "failed": 1,
    "percentage": 68
  },
  "results_url": null,  // Available when status = 'completed'
  "estimated_completion": "2025-11-22T15:30:00Z"
}
```

## System Design Decisions

### 1. Monolith vs Microservices

**Decision:** Start with modular monolith

**Reasoning:**
- Faster development (no inter-service communication complexity)
- Easier debugging and monitoring
- Lower infrastructure costs
- Can extract services later if needed

**Structure:**
```
src/
├── api/           # API routes
├── services/      # Business logic
│   ├── parser/
│   ├── screening/
│   └── batch/
├── models/        # Database models
├── lib/           # Shared utilities
└── workers/       # Background jobs
```

### 2. Synchronous vs Asynchronous Processing

**Decision:** Hybrid approach

- **Single resume:** Synchronous (wait for result)
  - User needs immediate feedback
  - 3-second response is acceptable

- **Batch (10+ resumes):** Asynchronous (queue + webhook)
  - Avoid HTTP timeouts
  - Better resource utilization
  - Can retry failures

### 3. Caching Strategy

**What to cache:**
1. **Job description embeddings** (7 days TTL)
   - Same JD used repeatedly for hiring campaigns
   - Cost savings: 40% reduction in API calls

2. **Parsed resume structure** (24 hours TTL)
   - If user re-screens same resume with different JD
   - Cost savings: Avoid duplicate parsing

3. **API key lookups** (1 hour TTL)
   - High-frequency operation
   - Reduces DB load

**What NOT to cache:**
- Screening results (must be fresh)
- User data (privacy)

### 4. File Storage Strategy

**Upload Flow:**
1. User uploads resume
2. Generate unique filename (UUID + extension)
3. Stream to Cloudflare R2
4. Store R2 URL in database
5. Process resume from R2
6. Delete file after 30 days (GDPR compliance)

**Why R2 over S3:**
- $0 egress fees (S3 charges $0.09/GB)
- S3-compatible API (easy migration if needed)
- Free tier: 10GB storage

### 5. Rate Limiting

**Tiers:**
- Free: 10 requests/minute, 100/month total
- Starter: 30 requests/minute, 250/month total
- Professional: 60 requests/minute, 750/month total
- Enterprise: Custom

**Implementation:**
- Redis-based sliding window
- Per-API-key tracking
- Returns `429 Too Many Requests` with `Retry-After` header

**Why rate limiting:**
- Prevent abuse
- Ensure fair usage
- Protect infrastructure

## AI Architecture

### Prompt Engineering Strategy

**Job Description Analysis Prompt:**
```
You are an expert recruiter analyzing a job description.

Extract:
1. Required qualifications (must-haves)
2. Preferred qualifications (nice-to-haves)
3. Years of experience required
4. Key skills (technical and soft)
5. Required certifications/licenses
6. Industry domain knowledge needed

Format as JSON.
```

**Resume Screening Prompt:**
```
You are an expert recruiter screening a candidate for a {industry} role.

Job Requirements:
{requirements}

Candidate Resume:
{resume}

Analyze:
1. Match score (0-100) based on requirements
2. Which requirements are met (with evidence from resume)
3. Which requirements are missing
4. Unique strengths not in requirements
5. Any concerns or red flags
6. Recommendation: strong_yes | yes | maybe | no
7. 2-3 sentence summary explaining your reasoning
8. 3 interview questions to ask this candidate

Be objective and specific. Cite resume sections.
Format as JSON.
```

### Model Selection Logic

```typescript
function selectAIModel(resume: ParsedResume, jobDescription: string): AIModel {
  // Use fast/cheap model if:
  if (
    resume.confidence > 0.90 &&           // High parsing confidence
    resume.format === 'standard' &&       // Standard resume format
    jobDescription.length < 2000 &&       // Short job description
    !jobDescription.includes('senior') && // Not senior role
    !jobDescription.includes('lead')
  ) {
    return 'claude-haiku';  // $0.01/resume
  }

  // Use powerful model for complex cases
  return 'claude-sonnet-4';  // $0.05/resume
}
```

### Cost Optimization

**Estimated Costs:**
| Scenario | Model | Cost/Resume | Resumes/Month | Total Cost |
|----------|-------|-------------|---------------|------------|
| **Early (80% Haiku)** | Mixed | $0.018 | 1,000 | $18 |
| **Growth (60% Haiku)** | Mixed | $0.026 | 10,000 | $260 |
| **Scale (40% Haiku)** | Mixed | $0.034 | 50,000 | $1,700 |

**Pricing Buffer:**
- Charge $0.75/resume
- Actual cost: $0.018-0.034
- Gross margin: 95-98%
- Buffer covers: infrastructure, support, R&D

## Security & Compliance

### Authentication & Authorization

1. **API Keys:**
   - Stored as bcrypt hashes (never plaintext)
   - Rate limited per key
   - Can be revoked instantly
   - Scope: read-only vs read-write (future)

2. **Web UI:**
   - Supabase Auth (email/password + magic links)
   - JWT session tokens
   - HttpOnly cookies
   - CSRF protection

### Data Privacy

1. **Resume Data:**
   - Encrypted at rest (R2 default encryption)
   - TLS in transit
   - Auto-delete after 30 days
   - User can delete immediately

2. **GDPR Compliance:**
   - Data export API
   - Right to deletion
   - Privacy policy
   - Cookie consent (if needed)

3. **PII Handling:**
   - Never log resume content
   - Mask emails/phones in logs
   - Aggregate analytics only

### Security Best Practices

- [x] HTTPS only (enforce redirects)
- [x] Content Security Policy headers
- [x] Rate limiting
- [x] Input validation (Zod)
- [x] SQL injection prevention (ORM)
- [x] XSS prevention (React auto-escaping)
- [x] Dependency scanning (Dependabot)
- [x] Secrets in environment variables (never in code)

## Monitoring & Observability

### Key Metrics

**Performance:**
- API response time (p50, p95, p99)
- Resume parsing time
- AI screening time
- Database query time

**Reliability:**
- Error rate by endpoint
- Failed resume parsings
- AI API failures
- Batch job completion rate

**Business:**
- Resumes processed per day
- API calls per customer
- Top-used features
- Conversion funnel (signup → first screen)

### Alerting

**Critical Alerts (PagerDuty / email):**
- Error rate > 5%
- API response time p95 > 10s
- Database connection failures
- Stripe payment webhook failures

**Warning Alerts (Slack):**
- Error rate > 1%
- AI costs > $50/day (unexpected spike)
- Disk usage > 80%
- Free tier abuse (same IP, many signups)

## Scalability Plan

### Phase 1: 0-50 Customers (Months 1-3)

**Infrastructure:**
- Single Railway instance ($10/month)
- Supabase free tier
- Cloudflare R2 free tier
- Upstash Redis free tier

**Capacity:**
- ~5,000 resumes/month
- ~100 concurrent users
- 99% uptime (acceptable for early stage)

### Phase 2: 50-200 Customers (Months 4-9)

**Infrastructure:**
- Railway scale to 2 instances ($50/month)
- Supabase Pro ($25/month)
- Upstash Redis paid tier ($10/month)

**Capacity:**
- ~30,000 resumes/month
- ~500 concurrent users
- 99.5% uptime

**Optimizations:**
- Add database read replicas
- Implement CDN caching
- Optimize AI prompts (reduce tokens)

### Phase 3: 200+ Customers (Months 10+)

**Infrastructure:**
- Multiple regions (US-East, US-West)
- Dedicated database instances
- Redis cluster
- Load balancer

**Capacity:**
- ~150,000 resumes/month
- ~2,000 concurrent users
- 99.9% uptime SLA

**Optimizations:**
- Batch AI requests
- Pre-warm connections
- Query optimization
- Index tuning

## Deployment Strategy

### Environments

1. **Development** (local)
   - Docker Compose for services
   - Supabase local instance
   - Test API keys

2. **Staging** (Railway)
   - Mirror of production
   - Test data
   - Pre-release testing

3. **Production** (Railway)
   - Live customer data
   - Monitoring enabled
   - Automated backups

### CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  test:
    - Lint (ESLint)
    - Type check (TypeScript)
    - Unit tests (Vitest)
    - Integration tests (Playwright)

  deploy:
    - Build Docker image
    - Push to Railway
    - Run migrations
    - Smoke tests
    - Rollback if failures
```

### Database Migrations

**Tool:** Drizzle Kit

**Process:**
1. Write migration in `drizzle/migrations/`
2. Test locally
3. Apply to staging
4. Verify staging
5. Apply to production (automated, zero-downtime)

**Rollback:** Keep last 5 migrations ready to rollback

## Disaster Recovery

### Backups

**Database:**
- Automated daily backups (Supabase)
- Point-in-time recovery (7 days)
- Manual snapshots before major changes

**File Storage:**
- Cloudflare R2 auto-replication
- No backups needed (resumes deleted after 30 days)

**Code:**
- Git (GitHub)
- Protected main branch
- Tag releases

### Recovery Time Objective (RTO)

- **Database failure:** <1 hour (restore from backup)
- **API failure:** <15 minutes (redeploy)
- **Data corruption:** <4 hours (manual recovery)

### Recovery Point Objective (RPO)

- **Database:** <24 hours (daily backups)
- **Files:** <1 hour (R2 replication)

---

**Version:** 1.0
**Last Updated:** November 22, 2025
**Owner:** Engineering Team
