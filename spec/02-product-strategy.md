# Product Strategy

## Product Vision

**Mission Statement:**
"Empower every staffing agency to screen candidates 10x faster with AI that understands their industry."

**3-Year Vision:**
By 2028, ResumeRank AI will be the default resume screening tool for 5,000+ staffing agencies, processing 10M+ resumes annually and helping agencies make better, faster hiring decisions.

## Product Positioning

### Value Proposition

**For staffing agencies** that waste 15-20 hours/week manually reviewing resumes,
**ResumeRank AI** is an AI-powered screening API
**that** automatically ranks candidates in under 3 seconds with industry-specific intelligence.
**Unlike** Workable or HireVue which require expensive full-platform adoption,
**our product** works standalone, integrates with any system, and costs 10x less.

### Product Principles

1. **Speed Over Perfection** - 3-second results beat 95% accuracy is better than 98% accuracy in 30 seconds
2. **Transparency** - Every score includes an explanation
3. **Human-in-Loop** - We assist, not replace, recruiters
4. **Vertical Focus** - Industry-specific models beat generic AI
5. **Developer Love** - API-first, excellent docs, fair pricing

## Core Product Features (MVP - Weeks 1-2)

### 1. Resume Parser API

**Endpoint:** `POST /api/v1/resumes/parse`

**Functionality:**
- Accept PDF, DOCX, TXT files up to 5MB
- Extract structured data:
  - Contact info (name, email, phone, location, LinkedIn)
  - Work experience (company, title, dates, responsibilities)
  - Education (degree, institution, year, GPA if present)
  - Skills (technical & soft skills)
  - Certifications & licenses
- Response time: <2 seconds
- Accuracy target: 90%+ for standard resume formats

**Input:**
```json
{
  "file": "base64_encoded_resume_content",
  "filename": "john_doe_resume.pdf"
}
```

**Output:**
```json
{
  "candidate": {
    "name": "John Doe",
    "email": "john.doe@example.com",
    "phone": "+1-555-123-4567",
    "location": "San Francisco, CA",
    "linkedin": "linkedin.com/in/johndoe"
  },
  "experience": [
    {
      "company": "Tech Corp",
      "title": "Senior Software Engineer",
      "start_date": "2020-03",
      "end_date": "present",
      "duration_months": 44,
      "highlights": [
        "Led team of 5 engineers",
        "Built microservices architecture"
      ]
    }
  ],
  "education": [...],
  "skills": ["Python", "React", "AWS", "Leadership"],
  "certifications": ["AWS Solutions Architect"],
  "total_years_experience": 8.5,
  "confidence_score": 0.94
}
```

### 2. AI Matching & Scoring

**Endpoint:** `POST /api/v1/resumes/screen`

**Functionality:**
- Accept resume (parsed or raw) + job description
- Generate 0-100 match score
- Identify matched requirements
- Identify missing requirements
- Highlight unique strengths
- Flag potential concerns
- Generate 2-3 sentence summary
- Response time: <3 seconds

**Input:**
```json
{
  "resume": { /* parsed resume object */ },
  "job_description": "We're seeking a Senior Nurse with 5+ years ICU experience, BLS and ACLS certified...",
  "industry": "healthcare"  // Optional: uses specialized model
}
```

**Output:**
```json
{
  "match_score": 87,
  "recommendation": "strong_yes",  // strong_yes | yes | maybe | no
  "summary": "Strong candidate with 7 years ICU nursing experience and all required certifications. Currently working at a Level 1 trauma center, exceeding the 5-year requirement. Leadership experience as charge nurse is a bonus.",
  "matched_requirements": [
    { "requirement": "5+ years ICU experience", "evidence": "7 years at UCSF Medical Center ICU" },
    { "requirement": "BLS certified", "evidence": "BLS certification valid through 2026" },
    { "requirement": "ACLS certified", "evidence": "ACLS certification valid through 2025" }
  ],
  "missing_requirements": [],
  "strengths": [
    "Charge nurse experience (leadership)",
    "Level 1 trauma center experience (high-acuity)",
    "PALS certified (pediatric advanced life support)"
  ],
  "concerns": [
    "Currently employed - may require notice period"
  ],
  "interview_questions": [
    "Can you describe your experience managing complex trauma cases in the ICU?",
    "What's your notice period at your current position?",
    "Tell me about a time you led a code blue as charge nurse."
  ]
}
```

### 3. Batch Processing Interface

**Page:** `/batch-upload`

**Functionality:**
- Drag-and-drop ZIP file with multiple resumes
- Or select multiple PDFs/DOCX files
- Link to job description (URL or paste text)
- Process up to 100 resumes in parallel
- Display progress bar
- Download results as CSV/Excel

**User Flow:**
1. User uploads ZIP with 50 resumes
2. Pastes job description
3. Selects industry (healthcare)
4. Clicks "Screen Candidates"
5. Progress: "Processing... 23/50 complete"
6. Results table shows ranked candidates
7. Export to Excel for ATS upload

**Output CSV Columns:**
- Candidate Name
- Email
- Phone
- Match Score
- Recommendation
- Years Experience
- Key Skills
- Summary
- Missing Requirements
- Resume Filename

### 4. API Dashboard

**Page:** `/dashboard`

**Functionality:**
- API key management (create, revoke, rotate)
- Usage statistics
  - Resumes processed this month
  - Remaining credits
  - Average match scores
  - Most screened roles
- Billing & invoices
- Integration docs

### 5. Authentication & Rate Limiting

**Free Tier:**
- 100 resumes/month
- Standard API access
- Email support
- Rate limit: 10 requests/minute

**Starter Plan ($149/month):**
- 250 resumes/month
- Standard API access
- Priority email support
- Rate limit: 30 requests/minute
- Overage: $0.75/resume

**Professional Plan ($299/month):**
- 750 resumes/month
- Standard + batch API
- Phone & email support
- Rate limit: 60 requests/minute
- Overage: $0.60/resume
- Zapier integration

## Product Roadmap

### Phase 1: MVP (Weeks 1-2) - COMPLETED BY DAY 14

**Goal:** Prove core value proposition works

- ✅ Resume parser (PDF, DOCX)
- ✅ AI screening with scoring
- ✅ Batch upload interface
- ✅ API authentication
- ✅ Basic dashboard
- ✅ Stripe payment integration
- ✅ Documentation site

**Success Criteria:**
- 3 design partners can screen 50 resumes end-to-end
- <3 second average response time
- 85%+ parsing accuracy
- NPS > 30 from design partners

### Phase 2: Beta Launch (Weeks 3-4) - COMPLETED BY DAY 30

**Goal:** First 10 paying customers

- ✅ Industry-specific models (healthcare, IT, warehouse)
- ✅ Improved UI/UX based on design partner feedback
- ✅ Email summaries (daily digest of screenings)
- ✅ CSV export improvements
- ✅ API SDKs (Python, JavaScript)
- ✅ Webhook support (notify when batch complete)

**Success Criteria:**
- 10 paying customers
- $1,500 MRR
- <5% churn
- 90%+ parsing accuracy

### Phase 3: Scale (Weeks 5-8) - COMPLETED BY DAY 60

**Goal:** $2,500+ MRR, product-market fit signals

- ✅ ATS integrations (Bullhorn, JobAdder)
- ✅ Zapier app
- ✅ Advanced filters (location, salary range, years exp)
- ✅ Candidate deduplication
- ✅ Chrome extension (screen from LinkedIn/Indeed)
- ✅ Team accounts (multi-user access)

**Success Criteria:**
- 15+ paying customers
- $2,500+ MRR
- 1+ customer on Professional plan
- Case study with ROI metrics

### Phase 4: Growth (Months 3-6)

**Goal:** $10,000 MRR, repeatable sales motion

**Features:**
- Custom AI model training (upload past hires)
- Bulk API (1000+ resumes at once)
- Compliance exports (EEOC-ready reports)
- Skills gap analysis
- Interview scheduling integration
- Mobile app (iOS/Android)

**Success Criteria:**
- 50+ paying customers
- $10,000+ MRR
- 1+ enterprise customer
- <5% monthly churn

### Phase 5: Enterprise (Months 7-12)

**Goal:** $35,000 MRR, enterprise-ready

**Features:**
- SSO (SAML, OAuth)
- SOC 2 Type II compliance
- On-premise deployment option
- Custom SLAs
- Dedicated account managers
- White-label API

**Success Criteria:**
- 150+ paying customers
- $35,000+ MRR
- 5+ enterprise customers
- Profitable

## Differentiation Strategy

### 1. Industry-Specific AI Models

**Problem:** Generic AI screening tools miss industry nuances

**Our Solution:**
- Healthcare model knows ICU > med-surg for critical roles
- IT model understands React + TypeScript > jQuery
- Warehouse model prioritizes forklift certification + safety record

**Implementation:**
- Train separate Claude prompts with industry examples
- Use industry-specific terminology in matching logic
- Provide industry templates for job descriptions

**Competitive Moat:** Incumbents use one-size-fits-all models

### 2. Transparent, Usage-Based Pricing

**Problem:** Competitors hide costs behind "custom pricing"

**Our Solution:**
- Public pricing on website
- Calculator showing exact costs
- No sales calls required for Starter/Professional
- Annual plans = 2 months free

**Competitive Moat:** Fastest time-to-value in market

### 3. API-First Architecture

**Problem:** Existing tools require full ATS migration

**Our Solution:**
- Works with any system via API
- Or use our web interface
- Zapier for no-code users
- Webhooks for real-time updates

**Competitive Moat:** Zero switching costs

### 4. Speed & Developer Experience

**Problem:** Slow processing, poor documentation

**Our Solution:**
- <3 second response times
- Interactive API docs (try in browser)
- SDKs in 3 languages
- Postman collection
- Video tutorials

**Competitive Moat:** Developer love creates word-of-mouth

## Technical Differentiation

### AI Architecture Advantages

**Hybrid Approach:**
1. **Fast Path** (80% of resumes): Rule-based extraction + Claude Haiku
   - Cost: $0.01/resume
   - Speed: 1-2 seconds
   - Accuracy: 88-92%

2. **Deep Path** (20% of resumes): Claude Sonnet with vision
   - Cost: $0.05/resume
   - Speed: 3-5 seconds
   - Accuracy: 95-98%

**Trigger for Deep Path:**
- Low confidence score on fast path
- Non-standard resume format
- Missing key fields
- User requests higher accuracy

### Caching Strategy

**Problem:** Identical job descriptions screened repeatedly

**Solution:**
- Cache job description embeddings
- Reuse requirement extraction
- Only re-run candidate matching

**Impact:** 40% cost reduction for agencies screening same role

### Continuous Learning

**Feedback Loop:**
1. User rates screening (thumbs up/down)
2. User corrects match score or flags
3. System learns which signals matter
4. Industry models improve over time

**Impact:** 2-3% accuracy improvement per month

## User Experience Principles

### 1. Don't Make Users Think

- Upload resume → Get results (no configuration)
- Default settings work for 90% of use cases
- Advanced options hidden behind "Show more"

### 2. Speed Matters

- Show progress indicators for batch jobs
- Partial results as they complete
- No forced waiting screens

### 3. Build Trust Through Transparency

- Always show why we gave a score
- Link to specific resume sections
- "Low confidence" warnings when appropriate
- Never hide limitations

### 4. Error Recovery

- PDF parsing failed? Extract text and try again
- Missing email? Still show contact info we found
- API error? Retry automatically with exponential backoff

## Success Metrics (Product)

### North Star Metric
**Resumes Screened Per Customer Per Month**

- Target Month 1: 50/customer
- Target Month 3: 150/customer
- Target Month 6: 300/customer

Higher usage = more value delivered = lower churn

### Product KPIs

| Metric | Target | Why It Matters |
|--------|--------|----------------|
| **API Response Time (p95)** | <3 seconds | Speed is core value prop |
| **Parsing Accuracy** | >90% | Bad data = bad matches |
| **Match Score Accuracy** | >85% thumbs up | Users trust the scores |
| **Batch Processing Success** | >98% | Failures kill workflow |
| **Daily Active Users** | 40% of customers | Sticky product |
| **NPS** | >40 | Word-of-mouth growth |
| **Feature Adoption** | 60% use batch | Power users stay |

### Product-Led Growth Metrics

| Metric | Month 1 | Month 3 | Month 6 |
|--------|---------|---------|---------|
| **Free-to-Paid Conversion** | 10% | 20% | 25% |
| **Free Trial Signups** | 25 | 75 | 150 |
| **Viral Coefficient** | 0.1 | 0.3 | 0.5 |

**Viral Coefficient Calculation:**
- Each customer refers 0.5 new customers on average (referral program + word-of-mouth)
- Target: >1.0 by Month 12 (self-sustaining growth)

## Risk Mitigation

### Technical Risks

| Risk | Mitigation |
|------|------------|
| **AI accuracy below expectations** | Design partners test extensively, set 85% threshold before launch |
| **API costs exceed pricing** | Build cost buffers (2x actual cost), monitoring alerts |
| **Resume format variations break parser** | Test with 500 real resumes from different sources |
| **Slow response times at scale** | Load testing with 100 concurrent requests, caching strategy |

### Product Risks

| Risk | Mitigation |
|------|------------|
| **Features don't match user needs** | 20 customer interviews, design partners co-create |
| **Too complex for target users** | Usability testing with 5 non-technical recruiters |
| **Competitors copy quickly** | Move fast, build moat with industry models + data |
| **Privacy/compliance concerns** | GDPR-ready from day 1, data retention policies, SOC 2 roadmap |

---

## Appendix: User Stories

### Agency Owner (Primary Persona)

**As a** staffing agency owner,
**I want to** screen 100 resumes in 5 minutes instead of 10 hours,
**So that** my recruiters can spend time talking to candidates instead of reading PDFs.

**As a** staffing agency owner,
**I want to** see which candidates best match the job requirements,
**So that** I can present only the top 10 to my client and win the placement.

### Recruiter (Secondary Persona)

**As a** recruiter at a staffing agency,
**I want to** know why a candidate scored 87 vs 92,
**So that** I can confidently explain to my manager why I'm moving forward with them.

**As a** recruiter,
**I want to** get interview questions generated automatically,
**So that** I can quickly prep for screening calls without research.

### Developer (API User)

**As a** developer building an ATS,
**I want to** integrate resume screening via API,
**So that** my users get AI-powered matching without me building the AI.

**As a** developer,
**I want** clear, RESTful API with good documentation,
**So that** I can integrate in 1 day instead of 1 week.

---

**Version:** 1.0
**Last Updated:** November 22, 2025
**Owner:** Product Team
