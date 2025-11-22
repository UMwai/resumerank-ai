# Getting Started with ResumeRank AI

## 📋 Prerequisites

- Node.js 22+ installed
- Anthropic API key (sign up at https://console.anthropic.com)
- PostgreSQL database (Supabase free tier recommended)
- Redis instance (Upstash free tier recommended)

## 🚀 Quick Start (5 minutes)

### 1. Install Dependencies

```bash
npm install
```

### 2. Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:

```bash
# Required for MVP
ANTHROPIC_API_KEY=sk-ant-xxxxx

# Optional for full functionality
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
```

### 3. Run the Development Server

```bash
npm run dev
```

The API will start on http://localhost:3000

### 4. Test the API

```bash
# Health check
curl http://localhost:3000/health

# Should return:
# {"status":"healthy","timestamp":"2025-11-22T...","version":"0.1.0"}
```

## 📚 Next Steps

### Option A: Jump Right In (For Builders)

1. **Test resume parsing:**
   - Create `POST /api/v1/resumes/parse` endpoint
   - Upload a sample PDF resume
   - See structured data extraction in action

2. **Test AI screening:**
   - Create `POST /api/v1/resumes/screen` endpoint
   - Paste a job description
   - Get AI-powered candidate analysis

3. **Build the UI:**
   - Set up Next.js frontend
   - Create upload interface
   - Display screening results

### Option B: Follow the 90-Day Plan (For Entrepreneurs)

1. **Week 1-2: MVP + Customer Discovery**
   - Finish core API endpoints
   - Conduct 10 customer interviews
   - Recruit 3 design partners
   - See: `/spec/04-go-to-market-strategy.md`

2. **Week 3-4: Beta Testing**
   - Onboard design partners
   - Iterate based on feedback
   - Validate pricing
   - Add payment integration

3. **Week 5-6: Public Launch**
   - Product Hunt launch
   - Launch on Reddit, Indie Hackers
   - Convert first paying customers
   - Target: $750 MRR

## 🛠️ Development Workflow

### Running Tests

```bash
npm test
```

### Type Checking

```bash
npm run build  # Compiles TypeScript
```

### Linting

```bash
npm run lint
```

## 📖 Key Files to Understand

| File | Purpose |
|------|---------|
| `src/index.ts` | API server entry point |
| `src/types/index.ts` | TypeScript type definitions |
| `src/services/parser/resume-parser.ts` | PDF/DOCX parsing + AI extraction |
| `src/services/screening/ai-screener.ts` | AI-powered candidate matching |

## 🧪 Testing with Sample Data

### Sample Resume (create `test/sample-resume.txt`)

```
John Doe
Senior Software Engineer
john.doe@example.com | +1-555-123-4567 | San Francisco, CA
linkedin.com/in/johndoe

EXPERIENCE
Tech Corp, Senior Software Engineer (March 2020 - Present)
- Led team of 5 engineers building microservices architecture
- Reduced API response time by 60% through caching optimization
- Mentored 3 junior developers

Startup Inc, Software Engineer (Jan 2017 - Feb 2020)
- Built React frontend for SaaS product
- Implemented CI/CD pipeline with GitHub Actions

EDUCATION
MIT, B.S. Computer Science (2017)
GPA: 3.8/4.0

SKILLS
JavaScript, TypeScript, React, Node.js, Python, AWS, Docker, Kubernetes

CERTIFICATIONS
AWS Solutions Architect - Associate (2022)
```

### Sample Job Description

```
We're seeking a Senior Software Engineer with 5+ years experience building
scalable web applications. Must have:
- Strong JavaScript/TypeScript skills
- Experience with React and Node.js
- Cloud experience (AWS, GCP, or Azure)
- Leadership/mentorship experience

Nice to have:
- Microservices architecture experience
- DevOps skills (Docker, Kubernetes, CI/CD)
- AWS certifications
```

## 💡 Tips for Success

1. **Start Simple:** Get the core parsing + screening working first
2. **Validate Early:** Talk to 10 potential customers before building everything
3. **Iterate Fast:** Ship imperfect features, improve based on feedback
4. **Track Metrics:** CAC, LTV, churn from day 1
5. **Stay Lean:** Use free tiers, avoid premature scaling

## 📊 Business Metrics to Track

Create a simple spreadsheet to track:

| Date | MRR | Customers | CAC | Churn | Resumes Processed |
|------|-----|-----------|-----|-------|-------------------|
| Week 1 | $0 | 0 | - | - | 0 |
| Week 2 | $0 | 3 design partners | $300 | 0% | 150 |
| ... | ... | ... | ... | ... | ... |

## 🚨 Common Issues

### "Module not found" errors
```bash
rm -rf node_modules package-lock.json
npm install
```

### Anthropic API errors
- Check your API key is correct in `.env`
- Verify you have credits: https://console.anthropic.com
- Check rate limits (free tier: 5 requests/min)

### TypeScript errors
```bash
npx tsc --noEmit  # Check for type errors
```

## 📞 Need Help?

- **Documentation:** See `/spec` folder for comprehensive guides
- **Architecture:** Read `/spec/03-technical-architecture.md`
- **Business Plan:** Read `/spec/STARTUP-SUMMARY.md`

## 🎯 Your First Week Goals

- [ ] API running locally
- [ ] Can parse a PDF resume
- [ ] Can screen a resume against a job description
- [ ] Talked to 5 potential customers
- [ ] Identified 1-2 design partners

---

**Remember:** The goal is $2,500 MRR in 90 days. Focus on customers first, perfect code second.

Let's build! 🚀
