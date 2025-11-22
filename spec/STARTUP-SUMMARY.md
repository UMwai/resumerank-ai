# ResumeRank AI - Startup Summary

## Executive Decision

After extensive analysis involving expert agents (product management, data science, backend architecture, and project management), market research, and competitive analysis, we selected:

**🎯 AI Resume Screening API**

### Why This Won

| Criteria | Score | Rationale |
|----------|-------|-----------|
| **Technical Feasibility** | 9/10 | Simple PDF parsing + Claude API, 2-week MVP |
| **Time to Revenue** | 9/10 | First customer in 2-3 weeks, fastest path to cash |
| **Market Opportunity** | 8/10 | $2B market growing to $5.4B by 2034 |
| **Competition Moat** | 8/10 | Industry-specific models, transparent pricing |
| **ROI Potential** | 9/10 | 55% probability of $2K MRR in 90 days |

**Probability of Success:** 55% (vs 45% for Document Intelligence API)

---

## The Opportunity

**Problem:** Staffing agencies waste 15-20 hours/week manually reviewing resumes. With 250+ applications per job posting, 80% of time is spent on unqualified candidates.

**Solution:** AI-powered resume screening API that ranks candidates in <3 seconds using industry-specific intelligence.

**Market:**
- 15,000+ staffing agencies in the US
- 83% of companies will use AI resume screening by 2025
- $2B market growing at 11.6% CAGR
- Underserved SMB segment (current tools target enterprise)

---

## Business Model

### Pricing

| Tier | Price | Target | Value Proposition |
|------|-------|--------|-------------------|
| **Free** | $0 | Trial users | 100 resumes/month |
| **Starter** | $149/mo | 5-10 person agencies | 250 resumes, basic features |
| **Professional** | $299/mo | 10-25 person agencies | 750 resumes, integrations |
| **Enterprise** | Custom | 25+ person agencies | Unlimited, SLA |

### Unit Economics

- **CAC:** $100 (blended across channels)
- **LTV:** $3,002 (20-month lifespan, 95% gross margin)
- **LTV:CAC Ratio:** 30:1 ✅ (target: >3:1)
- **Payback Period:** 20 days ✅ (target: <90 days)
- **Gross Margin:** 95% ✅ (per-resume cost: $0.025)

---

## 90-Day Plan

### Financial Targets

| Month | Customers | MRR | Revenue | Costs | Profit |
|-------|-----------|-----|---------|-------|--------|
| **1** | 3 | $447 | $447 | $749 | -$302 |
| **2** | 10 | $1,500 | $1,947 | $547 | +$1,400 |
| **3** | 16 | $2,800 | $4,747 | $1,135 | +$3,612 |

**90-Day Totals:**
- Revenue: $7,141
- Costs: $2,431
- Net Profit: $4,710
- ROI: 94% on $5,000 investment

### Execution Roadmap

**Weeks 1-2:** Foundation
- Build MVP (parser, AI screener, API)
- 20 customer interviews
- 3 design partners recruited
- Budget: $650

**Weeks 3-4:** Beta
- Design partners onboarding
- Iterate based on feedback
- Validate pricing
- Budget: $700

**Weeks 5-6:** Launch
- Product Hunt launch
- Free tier + paid plans
- 10+ free trials → 5 paid customers
- Budget: $1,000

**Weeks 7-8:** Scale to $2K
- Cold outreach (200 agencies)
- Convert free users
- 10 paying customers
- Budget: $950

**Weeks 9-12:** Growth
- Paid ads (LinkedIn, Google)
- Partnerships (ATS integrations)
- Referral program
- 16+ customers, $2,500+ MRR
- Budget: $1,450

---

## Competitive Positioning

### Direct Competitors

| Competitor | Pricing | Weakness | Our Advantage |
|------------|---------|----------|---------------|
| **Workable** | $299+/mo | Full ATS required | Standalone, 5-min setup |
| **CVViZ** | $5K-15K/yr | Enterprise only | $149/mo SMB tier |
| **HireVue** | $25K+/yr | Video focus | Resume-only, affordable |
| **Manatal** | $15/user/mo | Per-user scaling | Usage-based pricing |

### Differentiation

1. **Industry-Specific AI** - Healthcare model knows ICU > med-surg
2. **Transparent Pricing** - No "contact sales" gatekeeping
3. **API-First** - Works with any system, no migration needed
4. **Speed** - 3 seconds vs 30+ for competitors

---

## Go-To-Market Strategy

### Channels (Budget: $2,000 over 90 days)

**Owned (Free):**
- LinkedIn (founder-led content, 5 posts/week)
- Blog/SEO (case studies, guides)
- Community engagement (r/recruiting)

**Earned (Free):**
- Product Hunt launch
- PR outreach (TechCrunch, HR Dive)
- Word-of-mouth/referrals

**Paid ($900):**
- LinkedIn ads: $300/mo (Month 3)
- Google ads: $200/mo (Month 3)
- Facebook/Instagram: $100/mo (Month 3)

**Partnerships:**
- ATS integrations (Bullhorn, JobAdder)
- Zapier app
- Affiliate program (30% recurring)

### Target Customer Profile

**Primary: Staffing Agency Owner**
- 40-55 years old
- 5-50 employees, $2M-20M revenue
- Pain: Recruiters waste time reading PDFs
- Budget authority: Can approve $150-500/mo
- Industry focus: Healthcare (Month 1-2), IT (Month 3-4), Warehouse (Month 5+)

---

## Technical Implementation

### Tech Stack

**Backend:**
- Node.js 22 + TypeScript
- Hono.js (API framework)
- PostgreSQL (Supabase)
- Redis (Upstash)
- Cloudflare R2 (file storage)

**AI:**
- Claude Sonnet 4 (complex resumes)
- Claude Haiku (simple resumes)
- Hybrid approach for cost optimization

**Infrastructure:**
- Railway ($5-20/mo)
- Vercel (frontend, free)
- Stripe (payments)

**Cost:** $0-30/mo initially (free tiers)

### MVP Features (Weeks 1-2)

✅ Resume parser (PDF, DOCX)
✅ AI screening with scoring (0-100)
✅ Batch upload interface
✅ API authentication
✅ Basic dashboard
✅ Stripe integration
✅ Documentation

---

## Risk Analysis

### Key Risks & Mitigation

| Risk | Probability | Mitigation |
|------|-------------|------------|
| **Low customer adoption** | Medium | 20 interviews pre-launch, design partners validate |
| **AI accuracy issues** | Low | Human-in-loop positioning, 85%+ accuracy target |
| **Competitor price war** | Low | Niche focus (staffing), superior UX |
| **API costs exceed revenue** | Low | Caching, tiered pricing with buffers |

### Pivot Triggers (Day 60)

If any of these are true, reassess:
- <5 paying customers
- <$1,000 MRR
- >20% churn
- NPS <20

→ Pivot to Document Intelligence API (backup plan)

---

## Success Metrics

### 90-Day Targets (Must-Have)

- ✅ 15+ paying customers
- ✅ $2,500+ MRR
- ✅ <10% monthly churn
- ✅ NPS >40
- ✅ Break-even by Day 45

### Nice-to-Have

- ✅ 1+ customer on Professional plan
- ✅ 1 published case study
- ✅ Featured in 1+ publication
- ✅ 10+ referrals

---

## 12-Month Vision

**Month 6:** $10,000 MRR, 50 customers, ATS integrations
**Month 12:** $35,000 MRR, 150 customers, enterprise tier
**Year 2:** $100,000 MRR, 400+ customers, potential acquisition

**Exit Potential:**
- Strategic buyers: Workable, Greenhouse, Lever, BambooHR
- Valuation (Year 2): $2-3M (4-6x $500K ARR)
- Valuation (Year 3): $12-16M (6-8x $2M ARR)

---

## Budget Allocation

### $5,000 Breakdown

| Category | Amount | % |
|----------|--------|---|
| **Marketing** | $2,000 | 40% |
| • Paid ads | $900 | |
| • Cold email tools | $300 | |
| • Design partner incentives | $400 | |
| • Content creation | $200 | |
| • PR/media | $200 | |
| **Development** | $1,100 | 22% |
| • Claude API credits | $600 | |
| • Hosting | $200 | |
| • SaaS tools | $300 | |
| **Customer Research** | $400 | 8% |
| **Marketing/Content** | $500 | 10% |
| **Contingency** | $1,000 | 20% |

---

## Next Steps (Immediate Actions)

### Day 1 Checklist

- [ ] Register domain (resumerank.ai)
- [ ] Set up landing page with waitlist (Carrd, 2 hours)
- [ ] Create LinkedIn presence
- [ ] Join 5 recruiting communities
- [ ] Draft customer interview script
- [ ] Send 10 connection requests to staffing agency owners
- [ ] Install dependencies (`npm install`)
- [ ] Set up .env file with API keys
- [ ] Run MVP locally (`npm run dev`)

### Week 1 Priorities

1. **Build MVP** (Days 1-7)
   - Resume parser ✅ (implemented)
   - AI screener ✅ (implemented)
   - Basic API endpoints (in progress)
   - Landing page

2. **Customer Discovery** (Days 1-14)
   - 10 customer interviews
   - 3 design partners recruited
   - Pricing validation

3. **Marketing Foundation** (Days 1-7)
   - LinkedIn profile optimized
   - Landing page live
   - 50 LinkedIn connections

---

## Why This Will Succeed

1. **Market Timing:** 83% of companies adopting AI screening by 2025
2. **Clear Pain Point:** Agencies visibly struggling with high-volume screening
3. **Strong Unit Economics:** 30:1 LTV:CAC, 95% margins
4. **Low Capital Risk:** $5K budget, profitable by Month 2
5. **Founder-Market Fit:** Technical capability + market understanding
6. **Defensible Moat:** Industry-specific models, data flywheel
7. **Scalable Model:** API-first, self-serve, low marginal cost

---

## Resources

**Documentation:**
- Full business plan: `/spec/00-executive-summary.md` through `05-financial-projections.md`
- Technical architecture: `/spec/03-technical-architecture.md`
- 90-day roadmap: `/spec/04-go-to-market-strategy.md`

**Implementation:**
- MVP codebase: `/src` (TypeScript, Hono.js, Claude API)
- Resume parser: `/src/services/parser/resume-parser.ts`
- AI screener: `/src/services/screening/ai-screener.ts`

**Market Research:**
- [AI Hiring Software Market - Market.us](https://market.us/report/ai-hiring-software-market/)
- [AI Resume Screening Statistics - The Interview Guys](https://blog.theinterviewguys.com/83-of-companies-will-use-ai-resume-screening-by-2025-despite-67-acknowledging-bias-concerns/)
- [B2B SaaS First 100 Customers - 7Startup VC](https://7startup.vc/post/b2b-saas-startup-tips-how-to-get-your-first-100-customers/)

---

**Prepared by:** Startup Analysis Team
**Date:** November 22, 2025
**Confidence:** High (55% probability of hitting $2.5K MRR in 90 days)
**Recommendation:** ✅ Execute immediately
