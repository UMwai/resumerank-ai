# 90-Day Execution Roadmap: AI Resume Screening API vs Document Intelligence API

## Executive Summary

| Criteria | Option A: Resume Screening | Option B: Document Intelligence |
|----------|---------------------------|--------------------------------|
| Target Market | Staffing agencies (15,000+ in US) | Bookkeepers/Accountants (300,000+ in US) |
| Price Point | $0.50-2/resume or $149-299/mo | $0.15/invoice |
| Revenue Model | Subscription + usage | Pure usage-based |
| Technical Complexity | Medium-High | Medium |
| Sales Cycle | 2-4 weeks | 1-2 weeks |
| Competition | Crowded (Lever, Greenhouse, HireVue) | Moderate (Dext, Hubdoc) |
| Path to $2K MRR | 7-14 customers @ $149-299 | ~13,333 invoices/month |

---

# OPTION A: AI Resume Screening API

## Product Definition

### Core MVP Features (Weeks 1-4)
1. **Resume Parser** - Extract structured data (name, contact, education, experience, skills)
2. **Job Description Matcher** - Score candidates against job requirements
3. **Skills Extraction** - Identify and categorize hard/soft skills
4. **API Dashboard** - Usage tracking, API key management
5. **Batch Upload** - Process multiple resumes at once

### Pricing Tiers
- **Starter**: $149/month - 500 resumes, basic matching
- **Professional**: $249/month - 2,000 resumes, advanced analytics
- **Enterprise**: $299/month - 5,000 resumes, custom models, priority support
- **Pay-as-you-go**: $0.50/resume (no commitment)

---

## WEEK 1: Foundation & Market Validation
**Dates: Days 1-7**

### Development Tasks (20 hours)
- [ ] Set up FastAPI/Python backend with authentication
- [ ] Implement basic resume parsing (PDF, DOCX support)
- [ ] Create OpenAI/Claude integration for text extraction
- [ ] Build simple API endpoint: POST /api/v1/parse
- [ ] Deploy to Railway/Render with basic monitoring
- [ ] Create Stripe integration for payment processing

### Customer Development (10 hours)
**Target: Talk to 15 staffing agency owners/recruiters**

Where to find them:
- LinkedIn: Search "Owner" + "Staffing Agency" + city names
- Staffing Industry Analysts (SIA) member directory
- Local staffing association meetups
- Upwork: hire 2-3 recruiters for 30-min paid interviews ($50 each)
- Reddit: r/recruiting, r/staffing
- Facebook Groups: "Staffing Agency Owners", "Recruiting Professionals"

Interview script focus:
1. "Walk me through your resume screening process today"
2. "How many resumes do you process per week/month?"
3. "What's your biggest bottleneck in candidate screening?"
4. "What tools do you currently use? What do you pay?"
5. "If I could cut your screening time by 70%, what would that be worth?"

### Marketing Activities (5 hours)
- [ ] Create landing page with waitlist (Carrd or simple Next.js)
- [ ] Write 3 LinkedIn posts about staffing industry pain points
- [ ] Join 5 staffing-focused LinkedIn groups
- [ ] Set up Twitter/X account focused on HR tech

### Metrics to Track
| Metric | Target | How to Track |
|--------|--------|--------------|
| Discovery calls completed | 10 | Notion/spreadsheet |
| Waitlist signups | 25 | Landing page form |
| LinkedIn connections (recruiters) | 50 | LinkedIn |
| Pain point validation | 7/10 confirm problem | Interview notes |

### Success Criteria
- [x] MVP parsing endpoint working
- [x] 10+ customer interviews completed
- [x] Problem-solution fit validated (7+ confirm they'd pay)
- [x] Landing page live with 25+ waitlist signups

### Budget Allocation: $300
- Domain + hosting: $50
- Paid customer interviews: $150
- OpenAI API credits: $50
- LinkedIn Sales Navigator (trial): $0
- Miscellaneous: $50

### Risk Mitigation
- **Risk**: Can't get interviews scheduled
- **Mitigation**: Offer $25 Amazon gift card for 20-min calls; use warm intros from LinkedIn

---

## WEEK 2: Core Product Build
**Dates: Days 8-14**

### Development Tasks (25 hours)
- [ ] Build job description analysis endpoint
- [ ] Implement candidate-job matching algorithm (0-100 score)
- [ ] Create skills taxonomy (500+ skills mapped)
- [ ] Add batch processing capability (up to 50 resumes)
- [ ] Build basic dashboard UI (React/Next.js)
- [ ] Implement usage tracking and rate limiting
- [ ] Write API documentation (Swagger/OpenAPI)

### Customer Development (8 hours)
**Target: 10 more interviews + 3 design partners**

Design Partner Criteria:
- Process 100+ resumes/month
- Willing to test beta for 2 weeks
- Provide weekly feedback calls
- Potential to convert at $149-249/month

Outreach message template:
```
Subject: Quick question about your resume screening process

Hi [Name],

I noticed [Agency Name] specializes in [niche]. I'm building
an AI tool that helps staffing agencies screen resumes 70% faster.

Would you be open to a 15-minute call this week? I'd love to
understand your current process and see if this could help.

As a thank you, I'll share my research on what top agencies
are doing differently in 2024.

Best,
[Your name]
```

### Marketing Activities (5 hours)
- [ ] Publish first blog post: "How Top Staffing Agencies Screen 500 Resumes/Day"
- [ ] Create demo video (Loom, 3 minutes)
- [ ] Cold email 50 staffing agencies from target list
- [ ] Post in 3 staffing Facebook groups
- [ ] Engage daily on LinkedIn (30 min/day)

### Metrics to Track
| Metric | Target | How to Track |
|--------|--------|--------------|
| Total interviews | 20 (cumulative) | Spreadsheet |
| Design partners signed | 3 | Email confirmations |
| Waitlist signups | 75 | Landing page |
| API uptime | 99%+ | Monitoring |

### Success Criteria
- [ ] Matching algorithm achieving 80%+ accuracy on test set
- [ ] 3 design partners committed
- [ ] Demo video created
- [ ] 75+ waitlist signups

### Budget Allocation: $400
- OpenAI/Claude API (higher usage): $150
- Email tool (Instantly.ai): $97
- Loom Pro: $15
- Customer interview incentives: $100
- Hosting upgrade: $38

---

## WEEK 3: Beta Launch with Design Partners
**Dates: Days 15-21**

### Development Tasks (20 hours)
- [ ] Implement feedback from design partner testing
- [ ] Add resume ranking feature (sort by match score)
- [ ] Build simple ATS integration (Bullhorn API) - basic
- [ ] Create onboarding flow for new users
- [ ] Add email notifications (processing complete)
- [ ] Implement error handling and retry logic

### Customer Development (10 hours)
**Target: Onboard 3 design partners, 5 more warm leads**

Design Partner Success Protocol:
- Day 1: 30-min onboarding call, set up account
- Day 3: Check-in call, gather first impressions
- Day 7: Review usage data, pain points
- Day 14: Decision call - convert or learn why not

Questions for beta users:
1. "What worked better than expected?"
2. "What's still frustrating?"
3. "Would you pay $149/month for this? Why/why not?"
4. "Who else should I talk to?"

### Marketing Activities (8 hours)
- [ ] Write case study draft from design partner feedback
- [ ] Publish LinkedIn article: "We Analyzed 10,000 Resumes - Here's What We Found"
- [ ] Start cold LinkedIn outreach (20 connections/day)
- [ ] Guest post pitch to HR/staffing blogs (3 pitches)
- [ ] Create comparison landing page vs. manual screening

### Metrics to Track
| Metric | Target | How to Track |
|--------|--------|--------------|
| Design partner activation | 3/3 actively using | Dashboard |
| Resumes processed (beta) | 200+ | API logs |
| NPS from beta users | 8+ | Survey |
| Waitlist signups | 150 | Landing page |

### Success Criteria
- [ ] All 3 design partners actively using product
- [ ] 200+ resumes processed without critical bugs
- [ ] At least 2/3 design partners indicate willingness to pay
- [ ] First testimonial/quote obtained

### Budget Allocation: $400
- API costs (increased usage): $200
- Bullhorn API developer account: $0 (trial)
- Design partner thank-you gifts: $150
- Monitoring tools: $50

---

## WEEK 4: Pricing Validation & Pre-launch
**Dates: Days 22-28**

### Development Tasks (15 hours)
- [ ] Build subscription management (Stripe Checkout + Portal)
- [ ] Implement usage-based billing tracking
- [ ] Add team features (invite users, shared API key)
- [ ] Create admin dashboard for customer management
- [ ] Performance optimization (target: <2s per resume)
- [ ] Security audit and hardening

### Customer Development (12 hours)
**Target: Convert 2 design partners, validate pricing with 10 prospects**

Pricing Validation Approach:
1. **Van Westendorp Price Sensitivity**:
   - "At what price would this be so cheap you'd question quality?"
   - "At what price is this a bargain?"
   - "At what price is this getting expensive but you'd consider?"
   - "At what price is this too expensive?"

2. **Willingness-to-Pay Test**:
   - Show pricing page mockup
   - Ask: "Which tier would you choose and why?"
   - Test at $149, $199, $249 price points

3. **Competitive Anchoring**:
   - "How much do you spend on your current ATS/screening tools?"
   - "What percentage of that would you allocate to AI screening?"

### Marketing Activities (10 hours)
- [ ] Finalize pricing page with 3 tiers
- [ ] Create launch email sequence (3 emails)
- [ ] Prepare Product Hunt launch assets
- [ ] Write "Why We Built This" founder story
- [ ] Set up affiliate/referral tracking (Rewardful)
- [ ] Create LinkedIn carousel: "5 Signs Your Screening Process is Broken"

### Metrics to Track
| Metric | Target | How to Track |
|--------|--------|--------------|
| Pricing validation calls | 10 | Calendar |
| Pre-launch commitments | 5 | Email/verbal |
| Waitlist signups | 250 | Landing page |
| Design partner conversions | 2/3 | Stripe |

### Success Criteria
- [ ] Pricing validated ($149-249 sweet spot confirmed)
- [ ] 2+ design partners converted to paid ($298-498 MRR)
- [ ] 5+ pre-launch commitments
- [ ] Launch assets ready

### Budget Allocation: $350
- Stripe fees: $30
- Launch prep (design, copy): $150
- API costs: $120
- Rewardful: $29
- Miscellaneous: $21

---

## WEEK 5: Public Launch
**Dates: Days 29-35**

### Development Tasks (10 hours)
- [ ] Final bug fixes from beta feedback
- [ ] Implement analytics tracking (Mixpanel/PostHog)
- [ ] Create in-app onboarding tooltips
- [ ] Build public API status page
- [ ] Add live chat support (Crisp/Intercom)

### Customer Development (8 hours)
**Target: Convert waitlist, onboard first 5 paying customers**

Launch Day Playbook:
1. **Hour 1**: Email waitlist (250+ contacts)
2. **Hour 2**: Post on LinkedIn, Twitter, relevant communities
3. **Hour 4**: Product Hunt launch (if ready)
4. **Hour 6**: Personal outreach to top 20 waitlist contacts
5. **Day 2-3**: Follow-up calls with interested prospects
6. **Day 4-7**: Onboarding calls with new customers

### Marketing Activities (15 hours)
- [ ] Execute launch email sequence
- [ ] Post launch announcement on 10+ communities:
  - Hacker News (Show HN)
  - r/SaaS, r/startups, r/recruiting
  - Indie Hackers
  - LinkedIn (personal + groups)
  - Twitter/X
  - Staffing Facebook groups
  - SIA community forums
- [ ] Reach out to 5 HR tech bloggers/podcasters
- [ ] Start Google Ads test campaign ($10/day)
- [ ] Create customer onboarding video

### Metrics to Track
| Metric | Target | How to Track |
|--------|--------|--------------|
| Signups (free trial) | 30 | Dashboard |
| Paid conversions | 5 | Stripe |
| MRR | $745+ | Stripe |
| Churn | 0 | Stripe |
| Support tickets | <10 | Helpdesk |

### Success Criteria
- [ ] 5+ paying customers
- [ ] $500+ MRR
- [ ] <24 hour support response time
- [ ] Zero critical production issues

### Budget Allocation: $600
- Google Ads: $300
- Product Hunt launch boost: $0 (organic)
- API costs (higher usage): $200
- Support tools: $50
- Miscellaneous: $50

### MONTH 1 TOTAL BUDGET: $2,050

---

## WEEK 6: Growth Optimization
**Dates: Days 36-42**

### Development Tasks (12 hours)
- [ ] Build features requested by paying customers (prioritize)
- [ ] Add Chrome extension for LinkedIn recruiter integration
- [ ] Implement webhook notifications
- [ ] Create resume database/search feature
- [ ] Add custom scoring criteria configuration

### Customer Development (10 hours)
**Target: 5 more conversions, reduce churn risk**

Customer Success Protocol:
- Schedule 15-min check-in with each paying customer
- Ask: "On a scale of 1-10, how likely to recommend?"
- Identify power users vs. at-risk accounts
- Create customer health score

### Marketing Activities (12 hours)
- [ ] Analyze launch data, double down on working channels
- [ ] Write 2 SEO blog posts (long-tail keywords)
- [ ] Create partnership pitch for ATS vendors
- [ ] Launch referral program (give 1 month free, get 1 month free)
- [ ] Guest on 1 staffing industry podcast (pitch 10)

### Metrics to Track
| Metric | Target | How to Track |
|--------|--------|--------------|
| New paid customers | 5 | Stripe |
| Total paying customers | 10 | Stripe |
| MRR | $1,500+ | Stripe |
| Trial-to-paid rate | 25%+ | Dashboard |
| NPS | 40+ | Survey |

### Success Criteria
- [ ] 10+ paying customers
- [ ] $1,200+ MRR
- [ ] Clear product-market fit signals
- [ ] One partnership conversation started

### Budget Allocation: $500
- Google Ads (optimized): $250
- API costs: $150
- Podcast sponsorship: $0
- Content creation: $100

---

## WEEK 7: Scale Customer Acquisition
**Dates: Days 43-49**

### Development Tasks (10 hours)
- [ ] Build Zapier integration (connect to 1000+ apps)
- [ ] Add white-label option for agencies
- [ ] Implement A/B testing for matching algorithm
- [ ] Create embeddable widget for career pages
- [ ] Add resume anonymization feature (bias reduction)

### Customer Development (8 hours)
**Target: Enterprise pilot, expand use cases**

Enterprise Outreach:
- Target: Regional staffing firms (50-200 employees)
- Channel: LinkedIn + warm intros from current customers
- Offer: Free 2-week pilot, dedicated support
- Goal: 1 enterprise deal at $500+/month

### Marketing Activities (15 hours)
- [ ] Launch LinkedIn Ads campaign (target: recruiters)
- [ ] Create ROI calculator tool
- [ ] Publish customer case study
- [ ] Partner content with staffing blog (guest post)
- [ ] Email sequence to non-converted trials

### Metrics to Track
| Metric | Target | How to Track |
|--------|--------|--------------|
| New paid customers | 5 | Stripe |
| Total paying customers | 15 | Stripe |
| MRR | $2,000+ | Stripe |
| Enterprise pilots | 1 | CRM |
| CAC | <$100 | Calculated |

### Success Criteria
- [ ] 15+ paying customers
- [ ] $1,800+ MRR
- [ ] At least 1 customer on $299 tier
- [ ] ROI calculator driving conversions

### Budget Allocation: $550
- LinkedIn Ads: $300
- Google Ads: $150
- API costs: $100

---

## WEEK 8: Retention & Expansion
**Dates: Days 50-56**

### Development Tasks (10 hours)
- [ ] Build annual plan option (20% discount)
- [ ] Add usage alerts (approaching limit)
- [ ] Create team analytics dashboard
- [ ] Implement SSO (Google, Microsoft)
- [ ] Add candidate feedback loop feature

### Customer Development (10 hours)
**Target: Upsells, annual conversions, reduce churn**

Expansion Revenue Tactics:
1. Identify customers approaching usage limits
2. Offer upgrade with 1 month free
3. Push annual plans with 20% discount
4. Cross-sell additional seats

### Marketing Activities (12 hours)
- [ ] Create email nurture for trial users
- [ ] Build SEO content hub (10 articles planned)
- [ ] Launch partner program page
- [ ] Webinar: "AI in Recruiting: What's Working in 2024"
- [ ] Retargeting ads for website visitors

### Metrics to Track
| Metric | Target | How to Track |
|--------|--------|--------------|
| MRR | $2,200+ | Stripe |
| Net Revenue Retention | 100%+ | Calculated |
| Annual plan conversions | 2 | Stripe |
| Webinar registrations | 50 | Landing page |

### Success Criteria
- [ ] 18+ paying customers
- [ ] $2,000+ MRR achieved
- [ ] 2+ annual plan conversions
- [ ] Zero churn in month 2

### Budget Allocation: $450
- Ads (retargeting): $200
- Webinar platform: $50
- API costs: $150
- Content: $50

---

## WEEKS 9-12: Scale to $2K+ MRR
**Dates: Days 57-90**

### Development Priorities
- Advanced analytics and reporting
- Additional ATS integrations (JobAdder, PCRecruiter)
- AI-powered interview question generation
- Candidate pipeline management
- Mobile app (stretch goal)

### Growth Priorities
- Systematic outbound to staffing agencies (100/week)
- SEO content scaling (2 posts/week)
- Partnership with 1-2 ATS vendors
- Industry conference presence (virtual)
- Customer referral program optimization

### Week 9 Target: $2,000 MRR (20 customers @ $100 avg)
### Week 10 Target: $2,400 MRR (growth + upsells)
### Week 11 Target: $2,800 MRR (adding enterprise)
### Week 12 Target: $3,200 MRR (sustainable growth)

### Budget Allocation (Weeks 9-12): $1,450
- Ads: $800
- API costs: $400
- Tools/subscriptions: $150
- Events: $100

---

## OPTION A: TOTAL BUDGET BREAKDOWN

| Category | Amount | % of Budget |
|----------|--------|-------------|
| Development (API, hosting) | $1,100 | 22% |
| Paid Advertising | $2,000 | 40% |
| Customer Interviews/Incentives | $400 | 8% |
| Tools & Subscriptions | $500 | 10% |
| Content & Marketing | $500 | 10% |
| Contingency | $500 | 10% |
| **TOTAL** | **$5,000** | 100% |

---

# OPTION B: Document Intelligence API

## Product Definition

### Core MVP Features (Weeks 1-4)
1. **Invoice Parser** - Extract vendor, amount, date, line items, tax
2. **Receipt OCR** - Capture expense data from receipts
3. **Document Classification** - Auto-categorize document types
4. **QuickBooks Integration** - Push extracted data directly
5. **Batch Processing** - Handle multiple documents

### Pricing Model
- **Pay-as-you-go**: $0.15/invoice (no minimum)
- **Volume discounts**:
  - 1,000+/month: $0.12/invoice
  - 5,000+/month: $0.10/invoice
  - 10,000+/month: $0.08/invoice

### Revenue Math to Hit $2K MRR
- At $0.15/invoice: Need 13,333 invoices/month
- At $0.12/invoice: Need 16,667 invoices/month
- **Alternative**: 10 customers @ 1,000 invoices/month = $1,500 MRR
- **Alternative**: 20 customers @ 500 invoices/month = $1,500 MRR

---

## WEEK 1: Foundation & Market Validation
**Dates: Days 1-7**

### Development Tasks (20 hours)
- [ ] Set up FastAPI backend with file upload
- [ ] Implement OCR pipeline (Tesseract + GPT-4 Vision)
- [ ] Build invoice extraction model (vendor, amount, date, line items)
- [ ] Create API endpoint: POST /api/v1/extract
- [ ] Deploy to Railway/Render
- [ ] Set up Stripe usage-based billing

### Customer Development (10 hours)
**Target: Talk to 15 bookkeepers/accountants**

Where to find them:
- LinkedIn: "Bookkeeper" OR "Accountant" + "Small Business"
- Alignable (small business network)
- Local accounting association meetups
- Facebook Groups: "Bookkeeping Business Owners", "QuickBooks Users"
- Upwork: Hire bookkeepers for paid interviews ($30 each)
- Reddit: r/bookkeeping, r/accounting

Interview focus:
1. "How many invoices/receipts do you process monthly?"
2. "Walk me through your current data entry process"
3. "What tools do you use? (Dext, Hubdoc, manual?)"
4. "What's your biggest time sink?"
5. "Would you pay $0.15/invoice for 95% accurate extraction?"

### Marketing Activities (5 hours)
- [ ] Create landing page with API demo
- [ ] Write LinkedIn post about bookkeeper pain points
- [ ] Join 5 bookkeeping Facebook groups
- [ ] Create simple demo video (30 seconds)

### Metrics to Track
| Metric | Target | How to Track |
|--------|--------|--------------|
| Discovery calls | 10 | Spreadsheet |
| Waitlist signups | 30 | Landing page |
| Average invoices/month (prospects) | 500+ | Interview notes |
| Willingness to pay confirmed | 7/10 | Interview notes |

### Success Criteria
- [ ] Basic extraction working (90%+ accuracy on test set)
- [ ] 10+ interviews completed
- [ ] Volume opportunity validated (prospects process 500+ invoices/month)
- [ ] 30+ waitlist signups

### Budget Allocation: $250
- Domain + hosting: $40
- Paid interviews: $150
- OpenAI API (GPT-4 Vision): $50
- Miscellaneous: $10

---

## WEEK 2: Core Product Build
**Dates: Days 8-14**

### Development Tasks (25 hours)
- [ ] Add receipt parsing (expenses, merchant, category)
- [ ] Build confidence scoring (flag low-confidence extractions)
- [ ] Implement batch upload (up to 100 documents)
- [ ] Create basic dashboard (usage, history, API keys)
- [ ] Add export formats (CSV, JSON, QuickBooks format)
- [ ] Write comprehensive API documentation
- [ ] Build document preview with extracted fields highlighted

### Customer Development (8 hours)
**Target: 10 more interviews + 3 design partners**

Design Partner Profile:
- Process 200+ invoices/month
- Currently using manual entry or expensive tool
- Tech-savvy enough to use API or Zapier
- Willing to provide feedback

Outreach channels:
- Direct LinkedIn messages to bookkeepers
- Post in bookkeeping Facebook groups offering free pilot
- Partner with bookkeeping coaches/influencers

### Marketing Activities (5 hours)
- [ ] Publish blog: "The Hidden Cost of Manual Invoice Entry"
- [ ] Create comparison page: vs Dext, vs Hubdoc, vs Manual
- [ ] Cold email 50 bookkeeping firms
- [ ] Record full demo video (5 minutes)

### Metrics to Track
| Metric | Target | How to Track |
|--------|--------|--------------|
| Total interviews | 20 | Spreadsheet |
| Design partners signed | 3 | Email |
| Extraction accuracy | 93%+ | Test set |
| Waitlist signups | 80 | Landing page |

### Success Criteria
- [ ] Batch processing working
- [ ] 3 design partners committed
- [ ] 93%+ accuracy on standard invoices
- [ ] API documentation complete

### Budget Allocation: $350
- OpenAI API (increased): $200
- Email outreach tool: $50
- Customer interview gifts: $60
- Hosting: $40

---

## WEEK 3: Integration Focus
**Dates: Days 15-21**

### Development Tasks (20 hours)
- [ ] Build QuickBooks Online integration (OAuth, push entries)
- [ ] Add Xero integration (similar scope)
- [ ] Create Zapier integration (5 triggers/actions)
- [ ] Implement retry logic for failed extractions
- [ ] Add human review queue for low-confidence items
- [ ] Build simple approval workflow

### Customer Development (10 hours)
**Target: Onboard design partners, validate integrations**

Focus areas with beta users:
1. Which accounting software do they use?
2. What's their review/approval process?
3. How important is QuickBooks integration?
4. What accuracy level is acceptable?

### Marketing Activities (8 hours)
- [ ] Create QuickBooks integration announcement
- [ ] Post in QuickBooks user communities
- [ ] Write SEO article: "Best Invoice Scanning Software 2024"
- [ ] Reach out to bookkeeping YouTubers for reviews
- [ ] Create before/after time savings calculator

### Metrics to Track
| Metric | Target | How to Track |
|--------|--------|--------------|
| Design partner activation | 3/3 | Dashboard |
| Documents processed (beta) | 500+ | API logs |
| QuickBooks integration usage | 2/3 partners | Logs |
| Error rate | <5% | Monitoring |

### Success Criteria
- [ ] QuickBooks integration working end-to-end
- [ ] Design partners processing real invoices
- [ ] 500+ documents processed
- [ ] Error rate under 5%

### Budget Allocation: $400
- QuickBooks developer account: $0 (free tier)
- API costs (higher volume): $300
- Zapier partner program: $0
- Beta user incentives: $100

---

## WEEK 4: Pricing & Pre-launch
**Dates: Days 22-28**

### Development Tasks (15 hours)
- [ ] Implement usage tracking and metering
- [ ] Build billing dashboard (usage history, invoices)
- [ ] Add volume discount logic
- [ ] Create admin panel for customer management
- [ ] Performance optimization (target: <3s per document)
- [ ] Add support for more document types (POs, statements)

### Customer Development (12 hours)
**Target: Validate pricing, convert design partners**

Pricing Validation for Usage-Based:
1. "At $0.15/invoice, would you switch from your current tool?"
2. "What's your current cost per invoice (including time)?"
3. "Would you prefer subscription or pay-as-you-go?"
4. "At what price point does this become a no-brainer?"

Volume Commitment Test:
- Offer $0.10/invoice for 1,000+ monthly commitment
- Test appetite for monthly minimums

### Marketing Activities (10 hours)
- [ ] Finalize pricing page
- [ ] Create ROI calculator (time saved * hourly rate vs cost)
- [ ] Write customer success story from beta
- [ ] Prepare launch email sequence
- [ ] Set up affiliate program for bookkeeping influencers

### Metrics to Track
| Metric | Target | How to Track |
|--------|--------|--------------|
| Pricing calls completed | 10 | Calendar |
| Design partner conversions | 2/3 | Stripe |
| Pre-commitments | 5 | Email |
| Waitlist | 150 | Landing page |

### Success Criteria
- [ ] Pricing validated at $0.15 (or adjusted based on feedback)
- [ ] 2+ design partners converted (first revenue)
- [ ] QuickBooks + Xero integrations stable
- [ ] 150+ waitlist signups

### Budget Allocation: $350
- API costs: $250
- Stripe fees: $20
- Launch prep: $80

### MONTH 1 TOTAL BUDGET: $1,350

---

## WEEK 5: Public Launch
**Dates: Days 29-35**

### Development Tasks (10 hours)
- [ ] Final bug fixes from beta
- [ ] Add live chat support
- [ ] Create onboarding tutorial
- [ ] Build API status page
- [ ] Implement rate limiting for free tier

### Customer Development (8 hours)
**Target: Convert waitlist, first 10 paying customers**

Launch Playbook:
1. Email entire waitlist with launch announcement
2. Offer first 500 invoices free for early adopters
3. Personal calls with high-volume prospects
4. Same-day onboarding for interested customers

### Marketing Activities (15 hours)
- [ ] Execute launch campaign across channels:
  - QuickBooks community forums
  - Bookkeeping Facebook groups (10+)
  - r/bookkeeping, r/Entrepreneur
  - LinkedIn (personal + groups)
  - Indie Hackers, Hacker News
  - Product Hunt (if ready)
- [ ] Cold email blast to accounting firms (200 emails)
- [ ] Partner announcement with bookkeeping coach

### Metrics to Track
| Metric | Target | How to Track |
|--------|--------|--------------|
| Active accounts | 15 | Dashboard |
| Documents processed | 2,000 | API logs |
| Revenue | $300+ | Stripe |
| Conversion rate | 10%+ of waitlist | Calculated |

### Success Criteria
- [ ] 15+ active accounts
- [ ] 2,000+ documents processed
- [ ] $300+ revenue (first paying usage)
- [ ] Zero critical production issues

### Budget Allocation: $500
- Google Ads: $200
- API costs (launch spike): $250
- Support tools: $50

---

## WEEK 6: Volume Growth
**Dates: Days 36-42**

### Development Tasks (12 hours)
- [ ] Add requested features from customers
- [ ] Build Google Drive / Dropbox integration
- [ ] Create email forwarding feature (forward invoices to process)
- [ ] Implement multi-currency support
- [ ] Add custom field mapping

### Customer Development (10 hours)
**Target: High-volume customers, reduce churn risk**

Focus: Find and convert high-volume users
- Bookkeeping agencies (process for multiple clients)
- Property management companies
- E-commerce businesses
- Accounts payable departments

### Marketing Activities (12 hours)
- [ ] Analyze which channels drove conversions
- [ ] Double down on working channels
- [ ] Create industry-specific landing pages
- [ ] Write 3 SEO articles targeting long-tail keywords
- [ ] Launch referral program (give $20 credit, get $20)

### Metrics to Track
| Metric | Target | How to Track |
|--------|--------|--------------|
| Active accounts | 30 | Dashboard |
| Documents processed | 6,000 | API logs |
| Revenue | $900+ | Stripe |
| Average docs/customer | 200+ | Calculated |

### Success Criteria
- [ ] 30+ active accounts
- [ ] $800+ in revenue (on track for $1,600 MRR run rate)
- [ ] At least 3 high-volume customers (500+ docs/month)
- [ ] Referral program generating leads

### Budget Allocation: $550
- Google Ads: $300
- API costs: $200
- Referral credits: $50

---

## WEEK 7: Scale & Optimize
**Dates: Days 43-49**

### Development Tasks (10 hours)
- [ ] Build accounting firm dashboard (multi-client)
- [ ] Add approval workflows
- [ ] Create mobile upload app (React Native simple)
- [ ] Implement automatic categorization training
- [ ] Add duplicate detection

### Customer Development (8 hours)
**Target: Accounting firms (B2B2C model)**

New Segment: Accounting Firms
- They process invoices for dozens of clients
- One firm = 1,000+ invoices/month
- Longer sales cycle but higher LTV

Outreach:
- Partner with accounting software consultants
- Attend virtual CPA/bookkeeper conferences
- Create "Firm Edition" marketing

### Marketing Activities (15 hours)
- [ ] Launch "For Accounting Firms" landing page
- [ ] Create partner program for accountants
- [ ] Webinar: "Automate Your Clients' AP Process"
- [ ] LinkedIn Ads targeting accountants
- [ ] Case study with high-volume customer

### Metrics to Track
| Metric | Target | How to Track |
|--------|--------|--------------|
| Active accounts | 50 | Dashboard |
| Documents processed | 10,000 | API logs |
| Revenue | $1,500+ | Stripe |
| Accounting firm pilots | 2 | CRM |

### Success Criteria
- [ ] 50+ active accounts
- [ ] $1,400+ revenue
- [ ] 2+ accounting firm conversations
- [ ] 10,000+ documents processed

### Budget Allocation: $600
- LinkedIn Ads: $300
- Google Ads: $200
- Webinar platform: $50
- API costs: $50 (efficiency gains)

---

## WEEK 8: Path to $2K MRR
**Dates: Days 50-56**

### Development Tasks (10 hours)
- [ ] Build annual commitment plans with discount
- [ ] Add team management features
- [ ] Create white-label option for firms
- [ ] Implement SLA monitoring
- [ ] Add more accounting integrations (FreshBooks, Wave)

### Customer Development (10 hours)
**Target: Annual commitments, reduce churn, expand accounts**

Tactics:
1. Offer 25% discount for annual prepay
2. Contact customers approaching tier thresholds
3. Identify expansion opportunities (more document types)
4. Get case studies and testimonials

### Marketing Activities (12 hours)
- [ ] Push annual plans to existing customers
- [ ] Create "Why Firms Choose Us" content
- [ ] Partner content with accounting influencer
- [ ] Launch on accounting software marketplaces
- [ ] Retargeting campaign for trial users

### Metrics to Track
| Metric | Target | How to Track |
|--------|--------|--------------|
| Active accounts | 70 | Dashboard |
| Documents processed | 14,000 | API logs |
| Revenue | $2,100+ | Stripe |
| Annual commitments | 3 | Stripe |

### Success Criteria
- [ ] 70+ active accounts
- [ ] $2,000+ MRR (TARGET HIT)
- [ ] 3+ annual commitments
- [ ] Path to $3K MRR clear

### Budget Allocation: $500
- Ads: $300
- API costs: $100
- Partnership/content: $100

---

## WEEKS 9-12: Consolidate & Scale
**Dates: Days 57-90**

### Development Priorities
- Advanced ML for edge cases
- More integrations (Sage, NetSuite)
- Mobile app refinement
- API v2 with enhanced features
- Enterprise features (audit logs, compliance)

### Growth Priorities
- Scale accounting firm segment
- Build partner network
- Marketplace listings (QuickBooks App Store, Xero Marketplace)
- Content marketing at scale
- Consider Chrome extension for easy capture

### Week 9 Target: $2,400 MRR
### Week 10 Target: $2,800 MRR
### Week 11 Target: $3,200 MRR
### Week 12 Target: $3,600 MRR

### Budget Allocation (Weeks 9-12): $1,500
- Ads: $800
- API costs: $400
- Partnerships: $200
- Tools: $100

---

## OPTION B: TOTAL BUDGET BREAKDOWN

| Category | Amount | % of Budget |
|----------|--------|-------------|
| Development (API, hosting) | $1,400 | 28% |
| Paid Advertising | $1,800 | 36% |
| Customer Interviews/Incentives | $350 | 7% |
| Tools & Subscriptions | $350 | 7% |
| Content & Marketing | $400 | 8% |
| Contingency | $700 | 14% |
| **TOTAL** | **$5,000** | 100% |

---

# RISK MITIGATION STRATEGIES (BOTH OPTIONS)

## Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| API costs exceed projections | High | Set hard spending caps, optimize prompts, cache results |
| Accuracy below acceptable | High | Use confidence scoring, human review fallback, continuous training |
| Scaling issues | Medium | Start with serverless, monitor closely, have scaling plan |
| Security breach | Critical | SOC2-lite practices, encryption, regular audits |
| Third-party API outages | Medium | Multi-provider fallback (OpenAI + Anthropic), graceful degradation |

## Market Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| No one wants to pay | Critical | Validate pricing early, pivot to different segment |
| Price too low | High | Test higher prices first, add premium tier |
| Competition launches similar | Medium | Focus on niche, build relationships, move fast |
| Market too small | High | Validate TAM in week 1, have adjacent markets ready |

## Execution Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Can't find customers to interview | High | Use paid interviews, leverage warm intros, try different channels |
| Feature creep delays launch | High | Strict MVP scope, time-boxed sprints |
| Burnout (solo founder) | High | Protect weekends, automate early, consider part-time help |
| Cash runs out | Critical | Track weekly, cut non-essentials, have runway buffer |

---

# PIVOT TRIGGERS

## When to Pivot Within the Product

**Resume Screening (Option A)**:
- Pivot to different buyer: If staffing agencies aren't buying, try corporate HR
- Pivot to different feature: If matching not valued, focus on ATS integration
- Pivot to different pricing: If subscription not working, try pure usage-based

**Document Intelligence (Option B)**:
- Pivot to different document type: If invoices aren't enough, try contracts
- Pivot to different buyer: If bookkeepers aren't buying, try small businesses directly
- Pivot to different model: If usage-based isn't working, try subscription

## Hard Pivot Triggers (Switch Products Entirely)

Consider pivoting if by **Day 45**:
- [ ] Fewer than 3 paying customers
- [ ] MRR below $300
- [ ] Unable to complete 15+ customer interviews
- [ ] No clear path to $2K MRR

## Pivot Decision Framework

```
Week 6 Checkpoint:
├── MRR >= $800?
│   ├── Yes → Continue, double down on working channels
│   └── No →
│       ├── Have 5+ paying customers?
│       │   ├── Yes → Price problem, test higher prices
│       │   └── No → Acquisition problem, change channels
│       └── Have 20+ trials?
│           ├── Yes → Conversion problem, improve onboarding
│           └── No → Awareness problem, increase marketing spend
```

---

# FINAL RECOMMENDATION

## Probability Analysis: $2K MRR in 90 Days

### Option A: Resume Screening API

**Pros:**
- Clear buyer (staffing agencies have budget)
- Higher price points ($149-299/month)
- Fewer customers needed (7-14 for $2K MRR)
- Subscription = predictable revenue
- Strong pain point (time savings quantifiable)

**Cons:**
- Competitive market (many ATS tools have AI)
- Longer sales cycle (2-4 weeks)
- Requires demos/hand-holding
- Enterprise-ish buyers = slower decisions
- Technical complexity (ATS integrations)

**Probability of $2K MRR: 55%**

Key Success Factors:
- Get 3 design partners in week 2
- Convert 2/3 to paid by week 4
- Systematic outbound to hit 15+ customers

### Option B: Document Intelligence API

**Pros:**
- Larger market (300K+ bookkeepers in US)
- Shorter sales cycle (self-serve possible)
- Clear ROI calculation (time saved = money)
- Integrations (QuickBooks) are differentiator
- Lower barrier to try (pay-as-you-go)

**Cons:**
- Lower price point ($0.15/invoice)
- Need high volume for meaningful revenue
- Established competitors (Dext, Hubdoc)
- Usage-based = less predictable revenue
- May attract low-volume users

**Probability of $2K MRR: 45%**

Key Success Factors:
- Find 10+ customers processing 500+ invoices/month
- QuickBooks integration must work flawlessly
- Volume discounts to lock in commitments

---

## RECOMMENDATION: **OPTION A - AI Resume Screening API**

### Reasoning:

1. **Math is easier**: Need 10 customers at $200 avg vs. 13,333 invoices/month. With staffing agencies processing hundreds of resumes, each customer provides meaningful revenue.

2. **Clearer buyer**: Staffing agency owners have P&L responsibility and can make purchasing decisions quickly. Bookkeepers often work for someone else and need approval.

3. **Higher willingness to pay**: Staffing is a $150B industry with healthy margins. A $200/month tool that saves 10+ hours is an obvious ROI.

4. **Subscription stability**: Monthly subscriptions provide predictable MRR. Usage-based requires constant volume to maintain revenue.

5. **Defensibility**: Deep integration with staffing workflows creates switching costs. Invoice OCR is more commoditized.

6. **Expansion potential**: Can upsell to larger tiers, add seats, expand to interview scheduling, etc.

### However, Consider Option B If:

- You have existing relationships in accounting/bookkeeping
- You've personally experienced the invoice processing pain
- You prefer self-serve GTM over sales calls
- You want lower technical complexity
- You're more excited about the bookkeeping space

### Final Success Formula (Option A):

```
Week 1-2: 20 interviews, 3 design partners
Week 3-4: Beta launch, 2 paying customers ($400 MRR)
Week 5-6: Public launch, 8 more customers ($1,600 MRR)
Week 7-8: Scaling, 5 more customers + upsells ($2,200 MRR)
Week 9-12: Consolidation, enterprise ($3,000+ MRR)
```

### Commitment Required:

- **Hours/week**: 35-45 (development + sales + marketing)
- **Customer calls**: 5-8 per week for first 6 weeks
- **Content creation**: 2-3 pieces per week
- **Emotional resilience**: Expect 80%+ rejection on outbound

---

## IMMEDIATE NEXT STEPS (TODAY)

### Option A Selected:
1. [ ] Register domain (resumescreen.ai or similar)
2. [ ] Set up landing page with waitlist
3. [ ] Join 5 LinkedIn groups for recruiters/staffing
4. [ ] Draft customer interview script
5. [ ] Send 10 LinkedIn connection requests to staffing agency owners
6. [ ] Start coding basic resume parser

### Option B Selected:
1. [ ] Register domain (invoiceai.io or similar)
2. [ ] Set up landing page with demo
3. [ ] Join 5 bookkeeping Facebook groups
4. [ ] Draft customer interview script
5. [ ] Post intro in 2 bookkeeping communities
6. [ ] Start coding basic OCR pipeline

---

*Document created: Day 0 of 90-day sprint*
*Next review: Day 7*
*Success metric: $2,000+ MRR by Day 90*
