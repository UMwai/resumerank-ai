import { serve } from '@hono/node-server'
import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { logger } from 'hono/logger'

const app = new Hono()

// Middleware
app.use('*', logger())
app.use('*', cors())

// Health check
app.get('/health', (c) => {
  return c.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: '0.1.0'
  })
})

// API routes
app.get('/api/v1', (c) => {
  return c.json({
    message: 'ResumeRank AI API v1',
    endpoints: {
      parse: 'POST /api/v1/resumes/parse',
      screen: 'POST /api/v1/resumes/screen',
      batch: 'POST /api/v1/batch'
    },
    documentation: 'https://docs.resumerank.ai'
  })
})

// 404 handler
app.notFound((c) => {
  return c.json({ error: 'Not found', path: c.req.path }, 404)
})

// Error handler
app.onError((err, c) => {
  console.error(err)
  return c.json({
    error: 'Internal server error',
    message: err.message
  }, 500)
})

const port = parseInt(process.env.PORT || '3000')

console.log(`=€ ResumeRank AI starting on port ${port}`)

serve({
  fetch: app.fetch,
  port
})
