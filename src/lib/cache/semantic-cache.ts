import { createHash } from 'crypto'

/**
 * Semantic caching for AI responses to reduce costs
 * Implements GPT-5 mini's 90% cost reduction for repeated prompts
 */
export class SemanticCache {
  private cache: Map<string, {
    data: any
    timestamp: number
    hits: number
  }> = new Map()

  private readonly TTL = 7 * 24 * 60 * 60 * 1000 // 7 days
  private readonly MAX_ENTRIES = 10000

  /**
   * Generate cache key from job description
   * Uses first 1000 chars + hash for similarity matching
   */
  generateKey(jobDescription: string): string {
    // Normalize: lowercase, trim whitespace, remove extra spaces
    const normalized = jobDescription
      .toLowerCase()
      .trim()
      .replace(/\s+/g, ' ')

    // Create hash of full description
    const hash = createHash('sha256')
      .update(normalized)
      .digest('hex')
      .substring(0, 16)

    // Use first 200 chars + hash for key
    const prefix = normalized.substring(0, 200)
    return `${prefix}_${hash}`
  }

  /**
   * Get cached result if exists and not expired
   */
  get<T>(key: string): T | null {
    const entry = this.cache.get(key)

    if (!entry) {
      return null
    }

    // Check expiration
    const age = Date.now() - entry.timestamp
    if (age > this.TTL) {
      this.cache.delete(key)
      return null
    }

    // Increment hit counter
    entry.hits++

    return entry.data as T
  }

  /**
   * Set cache entry
   */
  set<T>(key: string, data: T): void {
    // Evict oldest entries if cache is full
    if (this.cache.size >= this.MAX_ENTRIES) {
      this.evictOldest()
    }

    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      hits: 0
    })
  }

  /**
   * Check if key exists in cache
   */
  has(key: string): boolean {
    const entry = this.cache.get(key)
    if (!entry) return false

    const age = Date.now() - entry.timestamp
    if (age > this.TTL) {
      this.cache.delete(key)
      return false
    }

    return true
  }

  /**
   * Get cache statistics
   */
  getStats(): {
    size: number
    hitRate: number
    avgAge: number
    totalHits: number
  } {
    let totalHits = 0
    let totalAge = 0
    const now = Date.now()

    for (const entry of this.cache.values()) {
      totalHits += entry.hits
      totalAge += (now - entry.timestamp)
    }

    const size = this.cache.size
    const avgAge = size > 0 ? totalAge / size : 0

    return {
      size,
      hitRate: size > 0 ? totalHits / size : 0,
      avgAge: avgAge / 1000 / 60, // minutes
      totalHits
    }
  }

  /**
   * Evict oldest 10% of entries
   */
  private evictOldest(): void {
    const entries = Array.from(this.cache.entries())
      .sort((a, b) => a[1].timestamp - b[1].timestamp)

    const toRemove = Math.floor(entries.length * 0.1)
    for (let i = 0; i < toRemove; i++) {
      this.cache.delete(entries[i][0])
    }
  }

  /**
   * Clear all cache entries
   */
  clear(): void {
    this.cache.clear()
  }

  /**
   * Calculate cost savings from cache hits
   */
  calculateSavings(
    cacheHits: number,
    costPerRequest: number,
    cacheDiscount: number = 0.90
  ): number {
    return cacheHits * costPerRequest * cacheDiscount
  }
}

// Export singleton
export const semanticCache = new SemanticCache()
