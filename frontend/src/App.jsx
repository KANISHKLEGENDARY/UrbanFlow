import { useState, useEffect, useCallback } from 'react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

const TIME_SLOTS = Array.from({ length: 96 }, (_, i) => i)
const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

const DEMAND_COLORS = {
  low: 'var(--demand-low)',
  moderate: 'var(--demand-moderate)',
  high: 'var(--demand-high)',
  critical: 'var(--demand-critical)',
}

function getTimeParts(timeSlot) {
  return {
    hour: Math.floor(timeSlot / 4),
    minute: (timeSlot % 4) * 15,
  }
}

function formatTimeSlot(timeSlot) {
  const { hour, minute } = getTimeParts(timeSlot)
  const suffix = hour < 12 ? 'AM' : 'PM'
  const displayHour = hour % 12 === 0 ? 12 : hour % 12
  return `${displayHour}:${minute.toString().padStart(2, '0')} ${suffix}`
}

/* ─── Auth Utilities ───────────────────────── */
function getAuthHeaders() {
  const token = localStorage.getItem('uf_access_token')
  return token ? { Authorization: `Bearer ${token}` } : {}
}

async function apiFetch(url, options = {}) {
  const headers = { ...getAuthHeaders(), ...(options.headers || {}) }
  const res = await fetch(url, { ...options, headers })
  if (res.status === 401) {
    // Try to refresh the token
    const refreshToken = localStorage.getItem('uf_refresh_token')
    if (refreshToken) {
      try {
        const refreshRes = await fetch(`${API_BASE}/api/v1/auth/refresh`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ refresh_token: refreshToken }),
        })
        if (refreshRes.ok) {
          const data = await refreshRes.json()
          localStorage.setItem('uf_access_token', data.access_token)
          // Retry original request with new token
          const retryHeaders = { ...options.headers, Authorization: `Bearer ${data.access_token}` }
          return fetch(url, { ...options, headers: retryHeaders })
        }
      } catch (e) { /* fall through to return 401 */ }
    }
    localStorage.removeItem('uf_access_token')
    localStorage.removeItem('uf_refresh_token')
    localStorage.removeItem('uf_user')
  }
  return res
}

/* ─── Auth Modal ───────────────────────────── */
function AuthModal({ onClose, onAuth }) {
  const [mode, setMode] = useState('login')
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const endpoint = mode === 'login' ? 'login' : 'register'
      const body = mode === 'login'
        ? { username, password }
        : { username, email, password }
      const res = await fetch(`${API_BASE}/api/v1/auth/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || `${endpoint} failed`)
      localStorage.setItem('uf_access_token', data.access_token)
      localStorage.setItem('uf_refresh_token', data.refresh_token)
      localStorage.setItem('uf_user', JSON.stringify({ username: data.username, email: data.email }))
      onAuth({ username: data.username, email: data.email })
    } catch (err) {
      setError(err.message)
    }
    setLoading(false)
  }

  return (
    <div className="auth-overlay" onClick={onClose}>
      <div className="auth-modal glass-card" onClick={e => e.stopPropagation()}>
        <button className="auth-close" onClick={onClose}>×</button>
        <h2 className="auth-title gradient-text">
          {mode === 'login' ? 'Welcome Back' : 'Create Account'}
        </h2>
        <p className="auth-subtitle">
          {mode === 'login' ? 'Sign in to UrbanFlow' : 'Join UrbanFlow'}
        </p>

        <form className="auth-form" onSubmit={handleSubmit}>
          <div className="auth-field">
            <label className="control-label">Username</label>
            <input
              type="text"
              className="auth-input"
              value={username}
              onChange={e => setUsername(e.target.value)}
              placeholder="Enter username"
              required
              minLength={3}
            />
          </div>
          {mode === 'register' && (
            <div className="auth-field">
              <label className="control-label">Email</label>
              <input
                type="email"
                className="auth-input"
                value={email}
                onChange={e => setEmail(e.target.value)}
                placeholder="Enter email address"
                required
              />
            </div>
          )}
          <div className="auth-field">
            <label className="control-label">Password</label>
            <input
              type="password"
              className="auth-input"
              value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder="Enter password"
              required
              minLength={6}
            />
          </div>

          {error && <div className="auth-error">{error}</div>}

          <button type="submit" className="auth-submit" disabled={loading}>
            {loading ? 'Processing...' : mode === 'login' ? 'Sign In' : 'Create Account'}
          </button>
        </form>

        <p className="auth-switch">
          {mode === 'login' ? "Don't have an account? " : 'Already have an account? '}
          <button className="auth-switch-btn" onClick={() => { setMode(mode === 'login' ? 'register' : 'login'); setError('') }}>
            {mode === 'login' ? 'Sign Up' : 'Sign In'}
          </button>
        </p>
      </div>
    </div>
  )
}

/* ─── Header Component ─────────────────────── */
function Header({ health, user, onShowAuth, onLogout }) {
  return (
    <header className="header">
      <div className="header-left">
        <div className="logo">
          <div className="logo-icon">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
              <path d="M14 2L26 8V20L14 26L2 20V8L14 2Z" stroke="url(#grad)" strokeWidth="2" fill="none"/>
              <path d="M14 10L20 13V19L14 22L8 19V13L14 10Z" fill="url(#grad)" opacity="0.6"/>
              <defs>
                <linearGradient id="grad" x1="2" y1="2" x2="26" y2="26">
                  <stop stopColor="#6366f1"/>
                  <stop offset="1" stopColor="#a855f7"/>
                </linearGradient>
              </defs>
            </svg>
          </div>
          <div>
            <h1 className="logo-text gradient-text">UrbanFlow</h1>
            <p className="logo-subtitle">NYC Ride Demand Intelligence</p>
          </div>
        </div>
      </div>
      <div className="header-right">
        <div className={`status-dot ${health?.model_loaded ? 'status-healthy' : 'status-error'}`} />
        <span className="header-status">
          {health?.model_loaded
            ? `${health.zones_available} zones active`
            : 'Connecting...'}
        </span>
        <div className="header-auth">
          {user ? (
            <>
              <span className="auth-user-badge">👤 {user.username}</span>
              <button className="auth-header-btn" onClick={onLogout}>Logout</button>
            </>
          ) : health?.auth_enabled ? (
            <button className="auth-header-btn auth-header-btn-primary" onClick={onShowAuth}>Sign In</button>
          ) : null}
        </div>
      </div>
    </header>
  )
}

/* ─── Stats Cards ──────────────────────────── */
function StatsCards({ heatmapData }) {
  if (!heatmapData || !heatmapData.zones) return null
  const { zones } = heatmapData
  const criticalCount = zones.filter(z => z.demand_level === 'critical').length
  const highCount = zones.filter(z => z.demand_level === 'high').length
  const avgSurge = (zones.reduce((s, z) => s + z.surge_multiplier, 0) / zones.length).toFixed(2)
  const topZone = zones.reduce((a, b) => a.predicted_demand > b.predicted_demand ? a : b, zones[0])

  const cards = [
    {
      label: 'Average Demand',
      value: `${(heatmapData.avg_demand * 100).toFixed(1)}%`,
      detail: `${heatmapData.total_zones} zones`,
      icon: '📊',
      color: 'var(--accent-cyan)',
    },
    {
      label: 'Critical Zones',
      value: criticalCount,
      detail: `${highCount} high demand`,
      icon: '🔴',
      color: 'var(--demand-critical)',
    },
    {
      label: 'Avg Surge',
      value: `${avgSurge}x`,
      detail: 'multiplier',
      icon: '⚡',
      color: 'var(--accent-amber)',
    },
    {
      label: 'Hottest Zone',
      value: topZone?.zone_name?.substring(0, 16) || '—',
      detail: `${(topZone?.predicted_demand * 100).toFixed(0)}% capacity`,
      icon: '🔥',
      color: 'var(--demand-high)',
    },
  ]

  return (
    <div className="stats-grid stagger-children">
      {cards.map((card, i) => (
        <div key={i} className="stat-card glass-card">
          <div className="stat-icon">{card.icon}</div>
          <div className="stat-content">
            <span className="stat-label">{card.label}</span>
            <span className="stat-value" style={{ color: card.color }}>{card.value}</span>
            <span className="stat-detail">{card.detail}</span>
          </div>
        </div>
      ))}
    </div>
  )
}

/* ─── Time Controls ────────────────────────── */
function TimeControls({ timeSlot, setTimeSlot, dayOfWeek, setDayOfWeek }) {
  return (
    <div className="controls glass-card">
      <div className="control-group">
        <label className="control-label">Time of Day</label>
        <div className="hour-slider-container">
          <input
            type="range"
            id="hour-slider"
            min="0"
            max={TIME_SLOTS.length - 1}
            value={timeSlot}
            onChange={(e) => setTimeSlot(parseInt(e.target.value))}
            className="hour-slider"
          />
          <span className="hour-display">{formatTimeSlot(timeSlot)}</span>
        </div>
      </div>
      <div className="control-group">
        <label className="control-label">Day of Week</label>
        <div className="day-pills">
          {DAYS.map((day, i) => (
            <button
              key={day}
              id={`day-btn-${i}`}
              className={`day-pill ${i === dayOfWeek ? 'day-pill-active' : ''}`}
              onClick={() => setDayOfWeek(i)}
            >
              {day.substring(0, 3)}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

/* ─── Zone Grid (Heatmap) ──────────────────── */
function ZoneGrid({ zones, selectedZone, onSelectZone }) {
  if (!zones || zones.length === 0) {
    return (
      <div className="zone-grid-container glass-card">
        <h3 className="section-title">Zone Demand Heatmap</h3>
        <div className="loading-state">Loading zone data...</div>
      </div>
    )
  }

  // Group by borough
  const boroughs = {}
  zones.forEach(z => {
    if (!boroughs[z.borough]) boroughs[z.borough] = []
    boroughs[z.borough].push(z)
  })

  // Sort each borough by demand
  Object.keys(boroughs).forEach(b => {
    boroughs[b].sort((a, c) => c.predicted_demand - a.predicted_demand)
  })

  const boroughOrder = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island', 'EWR']

  return (
    <div className="zone-grid-container glass-card">
      <h3 className="section-title">
        <span>🗺️ Zone Demand Heatmap</span>
        <span className="section-subtitle">{zones.length} zones</span>
      </h3>
      <div className="borough-sections">
        {boroughOrder.filter(b => boroughs[b]).map(borough => (
          <div key={borough} className="borough-section">
            <h4 className="borough-title">{borough}
              <span className="borough-count">{boroughs[borough].length} zones</span>
            </h4>
            <div className="zone-tiles">
              {boroughs[borough].slice(0, 20).map(zone => (
                <button
                  key={zone.zone_id}
                  id={`zone-tile-${zone.zone_id}`}
                  className={`zone-tile ${selectedZone?.zone_id === zone.zone_id ? 'zone-tile-selected' : ''}`}
                  style={{
                    '--demand-color': DEMAND_COLORS[zone.demand_level],
                    '--demand-opacity': Math.max(0.15, zone.predicted_demand),
                  }}
                  onClick={() => onSelectZone(zone)}
                  title={`${zone.zone_name}: ${(zone.predicted_demand * 100).toFixed(1)}%`}
                >
                  <div className="zone-tile-bar" />
                  <span className="zone-tile-name">{zone.zone_name?.substring(0, 18)}</span>
                  <span className="zone-tile-demand" style={{ color: DEMAND_COLORS[zone.demand_level] }}>
                    {(zone.predicted_demand * 100).toFixed(0)}%
                  </span>
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ─── Zone Detail Panel ────────────────────── */
function ZoneDetail({ zone, pricing, explanation, hour, minute, dayOfWeek }) {
  if (!zone) {
    return (
      <div className="zone-detail glass-card">
        <div className="empty-state">
          <span className="empty-icon">📍</span>
          <p>Select a zone from the heatmap to see details</p>
        </div>
      </div>
    )
  }

  return (
    <div className="zone-detail glass-card animate-fade-in-up">
      <div className="detail-header">
        <div>
          <h3 className="detail-zone-name">{zone.zone_name}</h3>
          <span className="detail-borough">{zone.borough} · Zone {zone.zone_id}</span>
        </div>
        <span className={`badge badge-${zone.demand_level}`}>
          {zone.demand_level}
        </span>
      </div>

      {/* Demand Gauge */}
      <div className="demand-gauge">
        <div className="gauge-label">
          <span>Predicted Demand</span>
          <span style={{ color: DEMAND_COLORS[zone.demand_level], fontWeight: 700 }}>
            {(zone.predicted_demand * 100).toFixed(1)}%
          </span>
        </div>
        <div className="gauge-bar">
          <div
            className="gauge-fill"
            style={{
              width: `${zone.predicted_demand * 100}%`,
              background: `linear-gradient(90deg, var(--demand-low), ${DEMAND_COLORS[zone.demand_level]})`,
            }}
          />
        </div>
      </div>

      {/* Surge Pricing */}
      {pricing && (
        <div className="pricing-section">
          <div className="pricing-header">
            <span>⚡ Surge Pricing</span>
            <span className="surge-value">{pricing.surge_multiplier}x</span>
          </div>
          <div className="pricing-fare">
            $15.00 base → <strong>${pricing.adjusted_fare}</strong>
          </div>
          <div className="pricing-factors">
            {pricing.factors?.map((f, i) => (
              <div key={i} className="factor-row">
                <span className="factor-name">{f.factor}</span>
                <span className="factor-impact">{f.impact}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Multi-Modal Choice Splits */}
      {zone && (
        <MultimodalChoice
          zone={zone}
          pricing={pricing}
          hour={hour}
          minute={minute}
          dayOfWeek={dayOfWeek}
        />
      )}

      {/* SHAP Explanation */}
      {explanation && (
        <div className="explanation-section">
          <h4 className="explanation-title">🧠 AI Explanation</h4>
          <p className="explanation-subtitle">Why this prediction?</p>
          <div className="shap-factors">
            {explanation.top_factors?.slice(0, 6).map((f, i) => (
              <div key={i} className="shap-row">
                <span className="shap-feature">{f.feature}</span>
                <div className="shap-bar-container">
                  <div
                    className={`shap-bar ${f.direction === 'positive' ? 'shap-bar-positive' : 'shap-bar-negative'}`}
                    style={{ width: `${Math.min(100, Math.abs(f.impact) * 500)}%` }}
                  />
                </div>
                <span className={`shap-impact ${f.direction === 'positive' ? 'shap-positive' : 'shap-negative'}`}>
                  {f.impact > 0 ? '+' : ''}{f.impact.toFixed(4)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

/* ─── ETA Panel ─────────────────────────────── */
function EtaPanel({ zones, selectedZone, hour, minute, dayOfWeek }) {
  const [pickupZone, setPickupZone] = useState('')
  const [dropoffZone, setDropoffZone] = useState('')
  const [eta, setEta] = useState(null)
  const [etaError, setEtaError] = useState('')
  const [etaLoading, setEtaLoading] = useState(false)

  const zoneOptions = [...(zones || [])].sort((a, b) => a.zone_name.localeCompare(b.zone_name))

  useEffect(() => {
    if (selectedZone) {
      setPickupZone(String(selectedZone.zone_id))
    }
  }, [selectedZone])

  useEffect(() => {
    if (!zoneOptions.length) return
    if (!pickupZone) setPickupZone(String(zoneOptions[0].zone_id))
    if (!dropoffZone) {
      const fallback = zoneOptions.find(z => String(z.zone_id) !== pickupZone) || zoneOptions[0]
      setDropoffZone(String(fallback.zone_id))
    }
  }, [zoneOptions, pickupZone, dropoffZone])

  const swapZones = () => {
    const tmp = pickupZone
    setPickupZone(dropoffZone)
    setDropoffZone(tmp)
    setEta(null)
    setEtaError('')
  }

  const estimateEta = async () => {
    if (!pickupZone || !dropoffZone) return
    setEtaLoading(true)
    setEtaError('')
    try {
      const res = await apiFetch(`${API_BASE}/api/v1/eta/estimate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pickup_zone: parseInt(pickupZone),
          dropoff_zone: parseInt(dropoffZone),
          hour,
          minute,
          day_of_week: dayOfWeek,
        }),
      })
      if (!res.ok) throw new Error(`ETA request failed (${res.status})`)
      setEta(await res.json())
    } catch (e) {
      setEta(null)
      setEtaError(e.message || 'ETA unavailable')
    }
    setEtaLoading(false)
  }

  return (
    <div className="eta-panel glass-card">
      <h3 className="section-title">
        <span>🕐 ETA Estimate</span>
        <span className="section-subtitle">{formatTimeSlot(hour * 4 + minute / 15)}</span>
      </h3>

      <div className="eta-form">
        <label className="eta-field">
          <span className="control-label">From</span>
          <select value={pickupZone} onChange={(e) => setPickupZone(e.target.value)} className="eta-select">
            {zoneOptions.map(zone => (
              <option key={zone.zone_id} value={zone.zone_id}>
                {zone.zone_name} ({zone.borough})
              </option>
            ))}
          </select>
        </label>
        <button className="eta-swap-btn" onClick={swapZones} title="Swap pickup and dropoff zones">⇅</button>
        <label className="eta-field">
          <span className="control-label">To</span>
          <select value={dropoffZone} onChange={(e) => setDropoffZone(e.target.value)} className="eta-select">
            {zoneOptions.map(zone => (
              <option key={zone.zone_id} value={zone.zone_id}>
                {zone.zone_name} ({zone.borough})
              </option>
            ))}
          </select>
        </label>
        <button className="eta-button" onClick={estimateEta} disabled={etaLoading || !zoneOptions.length}>
          {etaLoading ? 'Estimating...' : 'Estimate ETA'}
        </button>
      </div>

      {etaError && <div className="eta-error">{etaError}</div>}

      {eta && (
        <div className="eta-result">
          <div className="eta-route-direction">
            <span className="eta-route-zone">{eta.pickup_zone_name}</span>
            <span className="eta-route-arrow">→</span>
            <span className="eta-route-zone">{eta.dropoff_zone_name}</span>
          </div>
          <div className="eta-main">
            <span className="eta-value">{eta.adjusted_eta_minutes.toFixed(1)}</span>
            <span className="eta-unit">min</span>
          </div>
          <div className="eta-grid">
            <div>
              <span className="eta-label">Free-flow</span>
              <strong>{eta.base_eta_minutes.toFixed(1)} min</strong>
            </div>
            <div>
              <span className="eta-label">Traffic</span>
              <strong>{eta.traffic_multiplier.toFixed(2)}x</strong>
            </div>
            <div>
              <span className="eta-label">Distance</span>
              <strong>{eta.distance_miles.toFixed(1)} mi</strong>
            </div>
            <div>
              <span className="eta-label">Mode</span>
              <strong>{eta.graph_available ? 'Graph' : 'Fallback'}</strong>
            </div>
          </div>
          <p className="eta-route">{eta.route_summary}</p>
        </div>
      )}
    </div>
  )
}

function FeatureImportance({ features }) {
  if (!features || features.length === 0) return null

  const maxImportance = features[0]?.importance || 1

  return (
    <div className="feature-importance glass-card">
      <h3 className="section-title">
        <span>🏆 Global Feature Importance</span>
        <span className="section-subtitle">XGBoost</span>
      </h3>
      <div className="feature-bars stagger-children">
        {features.slice(0, 12).map((f, i) => (
          <div key={f.feature} className="feature-row">
            <span className="feature-rank">#{f.rank}</span>
            <span className="feature-name">{f.feature}</span>
            <div className="feature-bar-container">
              <div
                className="feature-bar"
                style={{ width: `${(f.importance / maxImportance) * 100}%` }}
              />
            </div>
            <span className="feature-score">{f.importance.toFixed(3)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ─── Top Zones Leaderboard ────────────────── */
function TopZones({ zones }) {
  if (!zones || zones.length === 0) return null

  const sorted = [...zones].sort((a, b) => b.predicted_demand - a.predicted_demand).slice(0, 10)

  return (
    <div className="top-zones glass-card">
      <h3 className="section-title">
        <span>🏙️ Highest Demand Zones</span>
        <span className="section-subtitle">Top 10</span>
      </h3>
      <div className="leaderboard stagger-children">
        {sorted.map((zone, i) => (
          <div key={zone.zone_id} className="leaderboard-row">
            <span className="lb-rank">{i + 1}</span>
            <div className="lb-info">
              <span className="lb-name">{zone.zone_name}</span>
              <span className="lb-borough">{zone.borough}</span>
            </div>
            <div className="lb-demand-bar">
              <div
                className="lb-bar-fill"
                style={{
                  width: `${zone.predicted_demand * 100}%`,
                  background: DEMAND_COLORS[zone.demand_level],
                }}
              />
            </div>
            <span className={`badge badge-${zone.demand_level}`}>
              {(zone.predicted_demand * 100).toFixed(0)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ─── System Status Panel ──────────────────── */
function SystemStatus({ health, user }) {
  if (!health) return null

  const items = [
    {
      label: 'ML Models',
      status: health.model_loaded,
      detail: health.model_loaded ? `${health.zones_available} zones` : 'Not loaded',
      icon: '🧠',
    },
    {
      label: 'ETA Service',
      status: health.eta_loaded,
      detail: health.eta_loaded ? 'Active' : 'Unavailable',
      icon: '🕐',
    },
    {
      label: 'Database',
      status: health.database_available,
      detail: health.database_available ? 'PostgreSQL connected' : 'File-backed mode',
      icon: '🗄️',
    },
    {
      label: 'Authentication',
      status: health.auth_enabled,
      detail: health.auth_enabled
        ? (user ? `Signed in as ${user.username}` : 'Sign in required')
        : 'Open access',
      icon: '🔐',
    },
  ]

  return (
    <div className="system-status glass-card">
      <h3 className="section-title">
        <span>⚙️ System Status</span>
        <span className="section-subtitle">v{health.version || '2.0.0'}</span>
      </h3>
      <div className="status-items">
        {items.map(item => (
          <div key={item.label} className="status-item">
            <span className="status-item-icon">{item.icon}</span>
            <div className="status-item-info">
              <span className="status-item-label">{item.label}</span>
              <span className="status-item-detail">{item.detail}</span>
            </div>
            <div className={`status-indicator ${item.status ? 'status-indicator-on' : 'status-indicator-off'}`} />
          </div>
        ))}
      </div>
    </div>
  )
}

/* ─── Multi-Modal Choice Component ─────────── */
function MultimodalChoice({ zone, pricing, hour, minute, dayOfWeek }) {
  const [simulatedSurge, setSimulatedSurge] = useState(1.0)
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)

  // Reset simulatedSurge when zone or baseline pricing changes
  useEffect(() => {
    if (pricing) {
      setSimulatedSurge(pricing.surge_multiplier)
    } else {
      setSimulatedSurge(1.0)
    }
  }, [zone, pricing])

  // Fetch choice splits when parameters change
  useEffect(() => {
    if (!zone) return
    setLoading(true)
    apiFetch(`${API_BASE}/api/v1/multimodal/choice`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        zone_id: zone.zone_id,
        hour,
        minute,
        day_of_week: dayOfWeek,
        surge_multiplier: simulatedSurge,
        base_fare: 15.0,
      }),
    })
      .then(res => {
        if (res.ok) return res.json()
        throw new Error('Choice split fetch failed')
      })
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false))
  }, [zone, hour, minute, dayOfWeek, simulatedSurge])

  if (!data) return null

  const carPct = (data.probabilities.car * 100).toFixed(0)
  const autoPct = (data.probabilities.auto * 100).toFixed(0)
  const bikePct = (data.probabilities.bike * 100).toFixed(0)

  return (
    <div className="multimodal-section">
      <h4 className="explanation-title">🚲 Multi-Modal Choice splits</h4>
      <p className="explanation-subtitle">How will riders shift under simulated surge?</p>
      
      {/* Surge Simulator Slider */}
      <div className="sim-slider-container">
        <div className="sim-slider-label">
          <span>Surge Multiplier: <strong style={{ color: 'var(--accent-amber)' }}>{simulatedSurge.toFixed(2)}x</strong></span>
        </div>
        <input 
          type="range"
          min="1.0"
          max="3.0"
          step="0.1"
          value={simulatedSurge}
          onChange={(e) => setSimulatedSurge(parseFloat(e.target.value))}
          className="sim-slider"
        />
      </div>

      {/* Stacked Mode Share Bar */}
      <div className="mode-share-bar">
        <div className="mode-share-segment segment-car" style={{ width: `${carPct}%` }} title={`Car: ${carPct}%`}>
          {carPct >= 12 && `${carPct}%`}
        </div>
        <div className="mode-share-segment segment-auto" style={{ width: `${autoPct}%` }} title={`Auto: ${autoPct}%`}>
          {autoPct >= 12 && `${autoPct}%`}
        </div>
        <div className="mode-share-segment segment-bike" style={{ width: `${bikePct}%` }} title={`Bike: ${bikePct}%`}>
          {bikePct >= 12 && `${bikePct}%`}
        </div>
      </div>

      {/* Details List */}
      <div className="mode-list">
        <div className="mode-item">
          <div className="mode-label-group">
            <span className="mode-dot dot-car" />
            <span className="mode-name">Car (Ride-hail)</span>
          </div>
          <span className="mode-details">${data.costs.car.toFixed(2)} · {data.durations.car.toFixed(0)}m</span>
          <span className="mode-prob" style={{ color: 'var(--accent-indigo)' }}>{carPct}%</span>
        </div>
        <div className="mode-item">
          <div className="mode-label-group">
            <span className="mode-dot dot-auto" />
            <span className="mode-name">Auto (Medallion)</span>
          </div>
          <span className="mode-details">${data.costs.auto.toFixed(2)} · {data.durations.auto.toFixed(0)}m</span>
          <span className="mode-prob" style={{ color: 'var(--accent-amber)' }}>{autoPct}%</span>
        </div>
        <div className="mode-item">
          <div className="mode-label-group">
            <span className="mode-dot dot-bike" />
            <span className="mode-name">Bike (Citi Bike)</span>
          </div>
          <span className="mode-details">${data.costs.bike.toFixed(2)} · {data.durations.bike.toFixed(0)}m</span>
          <span className="mode-prob" style={{ color: 'var(--demand-low)' }}>{bikePct}%</span>
        </div>
      </div>
    </div>
  )
}

/* ─── Fleet Rebalancing Component ─────────── */
function RebalancePanel({ hour, minute, dayOfWeek }) {
  const [fleetSize, setFleetSize] = useState(500)
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const runRebalance = async () => {
    setLoading(true)
    setError('')
    try {
      const validFleetSize = Math.max(100, Math.min(5000, Number(fleetSize) || 500))
      const validMinute = [0, 15, 30, 45].reduce((prev, curr) => 
        Math.abs(curr - minute) < Math.abs(prev - minute) ? curr : prev
      )
      const res = await apiFetch(`${API_BASE}/api/v1/simulation/rebalance`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hour,
          minute: validMinute,
          day_of_week: dayOfWeek,
          fleet_size: validFleetSize
        })
      })
      if (!res.ok) throw new Error(`Simulation failed (${res.status})`)
      setData(await res.json())
    } catch (e) {
      setData(null)
      setError(e.message || 'Simulation unavailable')
    }
    setLoading(false)
  }

  return (
    <div className="rebalance-panel glass-card">
      <h3 className="section-title">
        <span>🤖 Fleet Rebalancing Optimizer</span>
        <span className="section-subtitle">Automated Relocation</span>
      </h3>

      <div className="rebalance-controls">
        <label className="rebalance-field">
          <span className="control-label">Virtual Fleet Size</span>
          <input
            type="number"
            min="100"
            max="3000"
            step="50"
            value={fleetSize}
            onChange={(e) => setFleetSize(parseInt(e.target.value) || 500)}
            className="rebalance-input"
          />
        </label>
        <button 
          onClick={runRebalance} 
          disabled={loading}
          className="rebalance-button"
        >
          {loading ? 'Optimizing...' : 'Run Simulation'}
        </button>
      </div>

      {error && <div className="rebalance-error">{error}</div>}

      {data && (
        <div className="simulation-results">
          {/* Metrics grid */}
          <div className="sim-metrics-grid">
            <div className="sim-metric-card">
              <span className="sim-metric-label">Relocations</span>
              <span className="sim-metric-value">{data.metrics.vehicles_relocated}</span>
              <span className="sim-metric-desc">vehicles moved</span>
            </div>
            <div className="sim-metric-card">
              <span className="sim-metric-label">Wait Saved</span>
              <span className="sim-metric-value">{data.metrics.wait_time_reduction_minutes.toFixed(1)}m</span>
              <span className="sim-metric-desc">passenger relief</span>
            </div>
            <div className="sim-metric-card">
              <span className="sim-metric-label">Deadhead Distance</span>
              <span className="sim-metric-value">{data.metrics.total_deadhead_miles.toFixed(0)} mi</span>
              <span className="sim-metric-desc">positioning cost</span>
            </div>
            <div className="sim-metric-card">
              <span className="sim-metric-label">Active Placement</span>
              <span className="sim-metric-value">{data.metrics.fleet_utilization_pct.toFixed(0)}%</span>
              <span className="sim-metric-desc">fleet utilization</span>
            </div>
          </div>

          {/* Relocation commands list */}
          <h4 className="orders-title">📍 Dispatch Transfer Orders (Top transfers)</h4>
          {data.orders.length === 0 ? (
            <div className="no-orders">Fleet configuration is balanced. No relocations needed.</div>
          ) : (
            <div className="orders-list">
              {data.orders.map((o, idx) => (
                <div key={idx} className="order-row">
                  <div className="order-path">
                    <span className="order-from">{o.from_zone_name}</span>
                    <span className="order-arrow">➔</span>
                    <span className="order-to">{o.to_zone_name}</span>
                  </div>
                  <div className="order-details">
                    <span className="badge badge-count">{o.vehicle_count} cabs</span>
                    <span className="order-distance">{o.distance_miles} mi ({o.duration_minutes.toFixed(0)}m)</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/* ─── Ripple Congestion Component ─────────── */
function RippleCongestionPanel({ zone, hour, minute, dayOfWeek }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const threshold = 0.50

  const activeZoneId = zone?.zone_id || 75 // Default to East Harlem South if no zone selected

  const fetchRipple = useCallback(async () => {
    setLoading(true)
    try {
      const res = await apiFetch(`${API_BASE}/api/v1/congestion/ripple`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          zone_id: activeZoneId,
          hour,
          minute,
          day_of_week: dayOfWeek,
          threshold: threshold
        })
      })
      if (res.ok) {
        setData(await res.json())
      }
    } catch (e) {
      console.error('Ripple fetch error:', e)
    }
    setLoading(false)
  }, [activeZoneId, hour, minute, dayOfWeek, threshold])

  useEffect(() => {
    fetchRipple()
  }, [fetchRipple])

  return (
    <div className="ripple-panel glass-card">
      <h3 className="section-title">
        <span>🌊 Interconnected Traffic Flow & Ripple Spillover</span>
        <span className="section-subtitle">50% Capacity Threshold</span>
      </h3>

      {data ? (
        <div className="ripple-content">
          <div className="ripple-header-card">
            <div className="ripple-root-info">
              <span className="ripple-root-title">Root Cell: {data.root_zone_name} ({data.borough})</span>
              <span className="ripple-load-metric">
                Capacity Load: <strong>{data.capacity_load_pct}%</strong>
              </span>
            </div>
            <span className={`badge badge-${data.severity_level}`}>
              {data.severity_level.toUpperCase()}
            </span>
          </div>

          <div className="ripple-advice-box">
            <span className="advice-icon">🚦</span>
            <span className="advice-text">{data.mitigation_advice}</span>
          </div>

          <div className="ripple-kpi-grid">
            <div className="ripple-kpi">
              <span className="ripple-kpi-val">{data.total_affected_zones}</span>
              <span className="ripple-kpi-lbl">Adjacent Cells Affected</span>
            </div>
            <div className="ripple-kpi">
              <span className="ripple-kpi-val">+{data.max_delay_added_minutes}m</span>
              <span className="ripple-kpi-lbl">Max Delay Added</span>
            </div>
          </div>

          {/* 1-Hop Direct Neighbors */}
          <h4 className="ripple-section-heading">1-Hop Direct Neighbors (Bordering Cells)</h4>
          {data.direct_neighbors_impacted.length === 0 ? (
            <div className="no-orders">No direct neighbor spillover below threshold ({data.threshold_used_pct}%).</div>
          ) : (
            <div className="neighbor-list">
              {data.direct_neighbors_impacted.map((n, idx) => (
                <div key={idx} className="neighbor-row">
                  <div className="neighbor-path">
                    <span className="neighbor-arrow">➔</span>
                    <span className="neighbor-name">{n.zone_name}</span>
                    <span className="neighbor-borough">({n.borough})</span>
                  </div>
                  <div className="neighbor-metrics">
                    <span className="badge badge-high">Spillover: {n.spillover_intensity_pct}%</span>
                    <span className="neighbor-delay">+{n.delay_added_minutes}m delay</span>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* 2-Hop Secondary Neighbors */}
          {data.secondary_neighbors_impacted.length > 0 && (
            <>
              <h4 className="ripple-section-heading" style={{ marginTop: '16px' }}>2-Hop Secondary Ripple (Downstream)</h4>
              <div className="neighbor-list">
                {data.secondary_neighbors_impacted.map((s, idx) => (
                  <div key={idx} className="neighbor-row secondary-row">
                    <div className="neighbor-path">
                      <span className="neighbor-arrow">↳</span>
                      <span className="neighbor-name">{s.zone_name}</span>
                    </div>
                    <div className="neighbor-metrics">
                      <span className="badge badge-moderate">Ripple: {s.spillover_intensity_pct}%</span>
                      <span className="neighbor-delay">+{s.delay_added_minutes}m delay</span>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      ) : (
        <div className="loading-state">{loading ? 'Simulating flow propagation...' : 'Select a zone to analyze ripple effect.'}</div>
      )}
    </div>
  )
}

/* ─── Main App ─────────────────────────────── */
function App() {
  const [health, setHealth] = useState(null)
  const [timeSlot, setTimeSlot] = useState(17 * 4)
  const [dayOfWeek, setDayOfWeek] = useState(2)
  const [heatmapData, setHeatmapData] = useState(null)
  const [selectedZone, setSelectedZone] = useState(null)
  const [pricing, setPricing] = useState(null)
  const [explanation, setExplanation] = useState(null)
  const [featureImportance, setFeatureImportance] = useState([])
  const [loading, setLoading] = useState(false)
  const [showAuth, setShowAuth] = useState(false)
  const [user, setUser] = useState(() => {
    try { return JSON.parse(localStorage.getItem('uf_user')) } catch { return null }
  })
  const { hour, minute } = getTimeParts(timeSlot)

  const authRequired = health ? Boolean(health.auth_enabled) : null
  const isAuthenticated = Boolean(user && localStorage.getItem('uf_access_token'))
  const canUseApi = health !== null && (!authRequired || isAuthenticated)

  const handleAuth = (userData) => {
    setUser(userData)
    setShowAuth(false)
  }

  const handleLogout = () => {
    localStorage.removeItem('uf_access_token')
    localStorage.removeItem('uf_refresh_token')
    localStorage.removeItem('uf_user')
    setUser(null)
  }

  // Health check
  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(r => r.json())
      .then(data => {
        setHealth(data)
        if (data.auth_enabled && !localStorage.getItem('uf_access_token')) {
          setShowAuth(true)
        }
      })
      .catch(() => setHealth({ status: 'error', model_loaded: false, zones_available: 0 }))
  }, [])

  // Feature importance (public endpoint; still uses apiFetch when auth is on)
  useEffect(() => {
    if (!canUseApi) return
    apiFetch(`${API_BASE}/api/v1/explain/feature-importance`)
      .then(res => {
        if (res.ok) return res.json()
        throw new Error('Feature importance fetch failed')
      })
      .then(data => setFeatureImportance(data.features || []))
      .catch((e) => console.error(e))
  }, [canUseApi])

  // Fetch heatmap when time changes (uses apiFetch for auth)
  const fetchHeatmap = useCallback(async () => {
    if (!canUseApi) return
    setLoading(true)
    try {
      const res = await apiFetch(`${API_BASE}/api/v1/demand/heatmap`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hour, minute, day_of_week: dayOfWeek, day_of_month: 15, month: 3 }),
      })
      if (res.ok) {
        const data = await res.json()
        setHeatmapData(data)
      } else {
        console.error('Heatmap fetch failed:', res.status)
      }
    } catch (e) {
      console.error('Heatmap fetch error:', e)
    }
    setLoading(false)
  }, [hour, minute, dayOfWeek, canUseApi])

  useEffect(() => {
    fetchHeatmap()
  }, [fetchHeatmap])

  // Fetch zone details when selected (uses apiFetch for auth)
  useEffect(() => {
    if (!canUseApi || !selectedZone) {
      setPricing(null)
      setExplanation(null)
      return
    }

    // Pricing
    apiFetch(`${API_BASE}/api/v1/pricing/calculate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        zone_id: selectedZone.zone_id,
        hour,
        minute,
        day_of_week: dayOfWeek,
        day_of_month: 15,
        month: 3,
        base_fare: 15.0,
      }),
    })
      .then(res => {
        if (res.ok) return res.json()
        throw new Error('Pricing failed')
      })
      .then(setPricing)
      .catch(() => setPricing(null))

    // SHAP explanation
    apiFetch(`${API_BASE}/api/v1/explain/zone/${selectedZone.zone_id}?hour=${hour}&minute=${minute}&day_of_week=${dayOfWeek}&day_of_month=15&month=3`)
      .then(res => {
        if (res.ok) return res.json()
        throw new Error('Explanation failed')
      })
      .then(setExplanation)
      .catch(() => setExplanation(null))
  }, [selectedZone, hour, minute, dayOfWeek, canUseApi])

  return (
    <div className="app">
      <Header health={health} user={user} onShowAuth={() => setShowAuth(true)} onLogout={handleLogout} />

      {showAuth && (
        <AuthModal
          onClose={() => { if (!authRequired || isAuthenticated) setShowAuth(false) }}
          onAuth={handleAuth}
        />
      )}

      <main className="main-content">
        {/* Time Controls */}
        <TimeControls
          timeSlot={timeSlot}
          setTimeSlot={setTimeSlot}
          dayOfWeek={dayOfWeek}
          setDayOfWeek={setDayOfWeek}
        />

        {/* Stats Overview */}
        <StatsCards heatmapData={heatmapData} />

        {/* Main Continuous 2-Column Grid */}
        <div className="content-grid">
          {/* Left Column: Spatial Heatmap, Ripple Flow & Fleet Optimizer */}
          <div className="content-left">
            <ZoneGrid
              zones={heatmapData?.zones}
              selectedZone={selectedZone}
              onSelectZone={setSelectedZone}
            />
            <RippleCongestionPanel
              zone={selectedZone}
              hour={hour}
              minute={minute}
              dayOfWeek={dayOfWeek}
            />
            <RebalancePanel
              hour={hour}
              minute={minute}
              dayOfWeek={dayOfWeek}
            />
          </div>

          {/* Right Column: Zone Detail, Top Leaderboard, ETAs, SHAP Drivers & Health */}
          <div className="content-right">
            <ZoneDetail
              zone={selectedZone}
              pricing={pricing}
              explanation={explanation}
              hour={hour}
              minute={minute}
              dayOfWeek={dayOfWeek}
            />
            <TopZones zones={heatmapData?.zones} />
            <EtaPanel
              zones={heatmapData?.zones}
              selectedZone={selectedZone}
              hour={hour}
              minute={minute}
              dayOfWeek={dayOfWeek}
            />
            <FeatureImportance features={featureImportance} />
            <SystemStatus health={health} user={user} />
          </div>
        </div>
      </main>

      <footer className="footer">
        <p>
          UrbanFlow v2.0 · XGBoost + LightGBM Ensemble · SHAP Explainability ·
          NYC TLC HVFHV 15-min Data · {heatmapData?.total_zones || 0} Zones
          {health?.database_available && ' · PostgreSQL'}
          {health?.eta_loaded && ' · ETA Engine'}
          {health?.auth_enabled && ' · JWT Auth'}
        </p>
      </footer>
    </div>
  )
}

export default App
