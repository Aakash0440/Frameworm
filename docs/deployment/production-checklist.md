# Production Deployment Checklist

Complete checklist for deploying FRAMEWORM to production.

---

## Pre-Deployment

### 1. Code Quality
- [ ] All tests passing (`pytest tests/`)
- [ ] Code coverage >90% (`pytest --cov`)
- [ ] Linting clean (`black`, `flake8`, `mypy`)
- [ ] No security vulnerabilities (`bandit frameworm/`)

### 2. Configuration
- [ ] Config externalized (environment variables)
- [ ] Secrets in vault (not in code)
- [ ] Logging configured for production
- [ ] Error tracking setup (Sentry, etc.)

### 3. Performance
- [ ] Load tested (wrk, locust)
- [ ] Memory profiled (no leaks)
- [ ] Database queries optimized
- [ ] Caching strategy implemented

---

## Deployment

### 1. Health Checks
```python
from frameworm.production import HealthChecker

health = HealthChecker()
health.add_liveness_check('basic', lambda: HealthStatus(True))
health.add_readiness_check('model', check_model_loaded)
health.add_readiness_check('database', check_db_connection)
```

### 2. Graceful Shutdown
```python
from frameworm.production import GracefulShutdown

shutdown = GracefulShutdown(timeout=30)

@shutdown.on_shutdown
def cleanup():
    save_final_checkpoint()
    close_database_connections()

shutdown.start()
```

### 3. Rate Limiting
```python
from frameworm.production import RateLimiter

limiter = RateLimiter(max_requests=100, window_seconds=60)

# In request handler
if not limiter.allow(user_id):
    return {'error': 'Rate limit exceeded'}, 429
```

### 4. Request Validation
```python
from frameworm.production import RequestValidator

validator = RequestValidator()
validator.add_rule('image', type=bytes, max_size_mb=10)
validator.add_rule('batch_size', type=int, min_value=1, max_value=128)

# Validate all requests
validator.validate(request_data)
```

### 5. Security
```python
from frameworm.production import APIKeyAuth

auth = APIKeyAuth()

# Create key for user
api_key = auth.create_key('user_id')

# Verify in requests
if not auth.verify_key(request.headers['X-API-Key']):
    return {'error': 'Unauthorized'}, 401
```

---

## Monitoring

### 1. Metrics
- Expose Prometheus metrics on `/metrics`
- Track: request rate, latency p50/p95/p99, error rate, GPU utilization

### 2. Logging
- Structured JSON logs
- Include: request_id, user_id, latency, status
- Forward to centralized logging (ELK, Datadog)

### 3. Tracing
```python
from frameworm.production import setup_tracing

tracer = setup_tracing('frameworm-api', endpoint='http://jaeger:4317')
```

### 4. Alerting
- CPU >80% for 5min
- Memory >90%
- Error rate >1%
- p99 latency >500ms

---

## Post-Deployment

### 1. Smoke Tests
- [ ] Health endpoints responding
- [ ] Sample inference succeeds
- [ ] Metrics being collected
- [ ] Logs flowing

### 2. Gradual Rollout
- Start with 5% traffic
- Monitor metrics for 30min
- Increase to 25%, 50%, 100%
- Rollback if errors spike

### 3. Monitoring
- Set up dashboard (Grafana)
- Configure alerts (PagerDuty)
- Review logs daily
- Weekly performance review

---

## Disaster Recovery

### 1. Backups
- Model checkpoints: Daily to S3
- Database: Continuous replication
- Configs: Version controlled in Git

### 2. Rollback Plan
```bash
# Rollback deployment
kubectl rollout undo deployment/frameworm-api

# Restore database
pg_restore --clean backup.sql

# Revert model version
frameworm registry promote vae-mnist 1.0.0 production
```

### 3. Incident Response
1. Acknowledge alert
2. Check metrics/logs
3. Apply fix or rollback
4. Post-mortem write-up