"""Enhanced rate limiting manager with smarter controls."""

import time
import asyncio
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_minute: int = 30
    requests_per_hour: int = 1000
    daily_token_limit: int = 50000
    max_concurrent_requests: int = 3
    base_delay_seconds: float = 2.0
    backoff_multiplier: float = 1.5


class EnhancedRateLimitManager:
    """Smart rate limiting with adaptive controls"""
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        
        # Tracking
        self.request_times: List[float] = []
        self.hourly_requests: List[float] = []
        self.daily_tokens_used: int = 0
        self.daily_reset_time: datetime = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Adaptive controls
        self.consecutive_failures: int = 0
        self.current_delay: float = self.config.base_delay_seconds
        self.last_success_time: float = time.time()
        
        # Concurrent request tracking
        self.active_requests: int = 0
        
        print(f"[RATE-LIMIT] Enhanced rate limiting initialized")
        print(f"[RATE-LIMIT] Limits: {self.config.requests_per_minute}/min, {self.config.requests_per_hour}/hour")
    
    async def acquire_request_slot(self, estimated_tokens: int = 500) -> bool:
        """Acquire a slot for making a request with smart throttling"""
        
        # Check daily token limit
        if self._is_daily_limit_exceeded(estimated_tokens):
            return False
        
        # Check concurrent request limit
        if self.active_requests >= self.config.max_concurrent_requests:
            await self._wait_for_concurrent_slot()
        
        # Check rate limits
        if self._should_throttle():
            delay = self._calculate_adaptive_delay()
            print(f"[RATE-LIMIT] Throttling request, waiting {delay:.1f}s")
            await asyncio.sleep(delay)
        
        # Record request
        now = time.time()
        self.request_times.append(now)
        self.hourly_requests.append(now)
        self.active_requests += 1
        
        # Cleanup old records
        self._cleanup_old_records()
        
        return True
    
    def release_request_slot(self, tokens_used: int = 0, success: bool = True):
        """Release a request slot and update tracking"""
        self.active_requests = max(0, self.active_requests - 1)
        self.daily_tokens_used += tokens_used
        
        if success:
            self.consecutive_failures = 0
            self.current_delay = max(self.config.base_delay_seconds, self.current_delay * 0.9)
            self.last_success_time = time.time()
        else:
            self.consecutive_failures += 1
            self.current_delay = min(30.0, self.current_delay * self.config.backoff_multiplier)
    
    def can_proceed_with_analysis(self, num_files: int, num_agents: int, avg_file_size: int) -> Tuple[bool, str]:
        """Smart analysis feasibility check"""
        
        # Reset daily counters if needed
        self._check_daily_reset()
        
        # Estimate total tokens needed
        estimated_tokens_per_file = self._estimate_tokens_per_file(avg_file_size, num_agents)
        total_estimated_tokens = num_files * estimated_tokens_per_file
        
        # Check daily token budget
        remaining_tokens = self.config.daily_token_limit - self.daily_tokens_used
        if total_estimated_tokens > remaining_tokens:
            return False, f"Estimated {total_estimated_tokens:,} tokens needed, only {remaining_tokens:,} remaining today"
        
        # Check if we need to be conservative
        if self.consecutive_failures > 2:
            return False, f"Recent API failures detected, using conservative approach"
        
        # Check time-based limits
        recent_requests = len([t for t in self.request_times if time.time() - t < 300])  # Last 5 minutes
        if recent_requests > 50:
            return False, f"High recent request volume, cooling down"
        
        return True, "Analysis can proceed"
    
    def get_fallback_strategy(self, requested_agents: List[str]) -> Tuple[List[str], str]:
        """Get fallback agent selection when limits are hit"""
        
        # Priority order for agents when resources are limited
        agent_priority = {
            'security': 1,      # Highest priority
            'complexity': 2,    
            'performance': 3,   
            'documentation': 4, 
            'duplication': 5    # Lowest priority
        }
        
        # Sort by priority
        sorted_agents = sorted(requested_agents, key=lambda x: agent_priority.get(x, 10))
        
        # Conservative selection based on current state
        if self.consecutive_failures > 0:
            # Very conservative - only top 2 agents
            fallback_agents = sorted_agents[:2]
            strategy = "Conservative: Using only highest priority agents due to recent issues"
        elif self.daily_tokens_used > self.config.daily_token_limit * 0.8:
            # Moderate - top 3 agents
            fallback_agents = sorted_agents[:3]
            strategy = "Moderate: Limiting agents to preserve daily token budget"
        else:
            # Standard fallback - most agents except lowest priority
            fallback_agents = sorted_agents[:-1] if len(sorted_agents) > 3 else sorted_agents
            strategy = "Standard: Using most requested agents with one omitted"
        
        return fallback_agents, strategy
    
    def get_conservative_limits(self) -> Dict[str, int]:
        """Get conservative limits when things are going wrong"""
        return {
            'max_parallel_files': 1 if self.consecutive_failures > 2 else 2,
            'max_agents_per_file': 2 if self.consecutive_failures > 2 else 3,
            'delay_between_files': 10.0 if self.consecutive_failures > 2 else 5.0
        }
    
    def _is_daily_limit_exceeded(self, estimated_tokens: int) -> bool:
        """Check if daily token limit would be exceeded"""
        self._check_daily_reset()
        return (self.daily_tokens_used + estimated_tokens) > self.config.daily_token_limit
    
    def _should_throttle(self) -> bool:
        """Determine if we should throttle based on recent activity"""
        now = time.time()
        
        # Check requests per minute
        recent_minute = [t for t in self.request_times if now - t < 60]
        if len(recent_minute) >= self.config.requests_per_minute:
            return True
        
        # Check requests per hour  
        recent_hour = [t for t in self.hourly_requests if now - t < 3600]
        if len(recent_hour) >= self.config.requests_per_hour:
            return True
        
        # Adaptive throttling based on failures
        if self.consecutive_failures > 0:
            time_since_last_success = now - self.last_success_time
            if time_since_last_success < (self.consecutive_failures * 10):
                return True
        
        return False
    
    def _calculate_adaptive_delay(self) -> float:
        """Calculate smart delay based on current conditions"""
        base_delay = self.current_delay
        
        # Increase delay based on recent activity
        recent_requests = len([t for t in self.request_times if time.time() - t < 120])
        if recent_requests > 10:
            base_delay *= 1.5
        
        # Add jitter to avoid thundering herd
        import random
        jitter = random.uniform(0.8, 1.2)
        
        return base_delay * jitter
    
    async def _wait_for_concurrent_slot(self):
        """Wait for an available concurrent request slot"""
        wait_time = 0
        while self.active_requests >= self.config.max_concurrent_requests and wait_time < 30:
            await asyncio.sleep(1)
            wait_time += 1
        
        if wait_time >= 30:
            print("[RATE-LIMIT] WARNING: Waited 30s for concurrent slot, proceeding anyway")
    
    def _estimate_tokens_per_file(self, file_size: int, num_agents: int) -> int:
        """Estimate tokens needed per file"""
        # Base estimate: ~4 characters per token, plus prompt overhead
        base_tokens = (file_size // 4) + 500  # File content + prompt
        return base_tokens * num_agents * 1.2  # Agents + 20% overhead
    
    def _cleanup_old_records(self):
        """Clean up old timing records"""
        now = time.time()
        self.request_times = [t for t in self.request_times if now - t < 3600]  # Keep last hour
        self.hourly_requests = [t for t in self.hourly_requests if now - t < 3600]
    
    def _check_daily_reset(self):
        """Check if we need to reset daily counters"""
        now = datetime.now()
        if now.date() > self.daily_reset_time.date():
            self.daily_tokens_used = 0
            self.daily_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            print(f"[RATE-LIMIT] Daily counters reset")
    
    def get_status(self) -> Dict:
        """Get current rate limiting status"""
        now = time.time()
        recent_minute = len([t for t in self.request_times if now - t < 60])
        recent_hour = len([t for t in self.hourly_requests if now - t < 3600])
        
        return {
            'requests_last_minute': recent_minute,
            'requests_last_hour': recent_hour,
            'daily_tokens_used': self.daily_tokens_used,
            'daily_tokens_remaining': self.config.daily_token_limit - self.daily_tokens_used,
            'active_requests': self.active_requests,
            'consecutive_failures': self.consecutive_failures,
            'current_delay': self.current_delay
        }


# Global instance
rate_limit_manager = EnhancedRateLimitManager()