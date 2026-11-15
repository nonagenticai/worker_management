"""Agent Host Registry for discovering and routing to agent-host instances.

Instances (external launchd hosts; routed via Istio ingress on 443).
URLs come from Settings (``AGENT_HOST_MINI{1,2}_URL`` env vars on the
Deployment); empty values are filtered out.
"""

import asyncio
import logging
from dataclasses import dataclass

import httpx
from cachetools import TTLCache

from src.settings import get_settings

logger = logging.getLogger(__name__)


def _settings_agent_host_cache_ttl() -> int:
    return get_settings().agent_host_cache_ttl


def _build_agent_host_instances() -> dict[str, str]:
    """Build the {label: url} routing dict from Settings each call."""
    s = get_settings()
    candidates: list[tuple[str, str]] = [
        ("mini1", s.agent_host_mini1_url),
        ("mini2", s.agent_host_mini2_url),
    ]
    return {name: url for name, url in candidates if url}


@dataclass
class AgentHostInstance:
    """One agent host (mini1/mini2/etc.) tracked by the registry."""

    name: str
    url: str
    healthy: bool = False
    active_containers: int = 0
    max_containers: int = 10


class AgentHostRegistry:
    """Registry for agent-host instances with health checking and round-robin routing."""

    def __init__(self) -> None:
        self._health_cache: TTLCache[str, AgentHostInstance] = TTLCache(
            maxsize=10, ttl=_settings_agent_host_cache_ttl()
        )
        self._round_robin_idx: int = 0
        self._lock = asyncio.Lock()

    async def check_instance_health(self, name: str, url: str) -> AgentHostInstance:
        """Check health of a single agent-host instance."""
        instance = AgentHostInstance(name=name, url=url)
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{url}/health")
                if response.status_code == 200:
                    data = response.json()
                    instance.healthy = data.get("status") == "ok"
                    instance.active_containers = data.get("active_containers", 0)
                    instance.max_containers = data.get("max_containers", 10)
                    logger.debug(
                        "agent-host %s: healthy=%s, containers=%d/%d",
                        name,
                        instance.healthy,
                        instance.active_containers,
                        instance.max_containers,
                    )
                else:
                    logger.warning("agent-host %s returned HTTP %d", name, response.status_code)
        except httpx.ConnectError:
            logger.warning("agent-host %s unreachable at %s", name, url)
        except httpx.TimeoutException:
            logger.warning("agent-host %s timed out at %s", name, url)
        except Exception:
            logger.exception("agent-host %s health check error", name)

        self._health_cache[name] = instance
        return instance

    async def refresh(self) -> list[AgentHostInstance]:
        """Refresh health status of all instances concurrently."""
        instances = _build_agent_host_instances()
        tasks = [self.check_instance_health(name, url) for name, url in instances.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, AgentHostInstance)]

    async def get_healthy_instances(self) -> list[AgentHostInstance]:
        """Get all healthy instances, refreshing if cache is stale."""
        # Check cache first
        instances = _build_agent_host_instances()
        cached = [self._health_cache[name] for name in instances if name in self._health_cache]
        if cached:
            healthy = [i for i in cached if i.healthy]
            if healthy:
                return healthy

        # Refresh and return healthy ones
        all_instances = await self.refresh()
        return [i for i in all_instances if i.healthy]

    async def get_best_instance(self) -> AgentHostInstance | None:
        """Get the best instance using round-robin among healthy instances.

        Prefers instances with fewer active containers.
        """
        healthy = await self.get_healthy_instances()
        if not healthy:
            logger.error("No healthy agent-host instances available")
            return None

        # Sort by active container count (least loaded first), then round-robin
        healthy.sort(key=lambda i: i.active_containers)

        async with self._lock:
            idx = self._round_robin_idx % len(healthy)
            self._round_robin_idx += 1

        instance = healthy[idx]
        logger.info(
            "Selected agent-host %s (%s) with %d/%d containers",
            instance.name,
            instance.url,
            instance.active_containers,
            instance.max_containers,
        )
        return instance


# Global singleton
agent_host_registry = AgentHostRegistry()
