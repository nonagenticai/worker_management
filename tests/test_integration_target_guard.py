"""Unit tests for the production-target guard used by the integration suites."""

import pytest

from tests.integration.target_guard import (
    ALLOW_PROD_VAR,
    ProdTargetRefused,
    assert_not_prod,
    integration_target,
    is_prod_host,
    refusal,
)


@pytest.mark.parametrize(
    "host",
    [
        # DNS zones
        "procedures.nonagentic.ai",
        "workers.nonagentic.ai",
        "postgres.nonagentic.ai",
        "auth.nonagentic.ai",
        "nonagentic.ai",
        "pgbouncer.pgbouncer.svc.cluster.local",
        "procedures.procedures.svc",
        "10-42-1-7.default.pod.cluster.local",
        "WORKERS.INTERNAL.",
        # short in-cluster names (resolve through a pod's search path)
        "postgres",
        "pgbouncer",
        "pgbouncer.pgbouncer",
        "workers.workers",
        # k3s pod + service CIDRs
        "10.42.0.1",
        "10.42.255.254",
        "10.43.0.10",
        "10.43.230.219",
        # tailnet 100.64.0.0/10 (CGNAT range)
        "100.100.100.100",
        "100.100.100.101",
        "100.64.0.1",
        "100.127.255.254",
        "[fd7a:115c:a1e0::1]",
        "::ffff:100.100.100.100",
        # node public IPs
        "203.0.113.10",
        "203.0.113.11",
        "203.0.113.12",
        "203.0.113.13",
        "203.0.113.14",
        "203.0.113.15",
    ],
)
def test_prod_hosts_are_recognised(host):
    assert is_prod_host(host)


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "127.0.0.1",
        "::1",
        "[::1]",
        "api.notnonagentic.ai",
        "mini3.local",
        "db.test",
        "app.localhost",
        "10.41.255.255",  # just below the pod CIDR
        "10.44.0.0",  # just above the service CIDR
        "100.63.255.255",  # just below the tailnet
        "100.128.0.0",  # just above the tailnet
        "192.168.1.110",
        "203.0.113.99",  # a neighbouring host, not the blocked one
    ],
)
def test_local_hosts_are_not_prod(host):
    assert not is_prod_host(host)


def test_unset_target_means_skip_never_a_prod_default():
    assert integration_target("SOME_URL", env={}) is None
    assert integration_target("SOME_URL", env={"SOME_URL": "  "}) is None


def test_local_target_is_returned():
    env = {"SOME_URL": "http://localhost:8400"}
    assert integration_target("SOME_URL", env=env) == "http://localhost:8400"


def test_prod_target_is_refused():
    with pytest.raises(ProdTargetRefused, match=r"production host 'procedures\.nonagentic\.ai'"):
        integration_target("SOME_URL", env={"SOME_URL": "https://procedures.nonagentic.ai"})


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql+psycopg://u:p@pgbouncer.pgbouncer.svc.cluster.local:6432/db",
        "postgresql://u:p@100.100.100.100:5432/db",
        "postgresql://u:p@postgres:5432/db",
        "postgresql://u:p@10.43.12.34:6432/db",
        "postgresql://u:p@[fd7a:115c:a1e0::5]:5432/db",
        "postgresql://u:p@localhost:5432,postgres.nonagentic.ai:5432/db",
        "postgresql://u:p@/db?host=pgbouncer.pgbouncer",
        "host=postgres.nonagentic.ai port=5432 dbname=db",
        "host=localhost,203.0.113.10 dbname=db",
    ],
)
def test_prod_database_urls_are_refused_in_url_and_keyword_form(dsn):
    with pytest.raises(ProdTargetRefused, match="production host"):
        assert_not_prod("DATABASE_URL", dsn, env={})


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql+psycopg://u:p@127.0.0.1:5442/db",
        "postgresql://u:p@[::1]:5432/db",
        "postgresql://u:p@/db?host=/var/run/postgresql",
        "host=localhost port=5432 dbname=db",
    ],
)
def test_local_database_urls_pass(dsn):
    assert_not_prod("DATABASE_URL", dsn, env={})


def test_unparseable_target_is_refused_not_assumed_local():
    reason = refusal("X", "not a url", env={})
    assert reason is not None
    assert "no parseable host" in reason


def test_only_exactly_1_lifts_the_refusal():
    url = "https://workers.nonagentic.ai"
    assert refusal("X", url, env={ALLOW_PROD_VAR: "true"}) is not None
    assert refusal("X", url, env={ALLOW_PROD_VAR: "1"}) is None
