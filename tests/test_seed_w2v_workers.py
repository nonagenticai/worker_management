"""Tests for the canonical w2v worker seed YAML.

These tests guard against regressions in the seed file (e.g. someone
accidentally re-introducing the validator-bypass plain-English version
of course-curriculum-generator's prompt).
"""

from __future__ import annotations

from pathlib import Path

import yaml

SEED = Path(__file__).resolve().parent.parent / "seed" / "w2v_workers.yaml"


def test_seed_yaml_parses() -> None:
    spec = yaml.safe_load(SEED.read_text())
    assert isinstance(spec, dict)
    assert "workers" in spec
    assert isinstance(spec["workers"], list)
    assert spec["workers"], "seed must declare at least one worker"


def test_seed_yaml_covers_all_workers() -> None:
    spec = yaml.safe_load(SEED.read_text())
    names = {w["worker_name"] for w in spec["workers"]}
    assert names == {
        "brand-analysis",
        "analysis-site-summary",
        "icp-analysis",
        "course-curriculum-generator",
        # course_outline stage (course-outline) — distinct from the Phase-7
        # w2v_curriculum_outline; carries a larger max_tokens (see below).
        "course-outline",
        "slide-bullet-extractor",
        "edit-plan-generator",
        # Phase 7 — w2v_orchestrator curriculum_design + lesson_summarize.
        # Names ride underscores (not hyphens) to match the worker_name
        # constants baked into w2v_orchestrator's stage handlers.
        "w2v_curriculum_audience",
        "w2v_curriculum_outline",
        "w2v_curriculum_syllabus",
        "w2v_lesson_summarize",
        # html_render_service redesign workers (worker_name constants baked into
        # html_generator.py / accessibility_scorer.py).
        "redesign-html",
        "redesign-accessibility-scorer",
        # ORCH-1.2 script-doctor evaluator-optimizer loop (worker_name constants
        # baked into w2v_orchestrator src/services/stages/shared/script_doctor.py:
        # _CRITIQUE_WORKER / _REVISE_WORKER).
        "script-doctor-critique",
        "script-doctor-revise",
        # ORCH-1.3b retention_plan evaluator-optimizer loop (worker_name
        # constants referenced verbatim by w2v_orchestrator's retention_plan
        # stage: retention-critic evaluator / hook-doctor optimizer).
        "retention-critic",
        "hook-doctor",
        # ORCH-4.4 (A18) claim_finish evaluator-optimizer loop (worker_name
        # constants referenced verbatim by w2v_orchestrator's claim_finish
        # stage: claim-grounder evaluator / claim-softener optimizer).
        "claim-grounder",
        "claim-softener",
        # ORCH-1.4 voice-performance director (worker_name constant referenced
        # verbatim by w2v_orchestrator's demo_video voice_director stage).
        "voice-director",
        # A19 accessibility loop — a11y-critic ranks deterministic WCAG/silent-gap
        # remediations; audio-describer writes AD lines for narration-free gaps.
        "a11y-critic",
        "audio-describer",
        # S19/S20 encode_master evaluator-optimizer loop (worker_name constants
        # referenced verbatim by w2v_orchestrator's demo_video_encode_master
        # stage: encode-critic evaluator / mux-mastering-advisor optimizer).
        "encode-critic",
        "mux-mastering-advisor",
        # Marketing / audit "great report" enrichment workers (marketing-docs PR2-4);
        # called by analysis_service scorecards/{recommendation_writer,report_narrative,
        # claim_grounding,report_critic}.py with authoritative system_prompt+schema per call.
        "audit-recommendation-writer",
        "audit-exec-summary",
        "audit-claim-grounding",
        "audit-report-critic",
        # analysis_service critique workers (ux_reviewer.py / report_generator.py),
        # pinned to temperature 0 so a re-critique of the same page is repeatable.
        "critique-ux-review",
        "critique-summary",
        # Audiobook great-output workers — w2v_orchestrator's audiobook DAG calls
        # these worker_name constants verbatim (direct.py / ingest.py / qa.py) and
        # lists them in SEED_MANAGED, so this seed owns their model + system_prompt.
        "audiobook-director",
        "pronunciation-proposer",
        "audiobook-listenback-judge",
    }


def test_seed_yaml_includes_voice_director_worker() -> None:
    """ORCH-1.4 — the voice-performance director must be present and route via
    inference_router. w2v_orchestrator's demo_video voice_director stage calls
    worker_name verbatim, so a typo, rename, or missing seed here drops the call
    to WM auto-create, which serves a gemma-led prose default and breaks the
    structured per-line directive contract the stage's coercion helpers expect.
    """
    spec = yaml.safe_load(SEED.read_text())
    by_name = {w["worker_name"]: w for w in spec["workers"]}
    assert "voice-director" in by_name, "voice-director missing from seed"
    worker = by_name["voice-director"]
    assert worker["model_provider"] == "inference_router"
    assert worker["agent_type"] == "pydantic_ai"
    assert worker["system_prompt"].strip(), "voice-director has empty system_prompt"


def test_seed_yaml_includes_claim_finish_workers() -> None:
    """ORCH-4.4 (A18) — the claim_finish grounder/softener loop workers must be
    present and route via inference_router. w2v_orchestrator's claim_finish stage
    calls worker_name verbatim (claim-grounder / claim-softener) and lists them in
    SEED_MANAGED, so the register script skips them and this seed must own them; a
    typo, rename, or missing seed here drops the calls to WM auto-create, which
    serves a gemma-led prose default and breaks the structured grounding loop.
    """
    spec = yaml.safe_load(SEED.read_text())
    by_name = {w["worker_name"]: w for w in spec["workers"]}
    for name in ("claim-grounder", "claim-softener"):
        assert name in by_name, f"claim_finish worker {name} missing from seed"
        worker = by_name[name]
        assert worker["model_provider"] == "inference_router", name
        assert worker["agent_type"] == "pydantic_ai", name
        assert worker["system_prompt"].strip(), f"{name} has empty system_prompt"


def test_seed_yaml_includes_phase_seven_curriculum_workers() -> None:
    """Phase 7 — the four curriculum_design / lesson_summarize workers
    must be present and reference inference_router. w2v_orchestrator's
    stage handlers call worker_name verbatim, so a typo or rename here
    breaks the whole curriculum_design + training-course (lesson) DAG.
    """
    spec = yaml.safe_load(SEED.read_text())
    p7_names = {
        "w2v_curriculum_audience",
        "w2v_curriculum_outline",
        "w2v_curriculum_syllabus",
        "w2v_lesson_summarize",
    }
    by_name = {w["worker_name"]: w for w in spec["workers"]}
    for name in p7_names:
        assert name in by_name, f"Phase 7 worker {name} missing from seed"
        worker = by_name[name]
        assert worker["model_provider"] == "inference_router", name
        assert worker["agent_type"] == "pydantic_ai", name
        assert worker["system_prompt"].strip(), f"{name} has empty system_prompt"


def test_seed_yaml_includes_script_doctor_workers() -> None:
    """ORCH-1.2 — the script-doctor critique/revise loop workers must be
    present and route via inference_router. w2v_orchestrator's
    ScriptDoctorStage calls worker_name verbatim (_CRITIQUE_WORKER /
    _REVISE_WORKER); a typo, rename, or missing seed here drops the calls to
    WM auto-create, which serves a gemma-led prose default and breaks the
    structured critique -> revise loop.
    """
    spec = yaml.safe_load(SEED.read_text())
    by_name = {w["worker_name"]: w for w in spec["workers"]}
    for name in ("script-doctor-critique", "script-doctor-revise"):
        assert name in by_name, f"script-doctor worker {name} missing from seed"
        worker = by_name[name]
        assert worker["model_provider"] == "inference_router", name
        assert worker["agent_type"] == "pydantic_ai", name
        assert worker["system_prompt"].strip(), f"{name} has empty system_prompt"


def test_seed_yaml_includes_retention_workers() -> None:
    """ORCH-1.3b — the retention_plan critic/optimizer loop workers must be
    present and route via inference_router. w2v_orchestrator's retention_plan
    stage calls worker_name verbatim (retention-critic / hook-doctor) and lists
    them in SEED_MANAGED, so the register script skips them and this seed must
    own them; a typo, rename, or missing seed here drops the calls to WM
    auto-create, which serves a gemma-led prose default and breaks the
    structured retention loop.
    """
    spec = yaml.safe_load(SEED.read_text())
    by_name = {w["worker_name"]: w for w in spec["workers"]}
    for name in ("retention-critic", "hook-doctor"):
        assert name in by_name, f"retention worker {name} missing from seed"
        worker = by_name[name]
        assert worker["model_provider"] == "inference_router", name
        assert worker["agent_type"] == "pydantic_ai", name
        assert worker["model"] == "local-llm", name
        assert worker["system_prompt"].strip(), f"{name} has empty system_prompt"


def test_seed_yaml_includes_audiobook_workers() -> None:
    """Audiobook great-output — the director / pronunciation-proposer /
    listen-back-judge workers must be present and route via inference_router on
    the strong `local-llm` model. w2v_orchestrator's audiobook stages call these
    worker_name constants verbatim (direct.py / ingest.py / qa.py) and list them
    in SEED_MANAGED, so the register script skips them and this seed must own
    them; a typo, rename, or missing seed here drops the calls to WM auto-create,
    which serves a gemma-led prose default and breaks the structured
    directive / pronunciation / listen-back-judgment contracts.
    """
    spec = yaml.safe_load(SEED.read_text())
    by_name = {w["worker_name"]: w for w in spec["workers"]}
    for name in ("audiobook-director", "pronunciation-proposer", "audiobook-listenback-judge"):
        assert name in by_name, f"audiobook worker {name} missing from seed"
        worker = by_name[name]
        assert worker["model_provider"] == "inference_router", name
        assert worker["model"] == "local-llm", name
        assert worker["agent_type"] == "pydantic_ai", name
        assert worker["system_prompt"].strip(), f"{name} has empty system_prompt"


def test_critique_workers_are_deterministic() -> None:
    """analysis_service's critique workers must sample at temperature 0.

    WMClient.message carries no temperature and WM's MessageRequest forbids extra
    fields, so the seeded row is the only lever. At 0.1 the same site scored UX
    59.3 / 10 findings vs 52.4 / 12 on the same day. A non-zero value here (or a
    dropped key, which falls back to the 0.1 default) reintroduces that drift.
    """
    spec = yaml.safe_load(SEED.read_text())
    by_name = {w["worker_name"]: w for w in spec["workers"]}
    for name in ("critique-ux-review", "critique-summary"):
        assert name in by_name, f"critique worker {name} missing from seed"
        worker = by_name[name]
        assert "temperature" in worker, name
        assert worker["temperature"] == 0, name
        # Under WorkerCreateRequest's 16384 cap and the ~12069 self-hosted clamp.
        assert 0 < worker["max_tokens"] <= 12069, name


def test_redesign_html_carries_max_tokens() -> None:
    """redesign-html emits a full Tailwind page; the provider default (~4k)
    truncates it mid-markup.

    The 16000 cap used to be applied by w2v_orchestrator's
    scripts/register_w2v_workers.py (WORKER_MAX_TOKENS), which blind-PUT this
    worker on every run. That script now lists redesign-html in SEED_MANAGED and
    skips it (so the curated system_prompt below survives), which means the seed
    is the ONLY thing that can still apply the cap. Dropping this key silently
    reverts redesign-html to truncated output.
    """
    spec = yaml.safe_load(SEED.read_text())
    by_name = {w["worker_name"]: w for w in spec["workers"]}
    assert by_name["redesign-html"]["max_tokens"] == 16000
    # WorkerCreateRequest.validate_max_tokens caps at 16384.
    assert by_name["redesign-html"]["max_tokens"] <= 16384


def test_course_outline_carries_max_tokens() -> None:
    """course-outline outlines the whole course flow; at the provider default
    (~4k) WM 500s with "Model token limit (4000) exceeded" (the E2E `course`
    scenario). course-outline is NOT in w2v_orchestrator's register script
    W2V_WORKER_NAMES, so WM auto-created it at the 4k default; this seed is the
    only thing that applies the larger cap. Dropping this key reverts the 500.
    """
    spec = yaml.safe_load(SEED.read_text())
    by_name = {w["worker_name"]: w for w in spec["workers"]}
    assert "course-outline" in by_name, "course-outline missing from seed"
    assert by_name["course-outline"]["max_tokens"] == 8000
    # WorkerCreateRequest.validate_max_tokens caps at 16384.
    assert by_name["course-outline"]["max_tokens"] <= 16384


def test_seed_yaml_has_required_fields() -> None:
    spec = yaml.safe_load(SEED.read_text())
    for w in spec["workers"]:
        name = w["worker_name"]
        assert w["agent_type"] == "pydantic_ai", name
        assert w["model_provider"] == "inference_router", name
        # Abstract, model-agnostic name. inference-router maps `local-llm` to the
        # pgx-local vLLM provider; the actual model is swappable without touching
        # this seed.
        assert w["model"] == "local-llm", name
        assert isinstance(w["temperature"], (int, float)), name
        assert 0.0 <= w["temperature"] <= 2.0, name
        assert w["agent_role"], f"{name} missing agent_role"
        assert w["system_prompt"].strip(), f"{name} has empty system_prompt"


def test_course_curriculum_prompt_contains_backtick_field_refs() -> None:
    """The canonical prompt MUST use backtick-wrapped field references.

    A plain-English rewrite ("the title field" instead of "the `title` field")
    is the validator-bypass form from before commit 7fae8ac. This test fails
    loudly if anyone re-introduces it.
    """
    spec = yaml.safe_load(SEED.read_text())
    w = next(w for w in spec["workers"] if w["worker_name"] == "course-curriculum-generator")
    prompt = w["system_prompt"]
    assert "`" in prompt, (
        "course-curriculum-generator system_prompt must use backtick-wrapped "
        "field references (the validator fix permits this; bypassing to "
        "plain-English is a regression)"
    )
    # Spot-check a few field names that the canonical prompt wraps in backticks
    for field_ref in ("`title`", "`description`", "`lessons`"):
        assert field_ref in prompt, f"canonical prompt is missing {field_ref}"


def test_seed_yaml_validates_against_worker_create_request() -> None:
    """Every seeded worker must round-trip cleanly through WorkerCreateRequest.

    This is the same validation the live PUT endpoint performs, so a green
    test here means the seeder will not 422 against a server running the
    validator-fix.
    """
    from src.api.workers.models import WorkerCreateRequest

    spec = yaml.safe_load(SEED.read_text())
    for w in spec["workers"]:
        WorkerCreateRequest.model_validate(w)
