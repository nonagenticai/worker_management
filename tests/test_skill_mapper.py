"""Tests for src/utils/skill_mapper.py - skill mapping utilities."""

import pytest

from src.utils.skill_mapper import (
    AGENT_TYPE_LEVEL_MAPPING,
    CAPABILITY_SKILL_MAPPINGS,
    TOOL_SKILL_MAPPINGS,
    Skill,
    adjust_skill_levels_by_agent_type,
    infer_skill_level,
    map_capabilities_to_skills,
    map_tools_to_skills,
    merge_skill_lists,
)

# ============================================================================
# Skill Model Tests
# ============================================================================


class TestSkillModel:
    """Tests for the Skill pydantic model."""

    def test_skill_creation_with_all_fields(self):
        skill = Skill(name="coding", level="expert", domains=["python", "backend"])
        assert skill.name == "coding"
        assert skill.level == "expert"
        assert skill.domains == ["python", "backend"]

    def test_skill_creation_with_default_domains(self):
        skill = Skill(name="testing", level="intermediate")
        assert skill.name == "testing"
        assert skill.level == "intermediate"
        assert skill.domains == []

    def test_skill_creation_missing_name_raises(self):
        with pytest.raises(Exception):
            Skill(level="advanced")

    def test_skill_creation_missing_level_raises(self):
        with pytest.raises(Exception):
            Skill(name="coding")

    def test_skill_serialization(self):
        skill = Skill(name="coding", level="expert", domains=["python"])
        data = skill.model_dump()
        assert data == {"name": "coding", "level": "expert", "domains": ["python"]}

    def test_skill_empty_domains_list(self):
        skill = Skill(name="coding", level="beginner", domains=[])
        assert skill.domains == []


# ============================================================================
# map_tools_to_skills Tests
# ============================================================================


class TestMapToolsToSkills:
    """Tests for map_tools_to_skills function."""

    def test_known_tool_returns_correct_skill(self):
        result = map_tools_to_skills(["file_system"])
        assert len(result) == 1
        assert result[0]["name"] == "file_operations"
        assert result[0]["level"] == "intermediate"
        assert "filesystem" in result[0]["domains"]
        assert "io" in result[0]["domains"]

    def test_code_analyzer_maps_correctly(self):
        result = map_tools_to_skills(["code_analyzer"])
        assert len(result) == 1
        assert result[0]["name"] == "code_analysis"
        assert result[0]["level"] == "advanced"
        assert "python" in result[0]["domains"]
        assert "ast" in result[0]["domains"]

    def test_test_runner_maps_correctly(self):
        result = map_tools_to_skills(["test_runner"])
        assert len(result) == 1
        assert result[0]["name"] == "testing"
        assert result[0]["level"] == "intermediate"
        assert "pytest" in result[0]["domains"]

    def test_unknown_tool_returns_default_with_tool_name(self):
        result = map_tools_to_skills(["some_unknown_tool"])
        assert len(result) == 1
        assert result[0]["name"] == "some_unknown_tool"
        assert result[0]["level"] == "intermediate"
        assert result[0]["domains"] == ["general"]

    def test_deduplication_file_system_and_file_reader(self):
        """file_system and file_reader both map to file_operations; should deduplicate."""
        result = map_tools_to_skills(["file_system", "file_reader"])
        assert len(result) == 1
        assert result[0]["name"] == "file_operations"

    def test_deduplication_test_runner_and_coverage_analyzer(self):
        """test_runner and coverage_analyzer both map to testing; should deduplicate."""
        result = map_tools_to_skills(["test_runner", "coverage_analyzer"])
        assert len(result) == 1
        assert result[0]["name"] == "testing"

    def test_deduplication_keeps_first_occurrence(self):
        """When deduplicating, the first mapping encountered is kept."""
        result = map_tools_to_skills(["file_system", "file_reader"])
        # file_system is first, so its domains should be kept
        assert result[0]["domains"] == ["filesystem", "io"]

    def test_multiple_distinct_tools(self):
        result = map_tools_to_skills(["file_system", "code_analyzer", "test_runner"])
        assert len(result) == 3
        names = [s["name"] for s in result]
        assert "file_operations" in names
        assert "code_analysis" in names
        assert "testing" in names

    def test_empty_list_returns_empty(self):
        result = map_tools_to_skills([])
        assert result == []

    def test_mixed_known_and_unknown_tools(self):
        result = map_tools_to_skills(["file_system", "unknown_tool_xyz"])
        assert len(result) == 2
        names = [s["name"] for s in result]
        assert "file_operations" in names
        assert "unknown_tool_xyz" in names

    def test_returns_copies_not_originals(self):
        """Returned dicts should be copies so mutations don't affect the mapping."""
        result = map_tools_to_skills(["file_system"])
        result[0]["name"] = "mutated"
        # Original mapping should be unaffected
        assert TOOL_SKILL_MAPPINGS["file_system"]["name"] == "file_operations"

    def test_all_tool_mappings_have_required_keys(self):
        """Every entry in TOOL_SKILL_MAPPINGS should have name, level, domains."""
        for tool_name, skill_data in TOOL_SKILL_MAPPINGS.items():
            assert "name" in skill_data, f"{tool_name} missing 'name'"
            assert "level" in skill_data, f"{tool_name} missing 'level'"
            assert "domains" in skill_data, f"{tool_name} missing 'domains'"


# ============================================================================
# map_capabilities_to_skills Tests
# ============================================================================


class TestMapCapabilitiesToSkills:
    """Tests for map_capabilities_to_skills function."""

    def test_known_capability_maps_correctly(self):
        result = map_capabilities_to_skills(["code_implementation"])
        assert len(result) == 1
        assert result[0]["name"] == "coding"
        assert result[0]["level"] == "advanced"
        assert "python" in result[0]["domains"]

    def test_debugging_capability(self):
        result = map_capabilities_to_skills(["debugging"])
        assert len(result) == 1
        assert result[0]["name"] == "debugging"
        assert result[0]["level"] == "advanced"
        assert "troubleshooting" in result[0]["domains"]

    def test_architecture_design_is_expert(self):
        result = map_capabilities_to_skills(["architecture_design"])
        assert len(result) == 1
        assert result[0]["name"] == "system_design"
        assert result[0]["level"] == "expert"

    def test_unknown_capability_returns_default(self):
        result = map_capabilities_to_skills(["unknown_capability"])
        assert len(result) == 1
        assert result[0]["name"] == "unknown_capability"
        assert result[0]["level"] == "intermediate"
        assert result[0]["domains"] == ["general"]

    def test_deduplication_for_capabilities(self):
        """Passing the same capability twice should not produce duplicates."""
        result = map_capabilities_to_skills(["code_implementation", "code_implementation"])
        assert len(result) == 1
        assert result[0]["name"] == "coding"

    def test_multiple_distinct_capabilities(self):
        result = map_capabilities_to_skills(["code_implementation", "debugging", "deployment"])
        assert len(result) == 3
        names = [s["name"] for s in result]
        assert "coding" in names
        assert "debugging" in names
        assert "deployment" in names

    def test_empty_list_returns_empty(self):
        result = map_capabilities_to_skills([])
        assert result == []

    def test_returns_copies_not_originals(self):
        result = map_capabilities_to_skills(["debugging"])
        result[0]["name"] = "mutated"
        assert CAPABILITY_SKILL_MAPPINGS["debugging"]["name"] == "debugging"

    def test_all_capability_mappings_have_required_keys(self):
        for cap_name, skill_data in CAPABILITY_SKILL_MAPPINGS.items():
            assert "name" in skill_data, f"{cap_name} missing 'name'"
            assert "level" in skill_data, f"{cap_name} missing 'level'"
            assert "domains" in skill_data, f"{cap_name} missing 'domains'"


# ============================================================================
# infer_skill_level Tests
# ============================================================================


class TestInferSkillLevel:
    """Tests for infer_skill_level function."""

    def test_claude_code_boosts_code_analysis_to_expert(self):
        level = infer_skill_level("code_analyzer", "claude_code")
        assert level == "expert"

    def test_claude_code_boosts_testing_to_advanced(self):
        level = infer_skill_level("test_runner", "claude_code")
        assert level == "advanced"

    def test_claude_code_boosts_file_operations_to_advanced(self):
        level = infer_skill_level("file_system", "claude_code")
        assert level == "advanced"

    def test_claude_code_boosts_version_control_to_advanced(self):
        level = infer_skill_level("git_manager", "claude_code")
        assert level == "advanced"

    def test_generic_agent_returns_base_level(self):
        """An unknown agent type should return the base level from TOOL_SKILL_MAPPINGS."""
        level = infer_skill_level("file_system", "generic_agent")
        assert level == "intermediate"

    def test_generic_agent_returns_base_level_advanced(self):
        level = infer_skill_level("code_analyzer", "generic_agent")
        assert level == "advanced"

    def test_unknown_tool_and_unknown_agent_returns_intermediate(self):
        level = infer_skill_level("nonexistent_tool", "generic_agent")
        assert level == "intermediate"

    def test_pydantic_ai_boosts_ai_integration_to_expert(self):
        level = infer_skill_level("llm_interface", "pydantic_ai")
        assert level == "expert"

    def test_pydantic_ai_boosts_data_processing_to_advanced(self):
        level = infer_skill_level("data_transformer", "pydantic_ai")
        assert level == "advanced"

    def test_fastapi_specialist_boosts_api_development_to_expert(self):
        level = infer_skill_level("api_builder", "fastapi_specialist")
        assert level == "expert"

    def test_fastapi_specialist_boosts_testing_to_advanced(self):
        level = infer_skill_level("test_runner", "fastapi_specialist")
        assert level == "advanced"

    def test_data_engineer_boosts_database_to_expert(self):
        level = infer_skill_level("database_connector", "data_engineer")
        assert level == "expert"

    def test_data_engineer_boosts_data_processing_to_expert(self):
        level = infer_skill_level("data_transformer", "data_engineer")
        assert level == "expert"

    def test_devops_engineer_boosts_containerization_to_expert(self):
        level = infer_skill_level("docker_manager", "devops_engineer")
        assert level == "expert"

    def test_devops_engineer_boosts_monitoring_to_expert(self):
        level = infer_skill_level("logger", "devops_engineer")
        assert level == "expert"


# ============================================================================
# merge_skill_lists Tests
# ============================================================================


class TestMergeSkillLists:
    """Tests for merge_skill_lists function."""

    def test_merge_non_overlapping_skills(self):
        tool_skills = [{"name": "coding", "level": "advanced", "domains": ["python"]}]
        cap_skills = [{"name": "testing", "level": "intermediate", "domains": ["pytest"]}]
        result = merge_skill_lists(tool_skills, cap_skills)
        assert len(result) == 2
        names = [s["name"] for s in result]
        assert "coding" in names
        assert "testing" in names

    def test_merge_takes_higher_level_for_duplicates(self):
        tool_skills = [{"name": "coding", "level": "intermediate", "domains": ["python"]}]
        cap_skills = [{"name": "coding", "level": "expert", "domains": ["backend"]}]
        result = merge_skill_lists(tool_skills, cap_skills)
        assert len(result) == 1
        assert result[0]["name"] == "coding"
        assert result[0]["level"] == "expert"

    def test_merge_does_not_downgrade_level(self):
        tool_skills = [{"name": "coding", "level": "expert", "domains": ["python"]}]
        cap_skills = [{"name": "coding", "level": "intermediate", "domains": ["backend"]}]
        result = merge_skill_lists(tool_skills, cap_skills)
        assert len(result) == 1
        assert result[0]["level"] == "expert"

    def test_merge_combines_domains(self):
        tool_skills = [{"name": "coding", "level": "advanced", "domains": ["python"]}]
        cap_skills = [{"name": "coding", "level": "advanced", "domains": ["backend", "python"]}]
        result = merge_skill_lists(tool_skills, cap_skills)
        assert len(result) == 1
        domains = set(result[0]["domains"])
        assert "python" in domains
        assert "backend" in domains

    def test_merge_empty_lists(self):
        result = merge_skill_lists([], [])
        assert result == []

    def test_merge_one_empty_list(self):
        tool_skills = [{"name": "coding", "level": "advanced", "domains": ["python"]}]
        result = merge_skill_lists(tool_skills, [])
        assert len(result) == 1
        assert result[0]["name"] == "coding"

    def test_merge_multiple_duplicates(self):
        tool_skills = [
            {"name": "coding", "level": "intermediate", "domains": ["python"]},
            {"name": "testing", "level": "beginner", "domains": ["unit"]},
        ]
        cap_skills = [
            {"name": "coding", "level": "expert", "domains": ["backend"]},
            {"name": "testing", "level": "advanced", "domains": ["integration"]},
        ]
        result = merge_skill_lists(tool_skills, cap_skills)
        assert len(result) == 2
        skill_map = {s["name"]: s for s in result}
        assert skill_map["coding"]["level"] == "expert"
        assert skill_map["testing"]["level"] == "advanced"
        assert "python" in skill_map["coding"]["domains"]
        assert "backend" in skill_map["coding"]["domains"]
        assert "unit" in skill_map["testing"]["domains"]
        assert "integration" in skill_map["testing"]["domains"]

    def test_merge_preserves_unique_domains_no_duplicates(self):
        """Merged domain list should not contain duplicates (using set union)."""
        tool_skills = [{"name": "coding", "level": "advanced", "domains": ["python", "backend"]}]
        cap_skills = [{"name": "coding", "level": "advanced", "domains": ["python", "api"]}]
        result = merge_skill_lists(tool_skills, cap_skills)
        domains = result[0]["domains"]
        # Set union means no duplicates
        assert len(domains) == len(set(domains))
        assert set(domains) == {"python", "backend", "api"}


# ============================================================================
# adjust_skill_levels_by_agent_type Tests
# ============================================================================


class TestAdjustSkillLevelsByAgentType:
    """Tests for adjust_skill_levels_by_agent_type function."""

    def test_claude_code_adjusts_code_analysis_to_expert(self):
        skills = [{"name": "code_analysis", "level": "advanced", "domains": ["python"]}]
        result = adjust_skill_levels_by_agent_type(skills, "claude_code")
        assert result[0]["level"] == "expert"

    def test_claude_code_adjusts_testing_to_advanced(self):
        skills = [{"name": "testing", "level": "intermediate", "domains": ["pytest"]}]
        result = adjust_skill_levels_by_agent_type(skills, "claude_code")
        assert result[0]["level"] == "advanced"

    def test_does_not_downgrade_existing_level(self):
        """If a skill is already expert, it should not be downgraded."""
        skills = [{"name": "file_operations", "level": "expert", "domains": ["filesystem"]}]
        result = adjust_skill_levels_by_agent_type(skills, "claude_code")
        # file_operations gets boosted to "advanced" by claude_code,
        # but existing is "expert" which is higher, so it stays expert
        assert result[0]["level"] == "expert"

    def test_generic_agent_does_not_change_levels(self):
        skills = [{"name": "coding", "level": "intermediate", "domains": ["python"]}]
        result = adjust_skill_levels_by_agent_type(skills, "generic_agent")
        assert result[0]["level"] == "intermediate"

    def test_pydantic_ai_adjusts_ai_integration(self):
        skills = [{"name": "ai_integration", "level": "intermediate", "domains": ["llm"]}]
        result = adjust_skill_levels_by_agent_type(skills, "pydantic_ai")
        assert result[0]["level"] == "expert"

    def test_devops_engineer_adjusts_containerization(self):
        skills = [{"name": "containerization", "level": "intermediate", "domains": ["docker"]}]
        result = adjust_skill_levels_by_agent_type(skills, "devops_engineer")
        assert result[0]["level"] == "expert"

    def test_returns_copies_not_originals(self):
        skills = [{"name": "coding", "level": "intermediate", "domains": ["python"]}]
        result = adjust_skill_levels_by_agent_type(skills, "claude_code")
        result[0]["name"] = "mutated"
        assert skills[0]["name"] == "coding"

    def test_empty_skills_returns_empty(self):
        result = adjust_skill_levels_by_agent_type([], "claude_code")
        assert result == []

    def test_multiple_skills_adjusted(self):
        skills = [
            {"name": "code_analysis", "level": "intermediate", "domains": ["python"]},
            {"name": "testing", "level": "beginner", "domains": ["pytest"]},
            {"name": "monitoring", "level": "intermediate", "domains": ["logging"]},
        ]
        result = adjust_skill_levels_by_agent_type(skills, "claude_code")
        skill_map = {s["name"]: s for s in result}
        assert skill_map["code_analysis"]["level"] == "expert"
        assert skill_map["testing"]["level"] == "advanced"
        # monitoring is not boosted by claude_code
        assert skill_map["monitoring"]["level"] == "intermediate"


# ============================================================================
# Mapping Data Integrity Tests
# ============================================================================


class TestMappingDataIntegrity:
    """Tests to verify the mapping data structures are well-formed."""

    def test_agent_type_level_mapping_has_expected_agents(self):
        expected_agents = {
            "claude_code",
            "pydantic_ai",
            "fastapi_specialist",
            "data_engineer",
            "devops_engineer",
        }
        assert set(AGENT_TYPE_LEVEL_MAPPING.keys()) == expected_agents

    def test_all_agent_levels_are_valid(self):
        valid_levels = {"beginner", "intermediate", "advanced", "expert"}
        for agent_type, skill_levels in AGENT_TYPE_LEVEL_MAPPING.items():
            for skill_name, level in skill_levels.items():
                assert level in valid_levels, (
                    f"Agent '{agent_type}' skill '{skill_name}' has invalid level '{level}'"
                )

    def test_all_tool_skill_levels_are_valid(self):
        valid_levels = {"beginner", "intermediate", "advanced", "expert"}
        for tool_name, skill_data in TOOL_SKILL_MAPPINGS.items():
            assert skill_data["level"] in valid_levels, (
                f"Tool '{tool_name}' has invalid level '{skill_data['level']}'"
            )

    def test_all_capability_skill_levels_are_valid(self):
        valid_levels = {"beginner", "intermediate", "advanced", "expert"}
        for cap_name, skill_data in CAPABILITY_SKILL_MAPPINGS.items():
            assert skill_data["level"] in valid_levels, (
                f"Capability '{cap_name}' has invalid level '{skill_data['level']}'"
            )
