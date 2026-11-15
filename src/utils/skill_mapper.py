"""Skill mapping utilities for converting tools and capabilities to skill representations."""

from pydantic import BaseModel, Field


class Skill(BaseModel):
    """Skill representation model."""

    name: str = Field(..., description="Skill name")
    level: str = Field(..., description="Skill proficiency level")
    domains: list[str] = Field(default_factory=list, description="Related domains")


# Comprehensive tool to skill mappings
TOOL_SKILL_MAPPINGS: dict[str, dict] = {
    # File system tools
    "file_system": {
        "name": "file_operations",
        "level": "intermediate",
        "domains": ["filesystem", "io"],
    },
    "file_reader": {
        "name": "file_operations",
        "level": "intermediate",
        "domains": ["filesystem", "io", "reading"],
    },
    "file_writer": {
        "name": "file_operations",
        "level": "intermediate",
        "domains": ["filesystem", "io", "writing"],
    },
    "directory_manager": {
        "name": "file_operations",
        "level": "advanced",
        "domains": ["filesystem", "organization"],
    },
    # Code tools
    "code_analyzer": {
        "name": "code_analysis",
        "level": "advanced",
        "domains": ["python", "analysis", "ast"],
    },
    "syntax_checker": {
        "name": "code_quality",
        "level": "intermediate",
        "domains": ["python", "validation", "linting"],
    },
    "code_formatter": {
        "name": "code_quality",
        "level": "intermediate",
        "domains": ["python", "formatting", "style"],
    },
    "refactoring": {
        "name": "refactoring",
        "level": "advanced",
        "domains": ["python", "architecture", "design"],
    },
    # Testing tools
    "test_runner": {
        "name": "testing",
        "level": "intermediate",
        "domains": ["pytest", "quality_assurance"],
    },
    "coverage_analyzer": {
        "name": "testing",
        "level": "advanced",
        "domains": ["pytest", "quality_assurance", "metrics"],
    },
    "mock_generator": {
        "name": "testing",
        "level": "advanced",
        "domains": ["pytest", "mocking", "unit_testing"],
    },
    # Database tools
    "database_connector": {
        "name": "database_operations",
        "level": "intermediate",
        "domains": ["sql", "database", "data"],
    },
    "query_builder": {
        "name": "database_operations",
        "level": "advanced",
        "domains": ["sql", "query_optimization"],
    },
    "migration_manager": {
        "name": "database_operations",
        "level": "advanced",
        "domains": ["sql", "schema_management", "alembic"],
    },
    # API tools
    "api_client": {
        "name": "api_integration",
        "level": "intermediate",
        "domains": ["http", "rest", "integration"],
    },
    "api_builder": {
        "name": "api_development",
        "level": "advanced",
        "domains": ["fastapi", "rest", "backend"],
    },
    "openapi_generator": {
        "name": "api_development",
        "level": "advanced",
        "domains": ["fastapi", "openapi", "documentation"],
    },
    # Data processing tools
    "data_transformer": {
        "name": "data_processing",
        "level": "intermediate",
        "domains": ["pandas", "data_manipulation"],
    },
    "data_validator": {
        "name": "data_processing",
        "level": "advanced",
        "domains": ["pydantic", "validation", "schema"],
    },
    "csv_processor": {
        "name": "data_processing",
        "level": "intermediate",
        "domains": ["csv", "data_import", "data_export"],
    },
    # AI/ML tools
    "llm_interface": {
        "name": "ai_integration",
        "level": "advanced",
        "domains": ["llm", "ai", "integration"],
    },
    "prompt_manager": {
        "name": "ai_integration",
        "level": "intermediate",
        "domains": ["llm", "prompt_engineering"],
    },
    "embedding_generator": {
        "name": "ai_integration",
        "level": "advanced",
        "domains": ["llm", "embeddings", "vector_search"],
    },
    # Web tools
    "web_scraper": {
        "name": "web_scraping",
        "level": "intermediate",
        "domains": ["http", "parsing", "beautifulsoup"],
    },
    "browser_automation": {
        "name": "web_automation",
        "level": "advanced",
        "domains": ["selenium", "automation", "testing"],
    },
    # Version control tools
    "git_manager": {
        "name": "version_control",
        "level": "intermediate",
        "domains": ["git", "scm"],
    },
    "commit_analyzer": {
        "name": "version_control",
        "level": "advanced",
        "domains": ["git", "analysis", "history"],
    },
    # DevOps tools
    "docker_manager": {
        "name": "containerization",
        "level": "advanced",
        "domains": ["docker", "devops", "deployment"],
    },
    "ci_cd_builder": {
        "name": "devops",
        "level": "advanced",
        "domains": ["ci_cd", "automation", "deployment"],
    },
    # Monitoring tools
    "logger": {
        "name": "monitoring",
        "level": "intermediate",
        "domains": ["logging", "observability"],
    },
    "metrics_collector": {
        "name": "monitoring",
        "level": "advanced",
        "domains": ["metrics", "observability", "performance"],
    },
}


# Capability to skill mappings
CAPABILITY_SKILL_MAPPINGS: dict[str, dict] = {
    # Code capabilities
    "code_implementation": {
        "name": "coding",
        "level": "advanced",
        "domains": ["python", "implementation"],
    },
    "code_review": {
        "name": "code_review",
        "level": "advanced",
        "domains": ["python", "quality_assurance", "best_practices"],
    },
    "debugging": {
        "name": "debugging",
        "level": "advanced",
        "domains": ["python", "troubleshooting", "problem_solving"],
    },
    "architecture_design": {
        "name": "system_design",
        "level": "expert",
        "domains": ["architecture", "design_patterns", "scalability"],
    },
    # Testing capabilities
    "test_creation": {
        "name": "test_development",
        "level": "advanced",
        "domains": ["pytest", "tdd", "quality_assurance"],
    },
    "test_automation": {
        "name": "test_automation",
        "level": "advanced",
        "domains": ["pytest", "automation", "ci_cd"],
    },
    # Data capabilities
    "data_analysis": {
        "name": "data_analysis",
        "level": "advanced",
        "domains": ["pandas", "statistics", "insights"],
    },
    "data_modeling": {
        "name": "data_modeling",
        "level": "advanced",
        "domains": ["pydantic", "schema_design", "validation"],
    },
    "etl_pipeline": {
        "name": "data_engineering",
        "level": "advanced",
        "domains": ["etl", "data_processing", "pipelines"],
    },
    # API capabilities
    "api_design": {
        "name": "api_design",
        "level": "advanced",
        "domains": ["fastapi", "rest", "architecture"],
    },
    "api_testing": {
        "name": "api_testing",
        "level": "intermediate",
        "domains": ["http", "testing", "validation"],
    },
    # AI capabilities
    "ai_integration": {
        "name": "ai_development",
        "level": "advanced",
        "domains": ["llm", "ai", "integration"],
    },
    "prompt_engineering": {
        "name": "prompt_engineering",
        "level": "intermediate",
        "domains": ["llm", "optimization", "ai"],
    },
    "model_training": {
        "name": "machine_learning",
        "level": "expert",
        "domains": ["ml", "training", "optimization"],
    },
    # DevOps capabilities
    "deployment": {
        "name": "deployment",
        "level": "advanced",
        "domains": ["docker", "kubernetes", "production"],
    },
    "infrastructure_management": {
        "name": "infrastructure",
        "level": "expert",
        "domains": ["cloud", "terraform", "automation"],
    },
    # Documentation capabilities
    "documentation": {
        "name": "documentation",
        "level": "intermediate",
        "domains": ["technical_writing", "markdown", "api_docs"],
    },
    "code_documentation": {
        "name": "code_documentation",
        "level": "intermediate",
        "domains": ["docstrings", "sphinx", "api_reference"],
    },
}


# Agent type specific skill level adjustments
AGENT_TYPE_LEVEL_MAPPING: dict[str, dict[str, str]] = {
    "claude_code": {
        "coding": "expert",
        "code_review": "expert",
        "debugging": "expert",
        "refactoring": "expert",
        "code_analysis": "expert",
        "system_design": "expert",
    },
    "pydantic_ai": {
        "ai_development": "expert",
        "ai_integration": "expert",
        "prompt_engineering": "advanced",
        "data_modeling": "expert",
        "validation": "expert",
    },
    "fastapi_specialist": {
        "api_development": "expert",
        "api_design": "expert",
        "backend": "expert",
        "openapi": "expert",
    },
    "data_engineer": {
        "data_processing": "expert",
        "data_engineering": "expert",
        "etl": "expert",
        "database_operations": "expert",
    },
    "devops_engineer": {
        "devops": "expert",
        "deployment": "expert",
        "infrastructure": "expert",
        "containerization": "expert",
        "ci_cd": "expert",
    },
}


def map_tools_to_skills(tool_names: list[str]) -> list[dict]:
    """Map tool names to skill representations.

    Args:
        tool_names: List of tool names to map

    Returns:
        List of skill dictionaries with name, level, and domains

    Example:
        >>> map_tools_to_skills(["file_system", "code_analyzer"])
        [
            {"name": "file_operations", "level": "intermediate", "domains": ["filesystem", "io"]},
            {"name": "code_analysis", "level": "advanced", "domains": ["python", "analysis", "ast"]}
        ]
    """
    skills = []
    seen_skills = set()

    for tool_name in tool_names:
        # Get skill mapping, or create a default one
        skill_data = TOOL_SKILL_MAPPINGS.get(
            tool_name,
            {"name": tool_name, "level": "intermediate", "domains": ["general"]},
        )

        # Avoid duplicate skills
        skill_key = skill_data["name"]
        if skill_key not in seen_skills:
            skills.append(skill_data.copy())
            seen_skills.add(skill_key)

    return skills


def map_capabilities_to_skills(capabilities: list[str]) -> list[dict]:
    """Map agent template capabilities to skill representations.

    Args:
        capabilities: List of capability names to map

    Returns:
        List of skill dictionaries with name, level, and domains

    Example:
        >>> map_capabilities_to_skills(["code_implementation", "debugging"])
        [
            {"name": "coding", "level": "advanced", "domains": ["python", "implementation"]},
            {"name": "debugging", "level": "advanced", "domains": ["python", "troubleshooting", "problem_solving"]}
        ]
    """
    skills = []
    seen_skills = set()

    for capability in capabilities:
        # Get skill mapping, or create a default one
        skill_data = CAPABILITY_SKILL_MAPPINGS.get(
            capability,
            {"name": capability, "level": "intermediate", "domains": ["general"]},
        )

        # Avoid duplicate skills
        skill_key = skill_data["name"]
        if skill_key not in seen_skills:
            skills.append(skill_data.copy())
            seen_skills.add(skill_key)

    return skills


def infer_skill_level(tool_name: str, agent_type: str) -> str:  # noqa: PLR0911 -- early-return dispatch over (tool_name, agent_type) pairs
    """Infer skill level based on tool name and agent type.

    Args:
        tool_name: Name of the tool
        agent_type: Type of agent (e.g., "claude_code", "pydantic_ai")

    Returns:
        Skill level as string ("beginner", "intermediate", "advanced", "expert")

    Example:
        >>> infer_skill_level("code_analyzer", "claude_code")
        "expert"
        >>> infer_skill_level("file_system", "generic_agent")
        "intermediate"
    """
    # Get base skill name from tool
    tool_skill = TOOL_SKILL_MAPPINGS.get(tool_name, {})
    skill_name = tool_skill.get("name", tool_name)
    base_level = tool_skill.get("level", "intermediate")

    # Check if agent type has specific level for this skill
    agent_levels = AGENT_TYPE_LEVEL_MAPPING.get(agent_type, {})

    if skill_name in agent_levels:
        return agent_levels[skill_name]

    # Apply agent-specific boosts
    if agent_type == "claude_code":
        # Claude Code is expert at code-related skills
        if skill_name in [
            "coding",
            "code_review",
            "debugging",
            "refactoring",
            "code_analysis",
            "code_quality",
            "system_design",
        ]:
            return "expert"
        elif skill_name in ["testing", "file_operations", "version_control"]:
            return "advanced"

    elif agent_type == "pydantic_ai":
        # Pydantic AI is expert at AI and data modeling
        if skill_name in [
            "ai_development",
            "ai_integration",
            "data_modeling",
            "validation",
            "prompt_engineering",
        ]:
            return "expert"
        elif skill_name in ["api_development", "data_processing"]:
            return "advanced"

    elif agent_type == "fastapi_specialist":
        # FastAPI specialist is expert at API development
        if skill_name in ["api_development", "api_design", "backend"]:
            return "expert"
        elif skill_name in ["testing", "documentation", "database_operations"]:
            return "advanced"

    elif agent_type == "data_engineer":
        # Data engineer is expert at data processing
        if skill_name in [
            "data_processing",
            "data_engineering",
            "etl",
            "database_operations",
        ]:
            return "expert"
        elif skill_name in ["coding", "api_integration"]:
            return "advanced"

    elif agent_type == "devops_engineer":
        # DevOps engineer is expert at deployment and infrastructure
        if skill_name in [
            "devops",
            "deployment",
            "infrastructure",
            "containerization",
            "ci_cd",
            "monitoring",
        ]:
            return "expert"
        elif skill_name in ["scripting", "automation"]:
            return "advanced"

    # Return base level if no specific mapping found
    return str(base_level)


def merge_skill_lists(tool_skills: list[dict], capability_skills: list[dict]) -> list[dict]:
    """Merge tool skills and capability skills, taking the higher level when duplicates exist.

    Args:
        tool_skills: Skills derived from tools
        capability_skills: Skills derived from capabilities

    Returns:
        Merged list of unique skills with highest levels
    """
    skill_map = {}
    level_hierarchy = {"beginner": 0, "intermediate": 1, "advanced": 2, "expert": 3}

    # Process all skills
    for skill in tool_skills + capability_skills:
        skill_name = skill["name"]

        if skill_name not in skill_map:
            skill_map[skill_name] = skill.copy()
        else:
            # Merge domains
            existing_domains = set(skill_map[skill_name]["domains"])
            new_domains = set(skill["domains"])
            skill_map[skill_name]["domains"] = list(existing_domains | new_domains)

            # Take higher level
            existing_level = skill_map[skill_name]["level"]
            new_level = skill["level"]
            if level_hierarchy.get(new_level, 0) > level_hierarchy.get(existing_level, 0):
                skill_map[skill_name]["level"] = new_level

    return list(skill_map.values())


def adjust_skill_levels_by_agent_type(skills: list[dict], agent_type: str) -> list[dict]:
    """Adjust skill levels based on agent type specialization.

    Args:
        skills: List of skill dictionaries
        agent_type: Type of agent

    Returns:
        Updated skills list with adjusted levels
    """
    level_hierarchy = {"beginner": 0, "intermediate": 1, "advanced": 2, "expert": 3}
    adjusted_skills = []

    for skill in skills:
        skill_copy = skill.copy()
        skill_name = skill["name"]
        current_level = skill["level"]

        # Infer level based on agent type
        inferred_level = infer_skill_level(skill_name, agent_type)

        # Take the higher level
        if level_hierarchy.get(inferred_level, 0) > level_hierarchy.get(current_level, 0):
            skill_copy["level"] = inferred_level

        adjusted_skills.append(skill_copy)

    return adjusted_skills
