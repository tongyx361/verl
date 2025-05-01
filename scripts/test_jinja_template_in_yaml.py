#!/usr/bin/env python3
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import hydra
import jinja2
import yaml
from hydra.core.config_store import ConfigStore


def apply_template(template_str: str, content: Dict[str, Any]) -> str:
    """Apply a Jinja2 template with custom filters.

    Args:
        template_str: Jinja2 template string
        content: Dictionary containing variables to use in the template

    Returns:
        Rendered template string
    """
    # Create a Jinja2 environment
    env = jinja2.Environment()

    # Add custom filters to mimic the template's behavior
    def regex_replace(value, pattern, replacement):
        return re.sub(pattern, replacement, value, flags=re.DOTALL)

    env.filters["regex_replace"] = regex_replace

    # Compile the template
    template = env.from_string(template_str)

    # Render with the content
    return template.render(**content)


@dataclass
class YamlItemConfig:
    file_path: Path = Path("configs/rl_trainer/data/cot_boxed_sfx_template_rm_simplerl_fmt.yaml")
    field_path: List[str] = field(default_factory=lambda: ["last_user_msg_template"])


@dataclass
class TestJinjaTemplateInYamlConfig:
    template: YamlItemConfig = field(default_factory=YamlItemConfig)
    input_data: Dict[str, Any] = field(
        default_factory=lambda: {"content": "Question:\nWhat's 1+1?\nAnswer: Let's think step by step.\n"}
    )


cs = ConfigStore.instance()
MAIN_CONFIG_NAME = "main_config"
cs.store(name=MAIN_CONFIG_NAME, node=TestJinjaTemplateInYamlConfig)


def test_jinja_template_in_yaml(cfg: TestJinjaTemplateInYamlConfig) -> None:
    """Process templates in YAML files using Hydra configuration.

    Args:
        cfg: Hydra configuration object with:
            - yaml_file: Path to the YAML file containing the template
            - template_field: Field in the YAML file containing the template
            - input_data: Dictionary with input data for template rendering
            - debug: Whether to show debug information
    """
    # Read the YAML file
    try:
        with open(cfg.template.file_path) as file:
            config = yaml.safe_load(file)

        # Get the template from the specified field
        template_field_parts = cfg.template.field_path
        template_container = config
        for part in template_field_parts:
            template_container = template_container.get(part, {})

        if isinstance(template_container, str):
            template_str = template_container
        else:
            raise ValueError(f"Template field '{cfg.template.field_path}' not found or not a string")

    except Exception as e:
        print(f"Error reading or processing YAML file: {e}")
        return

    print("==== Original Template ====")
    print(template_str)
    print("\n==== Input Data ====")
    for key, value in cfg.input_data.items():
        print(f"{key}: {repr(value)}")

    # Extract the template logic (if there's a separator)
    if "\n\n" in template_str:
        template_logic = template_str.split("\n\n")[0]
        # Apply just the content transformations (without suffix)
        transformed_content = apply_template(template_logic, cfg.input_data)
        print("\n==== After Initial Transformations ====")
        print(repr(transformed_content))

    # Apply the full template
    result = apply_template(template_str, cfg.input_data)

    print("\n==== Final Result ====")
    print(repr(result))

    print("\n==== Final Result (Formatted) ====")
    print(result)


if __name__ == "__main__":
    hydra.main(version_base=None, config_name=MAIN_CONFIG_NAME)(test_jinja_template_in_yaml)()
