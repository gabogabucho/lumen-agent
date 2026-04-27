"""Comprehensive tests for ModelRouter and ModelRouterConfig."""

import unittest

from lumen.core.model_router import VALID_ROLES, ModelRouter, ModelRouterConfig


class BasicRoutingTests(unittest.TestCase):
    """REQ-1: Basic routing — default model returned for unknown role."""

    def test_basic_routing_returns_default(self):
        """Unknown role falls back to default model."""
        cfg = ModelRouterConfig(
            default="deepseek/deepseek-chat",
            use_default_for_all=False,
        )
        router = ModelRouter(cfg)
        self.assertEqual(router.get_model("nonexistent"), "deepseek/deepseek-chat")

    def test_main_role_returns_default(self):
        """'main' role always returns default when use_default_for_all is off."""
        cfg = ModelRouterConfig(
            default="deepseek/deepseek-chat",
            roles={"executor": "gpt-4o-mini"},
            use_default_for_all=False,
        )
        router = ModelRouter(cfg)
        self.assertEqual(router.get_model("main"), "deepseek/deepseek-chat")

    def test_no_config_returns_builtin_default(self):
        """Router with no config uses built-in default."""
        router = ModelRouter()
        self.assertEqual(router.get_model(), "deepseek/deepseek-chat")


class RoleSpecificRoutingTests(unittest.TestCase):
    """REQ-2: Role-specific routing — each role gets its model."""

    def test_planner_gets_own_model(self):
        cfg = ModelRouterConfig(
            default="deepseek/deepseek-chat",
            roles={"planner": "claude-sonnet-4-20250514"},
            use_default_for_all=False,
        )
        router = ModelRouter(cfg)
        self.assertEqual(router.get_model("planner"), "claude-sonnet-4-20250514")

    def test_executor_gets_own_model(self):
        cfg = ModelRouterConfig(
            default="deepseek/deepseek-chat",
            roles={"executor": "gpt-4o-mini"},
            use_default_for_all=False,
        )
        router = ModelRouter(cfg)
        self.assertEqual(router.get_model("executor"), "gpt-4o-mini")

    def test_summarizer_gets_own_model(self):
        cfg = ModelRouterConfig(
            default="deepseek/deepseek-chat",
            roles={"summarizer": "google/gemini-2.0-flash"},
            use_default_for_all=False,
        )
        router = ModelRouter(cfg)
        self.assertEqual(router.get_model("summarizer"), "google/gemini-2.0-flash")

    def test_responder_gets_own_model(self):
        cfg = ModelRouterConfig(
            default="deepseek/deepseek-chat",
            roles={"responder": "meta-llama/llama-3.3-70b-instruct"},
            use_default_for_all=False,
        )
        router = ModelRouter(cfg)
        self.assertEqual(router.get_model("responder"), "meta-llama/llama-3.3-70b-instruct")

    def test_all_roles_coexist(self):
        """Multiple roles can each have their own model."""
        cfg = ModelRouterConfig(
            default="deepseek/deepseek-chat",
            roles={
                "planner": "claude-sonnet-4-20250514",
                "executor": "gpt-4o-mini",
                "summarizer": "google/gemini-2.0-flash",
                "responder": "meta-llama/llama-3.3-70b-instruct",
            },
            use_default_for_all=False,
        )
        router = ModelRouter(cfg)
        self.assertEqual(router.get_model("planner"), "claude-sonnet-4-20250514")
        self.assertEqual(router.get_model("executor"), "gpt-4o-mini")
        self.assertEqual(router.get_model("summarizer"), "google/gemini-2.0-flash")
        self.assertEqual(router.get_model("responder"), "meta-llama/llama-3.3-70b-instruct")
        self.assertEqual(router.get_model("main"), "deepseek/deepseek-chat")


class UseDefaultForAllTests(unittest.TestCase):
    """REQ-3: use_default_for_all toggle — all roles return default."""

    def test_toggle_true_ignores_role_overrides(self):
        """When use_default_for_all is True, role overrides are ignored."""
        cfg = ModelRouterConfig(
            default="deepseek/deepseek-chat",
            roles={"planner": "claude-sonnet-4-20250514", "executor": "gpt-4o-mini"},
            use_default_for_all=True,
        )
        router = ModelRouter(cfg)
        self.assertEqual(router.get_model("planner"), "deepseek/deepseek-chat")
        self.assertEqual(router.get_model("executor"), "deepseek/deepseek-chat")
        self.assertEqual(router.get_model("main"), "deepseek/deepseek-chat")

    def test_toggle_false_uses_role_overrides(self):
        """When use_default_for_all is False, role overrides are active."""
        cfg = ModelRouterConfig(
            default="deepseek/deepseek-chat",
            roles={"planner": "claude-sonnet-4-20250514"},
            use_default_for_all=False,
        )
        router = ModelRouter(cfg)
        self.assertEqual(router.get_model("planner"), "claude-sonnet-4-20250514")
        self.assertEqual(router.get_model("main"), "deepseek/deepseek-chat")

    def test_set_use_default_for_all_live_toggle(self):
        """set_use_default_for_all toggles behavior at runtime."""
        cfg = ModelRouterConfig(
            default="deepseek/deepseek-chat",
            roles={"planner": "claude-sonnet-4-20250514"},
            use_default_for_all=False,
        )
        router = ModelRouter(cfg)
        self.assertEqual(router.get_model("planner"), "claude-sonnet-4-20250514")

        router.set_use_default_for_all(True)
        self.assertEqual(router.get_model("planner"), "deepseek/deepseek-chat")

        router.set_use_default_for_all(False)
        self.assertEqual(router.get_model("planner"), "claude-sonnet-4-20250514")


class FallbackChainTests(unittest.TestCase):
    """REQ-4 & REQ-5: Fallback chain — resolve_with_fallback behavior."""

    def test_resolve_with_fallback_returns_tuple(self):
        """resolve_with_fallback returns (primary, fallback)."""
        cfg = ModelRouterConfig(
            default="deepseek/deepseek-chat",
            fallback="google/gemini-2.0-flash",
            use_default_for_all=False,
        )
        router = ModelRouter(cfg)
        primary, fallback = router.resolve_with_fallback("main")
        self.assertEqual(primary, "deepseek/deepseek-chat")
        self.assertEqual(fallback, "google/gemini-2.0-flash")

    def test_resolve_with_fallback_role_specific(self):
        """Fallback chain uses role-specific model as primary."""
        cfg = ModelRouterConfig(
            default="deepseek/deepseek-chat",
            roles={"planner": "claude-sonnet-4-20250514"},
            fallback="google/gemini-2.0-flash",
            use_default_for_all=False,
        )
        router = ModelRouter(cfg)
        primary, fallback = router.resolve_with_fallback("planner")
        self.assertEqual(primary, "claude-sonnet-4-20250514")
        self.assertEqual(fallback, "google/gemini-2.0-flash")

    def test_no_fallback_when_same(self):
        """If primary == fallback, second element is empty string."""
        cfg = ModelRouterConfig(
            default="google/gemini-2.0-flash",
            fallback="google/gemini-2.0-flash",
            use_default_for_all=False,
        )
        router = ModelRouter(cfg)
        primary, fallback = router.resolve_with_fallback("main")
        self.assertEqual(primary, "google/gemini-2.0-flash")
        self.assertEqual(fallback, "")

    def test_no_fallback_when_same_role_specific(self):
        """Empty fallback when role-specific model equals fallback."""
        cfg = ModelRouterConfig(
            default="deepseek/deepseek-chat",
            roles={"planner": "google/gemini-2.0-flash"},
            fallback="google/gemini-2.0-flash",
            use_default_for_all=False,
        )
        router = ModelRouter(cfg)
        primary, fallback = router.resolve_with_fallback("planner")
        self.assertEqual(primary, "google/gemini-2.0-flash")
        self.assertEqual(fallback, "")

    def test_get_fallback_returns_guaranteed_fallback(self):
        """get_fallback always returns the configured fallback."""
        cfg = ModelRouterConfig(fallback="google/gemini-2.0-flash")
        router = ModelRouter(cfg)
        self.assertEqual(router.get_fallback(), "google/gemini-2.0-flash")


class ConfigFromDictTests(unittest.TestCase):
    """REQ-6, REQ-7, REQ-8: Config from dict parsing."""

    def test_full_models_section(self):
        """from_config parses full models section correctly."""
        config = {
            "models": {
                "default": "claude-sonnet-4-20250514",
                "roles": {
                    "planner": "gpt-4o-mini",
                    "executor": "deepseek/deepseek-chat",
                },
                "fallback": "google/gemini-2.0-flash",
                "use_default_for_all": False,
            }
        }
        cfg = ModelRouterConfig.from_config(config)
        self.assertEqual(cfg.default, "claude-sonnet-4-20250514")
        self.assertEqual(cfg.roles["planner"], "gpt-4o-mini")
        self.assertEqual(cfg.roles["executor"], "deepseek/deepseek-chat")
        self.assertEqual(cfg.fallback, "google/gemini-2.0-flash")
        self.assertFalse(cfg.use_default_for_all)

    def test_legacy_single_model(self):
        """from_config falls back to config['model'] when no models section."""
        config = {"model": "gpt-4o-mini"}
        cfg = ModelRouterConfig.from_config(config)
        self.assertEqual(cfg.default, "gpt-4o-mini")
        self.assertEqual(cfg.fallback, "google/gemini-2.0-flash")  # default fallback
        self.assertTrue(cfg.use_default_for_all)  # default toggle

    def test_none_config_returns_defaults(self):
        """from_config with None returns all defaults."""
        cfg = ModelRouterConfig.from_config(None)
        self.assertEqual(cfg.default, "deepseek/deepseek-chat")
        self.assertEqual(cfg.fallback, "google/gemini-2.0-flash")
        self.assertTrue(cfg.use_default_for_all)
        self.assertEqual(cfg.roles, {})

    def test_empty_dict_returns_defaults(self):
        """from_config with {} returns all defaults."""
        cfg = ModelRouterConfig.from_config({})
        self.assertEqual(cfg.default, "deepseek/deepseek-chat")
        self.assertEqual(cfg.fallback, "google/gemini-2.0-flash")
        self.assertTrue(cfg.use_default_for_all)

    def test_models_section_with_string_value_ignored(self):
        """from_config treats non-dict models value as legacy."""
        config = {"models": "some-string"}
        cfg = ModelRouterConfig.from_config(config)
        self.assertEqual(cfg.default, "deepseek/deepseek-chat")

    def test_partial_models_section(self):
        """from_config fills missing models fields with defaults."""
        config = {"models": {"default": "claude-sonnet-4-20250514"}}
        cfg = ModelRouterConfig.from_config(config)
        self.assertEqual(cfg.default, "claude-sonnet-4-20250514")
        self.assertEqual(cfg.fallback, "google/gemini-2.0-flash")
        self.assertTrue(cfg.use_default_for_all)
        self.assertEqual(cfg.roles, {})


class InvalidRolesFilteredTests(unittest.TestCase):
    """REQ-9: Invalid roles filtered — keys not in VALID_ROLES are ignored."""

    def test_invalid_role_keys_ignored(self):
        """Roles with invalid keys are filtered out."""
        config = {
            "models": {
                "default": "deepseek/deepseek-chat",
                "roles": {
                    "planner": "gpt-4o-mini",
                    "hacker": "evil-model",          # not valid
                    "executor": "claude-sonnet-4-20250514",
                    "invalid": 123,                   # not a string
                },
            }
        }
        cfg = ModelRouterConfig.from_config(config)
        self.assertIn("planner", cfg.roles)
        self.assertIn("executor", cfg.roles)
        self.assertNotIn("hacker", cfg.roles)
        self.assertNotIn("invalid", cfg.roles)

    def test_non_string_role_values_ignored(self):
        """Role values that aren't strings are filtered out."""
        config = {
            "models": {
                "roles": {
                    "planner": 42,            # not a string
                    "executor": None,         # not a string
                    "summarizer": ["list"],   # not a string
                }
            }
        }
        cfg = ModelRouterConfig.from_config(config)
        self.assertEqual(cfg.roles, {})


class SetRoleModelTests(unittest.TestCase):
    """REQ-10: set_role_model — valid/invalid role handling."""

    def test_set_valid_role_returns_true(self):
        cfg = ModelRouterConfig(use_default_for_all=False)
        router = ModelRouter(cfg)
        self.assertTrue(router.set_role_model("planner", "gpt-4o-mini"))
        self.assertEqual(router.get_model("planner"), "gpt-4o-mini")

    def test_set_invalid_role_returns_false(self):
        router = ModelRouter()
        self.assertFalse(router.set_role_model("hacker", "evil-model"))

    def test_set_empty_model_returns_false(self):
        router = ModelRouter()
        self.assertFalse(router.set_role_model("planner", ""))

    def test_set_none_model_returns_false(self):
        router = ModelRouter()
        self.assertFalse(router.set_role_model("planner", None))

    def test_set_main_role_model(self):
        router = ModelRouter()
        # "main" is in VALID_ROLES but get_model always returns default for "main"
        self.assertTrue(router.set_role_model("main", "gpt-4o-mini"))
        # Even though set succeeds, get_model("main") returns default (by design)
        cfg = router.config
        self.assertEqual(cfg.roles.get("main"), "gpt-4o-mini")


class SetDefaultFallbackTests(unittest.TestCase):
    """REQ-11: set_default / set_fallback — valid/empty model handling."""

    def test_set_default_valid(self):
        router = ModelRouter()
        self.assertTrue(router.set_default("gpt-4o-mini"))
        self.assertEqual(router.get_model("main"), "gpt-4o-mini")

    def test_set_default_empty_returns_false(self):
        router = ModelRouter()
        self.assertFalse(router.set_default(""))
        self.assertEqual(router.get_model("main"), "deepseek/deepseek-chat")

    def test_set_default_none_returns_false(self):
        router = ModelRouter()
        self.assertFalse(router.set_default(None))

    def test_set_fallback_valid(self):
        router = ModelRouter()
        self.assertTrue(router.set_fallback("gpt-4o-mini"))
        self.assertEqual(router.get_fallback(), "gpt-4o-mini")

    def test_set_fallback_empty_returns_false(self):
        router = ModelRouter()
        self.assertFalse(router.set_fallback(""))
        self.assertEqual(router.get_fallback(), "google/gemini-2.0-flash")

    def test_set_fallback_none_returns_false(self):
        router = ModelRouter()
        self.assertFalse(router.set_fallback(None))


class ListRolesTests(unittest.TestCase):
    """REQ-12: list_roles — returns default + fallback + configured roles."""

    def test_list_roles_includes_default_and_fallback(self):
        router = ModelRouter()
        roles = router.list_roles()
        self.assertIn("default", roles)
        self.assertIn("fallback", roles)
        self.assertEqual(roles["default"], "deepseek/deepseek-chat")
        self.assertEqual(roles["fallback"], "google/gemini-2.0-flash")

    def test_list_roles_includes_configured_roles(self):
        cfg = ModelRouterConfig(
            roles={"planner": "gpt-4o-mini", "executor": "claude-sonnet-4-20250514"},
            use_default_for_all=False,
        )
        router = ModelRouter(cfg)
        roles = router.list_roles()
        self.assertEqual(roles["planner"], "gpt-4o-mini")
        self.assertEqual(roles["executor"], "claude-sonnet-4-20250514")

    def test_list_roles_returns_all_keys(self):
        cfg = ModelRouterConfig(
            roles={"planner": "gpt-4o-mini"},
            use_default_for_all=False,
        )
        router = ModelRouter(cfg)
        roles = router.list_roles()
        # default + fallback + 1 role = 3 keys
        self.assertEqual(len(roles), 3)


class LiveConfigUpdateTests(unittest.TestCase):
    """REQ-13: update_config / update_from_dict — live config updates."""

    def test_update_config_changes_routing(self):
        router = ModelRouter()
        self.assertEqual(router.get_model("main"), "deepseek/deepseek-chat")

        new_cfg = ModelRouterConfig(default="gpt-4o-mini", use_default_for_all=True)
        router.update_config(new_cfg)
        self.assertEqual(router.get_model("main"), "gpt-4o-mini")

    def test_update_from_dict(self):
        router = ModelRouter()
        self.assertEqual(router.get_model("main"), "deepseek/deepseek-chat")

        router.update_from_dict({
            "default": "claude-sonnet-4-20250514",
            "roles": {"planner": "gpt-4o-mini"},
            "use_default_for_all": False,
        })
        self.assertEqual(router.get_model("main"), "claude-sonnet-4-20250514")
        self.assertEqual(router.get_model("planner"), "gpt-4o-mini")

    def test_update_config_preserves_old_config_reference(self):
        """Old config object is not modified by update."""
        old_cfg = ModelRouterConfig(default="deepseek/deepseek-chat")
        router = ModelRouter(old_cfg)

        new_cfg = ModelRouterConfig(default="gpt-4o-mini")
        router.update_config(new_cfg)

        # Old config unchanged
        self.assertEqual(old_cfg.default, "deepseek/deepseek-chat")
        # Router uses new config
        self.assertEqual(router.config.default, "gpt-4o-mini")


class ToDictRoundTripTests(unittest.TestCase):
    """REQ-14: to_dict round-trip — serialize and re-parse."""

    def test_to_dict_round_trip(self):
        """Serialize to dict and re-parse produces equivalent config."""
        original = ModelRouterConfig(
            default="claude-sonnet-4-20250514",
            roles={"planner": "gpt-4o-mini", "executor": "deepseek/deepseek-chat"},
            fallback="google/gemini-2.0-flash",
            use_default_for_all=False,
        )
        as_dict = original.to_dict()

        # Re-parse
        reparsed = ModelRouterConfig.from_config({"models": as_dict})

        self.assertEqual(reparsed.default, original.default)
        self.assertEqual(reparsed.roles, original.roles)
        self.assertEqual(reparsed.fallback, original.fallback)
        self.assertEqual(reparsed.use_default_for_all, original.use_default_for_all)

    def test_to_dict_keys(self):
        """to_dict produces expected keys."""
        cfg = ModelRouterConfig()
        d = cfg.to_dict()
        self.assertIn("default", d)
        self.assertIn("roles", d)
        self.assertIn("fallback", d)
        self.assertIn("use_default_for_all", d)

    def test_to_dict_roles_is_dict(self):
        """to_dict roles is a plain dict, not the dataclass field default."""
        cfg = ModelRouterConfig(roles={"planner": "gpt-4o-mini"})
        d = cfg.to_dict()
        self.assertIsInstance(d["roles"], dict)
        self.assertEqual(d["roles"]["planner"], "gpt-4o-mini")


class ValidRolesConstantTests(unittest.TestCase):
    """Ensure VALID_ROLES constant is correct."""

    def test_valid_roles_contains_expected(self):
        self.assertIn("planner", VALID_ROLES)
        self.assertIn("executor", VALID_ROLES)
        self.assertIn("summarizer", VALID_ROLES)
        self.assertIn("responder", VALID_ROLES)
        self.assertIn("main", VALID_ROLES)
        self.assertEqual(len(VALID_ROLES), 5)


if __name__ == "__main__":
    unittest.main()
