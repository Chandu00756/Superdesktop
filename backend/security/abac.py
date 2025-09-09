"""
ABAC Policy Engine with simple predicate-based DSL.
"""
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Decision:
    allow: bool
    reasons: list[str]


Predicate = Callable[[dict[str, Any]], bool]


@dataclass
class Rule:
    name: str
    when: Predicate
    effect: bool  # True=allow, False=deny
    reason: str


class Policy:
    def __init__(self, rules: list[Rule], default_allow: bool = False):
        self.rules = rules
        self.default_allow = default_allow

    def evaluate(self, subject: dict, resource: dict, action: str, ctx: dict) -> Decision:
        input_ctx = {"sub": subject, "res": resource, "act": action, "ctx": ctx}
        reasons: list[str] = []
        for r in self.rules:
            try:
                if r.when(input_ctx):
                    reasons.append(r.reason)
                    return Decision(allow=r.effect, reasons=reasons)
            except Exception:
                continue
        return Decision(allow=self.default_allow, reasons=reasons or ["default"])


# Example predicates (can be composed externally)
def after_hours(inp):
    hour = int(inp["ctx"].get("now_hour", 0))
    return hour < 8 or hour >= 20


def is_admin(inp):
    return "admin" in (inp["sub"].get("roles") or [])


def is_owner(inp):
    return inp["sub"].get("id") == inp["res"].get("owner_id")
