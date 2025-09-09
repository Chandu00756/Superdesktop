"""
Omega Super Desktop Console v2.0 - Policy Management Engine
Production-grade policy engine with rule parsing, conflict resolution,
compliance validation, and dynamic enforcement capabilities.
"""

import asyncio
import logging
import time
import json
import re
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
from enum import Enum
import yaml

logger = logging.getLogger(__name__)

class PolicyType(Enum):
    SECURITY = "security"
    RESOURCE = "resource"
    ACCESS = "access"
    COMPLIANCE = "compliance"
    NETWORK = "network"
    DATA = "data"
    AUDIT = "audit"
    PERFORMANCE = "performance"

class PolicyScope(Enum):
    GLOBAL = "global"
    NODE = "node"
    USER = "user"
    GROUP = "group"
    WORKLOAD = "workload"
    SESSION = "session"

class PolicyEnforcement(Enum):
    ENFORCE = "enforce"
    WARN = "warn"
    AUDIT = "audit"
    BLOCK = "block"

class PolicyPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

class PolicyStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    TESTING = "testing"

@dataclass
class PolicyCondition:
    """A single policy condition"""
    field: str
    operator: str  # eq, ne, gt, lt, gte, lte, in, not_in, contains, regex
    value: Any
    case_sensitive: bool = True

@dataclass
class PolicyRule:
    """A policy rule with conditions and actions"""
    rule_id: str
    name: str
    description: str
    conditions: List[PolicyCondition]
    actions: List[Dict[str, Any]]
    logical_operator: str = "AND"  # AND, OR
    weight: float = 1.0
    enabled: bool = True

@dataclass
class Policy:
    """Complete policy definition"""
    policy_id: str
    name: str
    description: str
    policy_type: PolicyType
    scope: PolicyScope
    enforcement: PolicyEnforcement
    priority: PolicyPriority
    status: PolicyStatus
    rules: List[PolicyRule]
    
    # Metadata
    version: str = "1.0"
    created_by: str = "system"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    
    # Advanced features
    conflict_resolution: str = "priority"  # priority, merge, override
    inheritance: bool = True
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PolicyEvaluation:
    """Result of policy evaluation"""
    policy_id: str
    rule_id: Optional[str]
    evaluation_time: float
    context: Dict[str, Any]
    matched: bool
    actions_triggered: List[Dict[str, Any]]
    enforcement_action: PolicyEnforcement
    confidence: float = 1.0
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PolicyViolation:
    """Policy violation record"""
    violation_id: str
    policy_id: str
    rule_id: str
    severity: str
    description: str
    context: Dict[str, Any]
    timestamp: float
    resolved: bool = False
    resolution_notes: str = ""
    user_id: Optional[str] = None
    node_id: Optional[str] = None

@dataclass
class PolicyConflict:
    """Policy conflict detection result"""
    conflict_id: str
    policy_ids: List[str]
    conflict_type: str  # enforcement, action, scope
    description: str
    severity: str  # critical, high, medium, low
    resolution_strategy: str
    auto_resolvable: bool
    detected_at: float = field(default_factory=time.time)

class PolicyEngine:
    """Production-grade policy management and enforcement engine"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Policy storage
        self.policies: Dict[str, Policy] = {}
        self.active_policies: Dict[PolicyScope, List[str]] = defaultdict(list)
        self.policy_cache: Dict[str, List[PolicyEvaluation]] = {}
        
        # Evaluation state
        self.violations: List[PolicyViolation] = []
        self.conflicts: List[PolicyConflict] = []
        self.evaluation_history: deque = deque(maxlen=10000)
        
        # Performance tracking
        self.metrics = {
            'total_evaluations': 0,
            'policy_matches': 0,
            'violations_detected': 0,
            'conflicts_detected': 0,
            'average_evaluation_time': 0.0,
            'cache_hits': 0,
            'active_policies_count': 0
        }
        
        # Built-in operators
        self.operators = {
            'eq': lambda a, b: a == b,
            'ne': lambda a, b: a != b,
            'gt': lambda a, b: self._safe_compare(a, b, lambda x, y: x > y),
            'lt': lambda a, b: self._safe_compare(a, b, lambda x, y: x < y),
            'gte': lambda a, b: self._safe_compare(a, b, lambda x, y: x >= y),
            'lte': lambda a, b: self._safe_compare(a, b, lambda x, y: x <= y),
            'in': lambda a, b: a in b if isinstance(b, (list, tuple, set, str)) else False,
            'not_in': lambda a, b: a not in b if isinstance(b, (list, tuple, set, str)) else True,
            'contains': lambda a, b: b in a if isinstance(a, (list, tuple, set, str)) else False,
            'regex': lambda a, b: bool(re.search(str(b), str(a))),
            'starts_with': lambda a, b: str(a).startswith(str(b)),
            'ends_with': lambda a, b: str(a).endswith(str(b)),
            'length_eq': lambda a, b: len(a) == b if hasattr(a, '__len__') else False,
            'length_gt': lambda a, b: len(a) > b if hasattr(a, '__len__') else False,
            'length_lt': lambda a, b: len(a) < b if hasattr(a, '__len__') else False
        }
        
        # Custom action handlers
        self.action_handlers: Dict[str, Callable] = {}
        
        logger.info("Policy Engine initialized")
    
    async def create_policy(self, policy: Policy) -> bool:
        """Create a new policy"""
        try:
            # Validate policy
            if not await self._validate_policy(policy):
                return False
            
            # Check for conflicts
            conflicts = await self._detect_conflicts(policy)
            if conflicts:
                logger.warning(f"Policy {policy.policy_id} has conflicts: {len(conflicts)}")
                self.conflicts.extend(conflicts)
            
            # Store policy
            policy.updated_at = time.time()
            self.policies[policy.policy_id] = policy
            
            # Update active policies if policy is active
            if policy.status == PolicyStatus.ACTIVE:
                self.active_policies[policy.scope].append(policy.policy_id)
            
            # Clear related cache
            self._invalidate_cache(policy.scope)
            
            # Update metrics
            self.metrics['active_policies_count'] = sum(len(policies) for policies in self.active_policies.values())
            
            logger.info(f"Policy {policy.policy_id} created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create policy {policy.policy_id}: {e}")
            return False
    
    async def update_policy(self, policy_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing policy"""
        try:
            if policy_id not in self.policies:
                logger.error(f"Policy {policy_id} not found")
                return False
            
            policy = self.policies[policy_id]
            old_scope = policy.scope
            old_status = policy.status
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(policy, key):
                    setattr(policy, key, value)
            
            policy.updated_at = time.time()
            
            # Validate updated policy
            if not await self._validate_policy(policy):
                return False
            
            # Handle scope/status changes
            if old_scope != policy.scope or old_status != policy.status:
                # Remove from old scope
                if policy_id in self.active_policies[old_scope]:
                    self.active_policies[old_scope].remove(policy_id)
                
                # Add to new scope if active
                if policy.status == PolicyStatus.ACTIVE:
                    self.active_policies[policy.scope].append(policy_id)
            
            # Clear cache
            self._invalidate_cache(old_scope)
            if old_scope != policy.scope:
                self._invalidate_cache(policy.scope)
            
            logger.info(f"Policy {policy_id} updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update policy {policy_id}: {e}")
            return False
    
    async def delete_policy(self, policy_id: str) -> bool:
        """Delete a policy"""
        try:
            if policy_id not in self.policies:
                return False
            
            policy = self.policies[policy_id]
            
            # Remove from active policies
            if policy_id in self.active_policies[policy.scope]:
                self.active_policies[policy.scope].remove(policy_id)
            
            # Delete policy
            del self.policies[policy_id]
            
            # Clear cache
            self._invalidate_cache(policy.scope)
            
            # Update metrics
            self.metrics['active_policies_count'] = sum(len(policies) for policies in self.active_policies.values())
            
            logger.info(f"Policy {policy_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete policy {policy_id}: {e}")
            return False
    
    async def evaluate_policies(
        self,
        context: Dict[str, Any],
        scope: Optional[PolicyScope] = None,
        policy_types: Optional[List[PolicyType]] = None
    ) -> List[PolicyEvaluation]:
        """Evaluate policies against given context"""
        start_time = time.time()
        
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(context, scope, policy_types)
            
            # Check cache
            if cache_key in self.policy_cache:
                cache_result = self.policy_cache[cache_key]
                if time.time() - cache_result[0].evaluation_time < 300:  # 5 minute cache
                    self.metrics['cache_hits'] += 1
                    return cache_result
            
            # Get applicable policies
            applicable_policies = await self._get_applicable_policies(context, scope, policy_types)
            
            evaluations = []
            
            for policy in applicable_policies:
                if policy.status != PolicyStatus.ACTIVE:
                    continue
                
                policy_evaluation = await self._evaluate_policy(policy, context)
                if policy_evaluation:
                    evaluations.append(policy_evaluation)
                    
                    # Handle violations
                    if policy_evaluation.matched and policy_evaluation.enforcement_action in [PolicyEnforcement.BLOCK, PolicyEnforcement.ENFORCE]:
                        await self._handle_violation(policy, policy_evaluation)
            
            # Sort by priority and confidence
            evaluations.sort(key=lambda e: (
                self.policies[e.policy_id].priority.value,
                -e.confidence
            ))
            
            # Cache result
            if evaluations:
                self.policy_cache[cache_key] = evaluations
            
            # Update metrics
            evaluation_time = time.time() - start_time
            self.metrics['total_evaluations'] += 1
            self.metrics['policy_matches'] += len([e for e in evaluations if e.matched])
            self.metrics['average_evaluation_time'] = (
                (self.metrics['average_evaluation_time'] * (self.metrics['total_evaluations'] - 1) + evaluation_time)
                / self.metrics['total_evaluations']
            )
            
            # Store in history
            self.evaluation_history.extend(evaluations)
            
            return evaluations
            
        except Exception as e:
            logger.error(f"Policy evaluation failed: {e}")
            return []
    
    async def _evaluate_policy(self, policy: Policy, context: Dict[str, Any]) -> Optional[PolicyEvaluation]:
        """Evaluate a single policy against context"""
        try:
            matched_rules = []
            all_actions = []
            
            for rule in policy.rules:
                if not rule.enabled:
                    continue
                
                rule_matched = await self._evaluate_rule(rule, context)
                
                if rule_matched:
                    matched_rules.append(rule.rule_id)
                    all_actions.extend(rule.actions)
            
            # Determine if policy matched based on rules
            policy_matched = len(matched_rules) > 0
            
            if policy_matched:
                # Execute actions based on enforcement mode
                triggered_actions = []
                if policy.enforcement in [PolicyEnforcement.ENFORCE, PolicyEnforcement.BLOCK]:
                    triggered_actions = all_actions
                elif policy.enforcement == PolicyEnforcement.WARN:
                    triggered_actions = [{'type': 'warning', 'message': f'Policy {policy.name} violated'}]
                elif policy.enforcement == PolicyEnforcement.AUDIT:
                    triggered_actions = [{'type': 'audit', 'policy_id': policy.policy_id}]
                
                return PolicyEvaluation(
                    policy_id=policy.policy_id,
                    rule_id=matched_rules[0] if matched_rules else None,
                    evaluation_time=time.time(),
                    context=context.copy(),
                    matched=True,
                    actions_triggered=triggered_actions,
                    enforcement_action=policy.enforcement,
                    confidence=1.0,
                    message=f"Policy {policy.name} matched",
                    metadata={'matched_rules': matched_rules}
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Policy evaluation failed for {policy.policy_id}: {e}")
            return None
    
    async def _evaluate_rule(self, rule: PolicyRule, context: Dict[str, Any]) -> bool:
        """Evaluate a single rule against context"""
        try:
            if not rule.conditions:
                return True  # Empty conditions always match
            
            condition_results = []
            
            for condition in rule.conditions:
                result = await self._evaluate_condition(condition, context)
                condition_results.append(result)
            
            # Apply logical operator
            if rule.logical_operator.upper() == "OR":
                return any(condition_results)
            else:  # Default to AND
                return all(condition_results)
            
        except Exception as e:
            logger.error(f"Rule evaluation failed for {rule.rule_id}: {e}")
            return False
    
    async def _evaluate_condition(self, condition: PolicyCondition, context: Dict[str, Any]) -> bool:
        """Evaluate a single condition against context"""
        try:
            # Extract field value from context
            field_value = self._extract_field_value(condition.field, context)
            
            if field_value is None:
                return False
            
            # Apply case sensitivity
            if isinstance(field_value, str) and not condition.case_sensitive:
                field_value = field_value.lower()
                if isinstance(condition.value, str):
                    condition.value = condition.value.lower()
                elif isinstance(condition.value, list):
                    condition.value = [v.lower() if isinstance(v, str) else v for v in condition.value]
            
            # Get operator function
            if condition.operator not in self.operators:
                logger.error(f"Unknown operator: {condition.operator}")
                return False
            
            operator_func = self.operators[condition.operator]
            
            # Evaluate condition
            return operator_func(field_value, condition.value)
            
        except Exception as e:
            logger.error(f"Condition evaluation failed: {e}")
            return False
    
    def _extract_field_value(self, field_path: str, context: Dict[str, Any]) -> Any:
        """Extract field value from context using dot notation"""
        try:
            keys = field_path.split('.')
            value = context
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                elif hasattr(value, key):
                    value = getattr(value, key)
                else:
                    return None
            
            return value
            
        except Exception:
            return None
    
    def _safe_compare(self, a: Any, b: Any, compare_func: Callable) -> bool:
        """Safely compare values with type conversion"""
        try:
            # Try direct comparison first
            return compare_func(a, b)
        except (TypeError, ValueError):
            try:
                # Try converting to numbers
                return compare_func(float(a), float(b))
            except (TypeError, ValueError):
                try:
                    # Try string comparison
                    return compare_func(str(a), str(b))
                except:
                    return False
    
    async def _get_applicable_policies(
        self,
        context: Dict[str, Any],
        scope: Optional[PolicyScope] = None,
        policy_types: Optional[List[PolicyType]] = None
    ) -> List[Policy]:
        """Get policies applicable to the given context"""
        try:
            applicable = []
            
            # Determine applicable scopes
            scopes_to_check = [scope] if scope else list(PolicyScope)
            
            for check_scope in scopes_to_check:
                policy_ids = self.active_policies.get(check_scope, [])
                
                for policy_id in policy_ids:
                    if policy_id not in self.policies:
                        continue
                    
                    policy = self.policies[policy_id]
                    
                    # Check policy type filter
                    if policy_types and policy.policy_type not in policy_types:
                        continue
                    
                    # Check if policy applies to context
                    if await self._policy_applies_to_context(policy, context):
                        applicable.append(policy)
            
            # Sort by priority
            applicable.sort(key=lambda p: p.priority.value)
            
            return applicable
            
        except Exception as e:
            logger.error(f"Failed to get applicable policies: {e}")
            return []
    
    async def _policy_applies_to_context(self, policy: Policy, context: Dict[str, Any]) -> bool:
        """Check if policy applies to the given context"""
        try:
            # Check scope-specific applicability
            if policy.scope == PolicyScope.NODE:
                return 'node_id' in context
            elif policy.scope == PolicyScope.USER:
                return 'user_id' in context
            elif policy.scope == PolicyScope.GROUP:
                return 'group_id' in context or 'user_groups' in context
            elif policy.scope == PolicyScope.WORKLOAD:
                return 'workload_id' in context or 'job_id' in context
            elif policy.scope == PolicyScope.SESSION:
                return 'session_id' in context
            else:  # GLOBAL
                return True
            
        except Exception:
            return True  # Default to applicable
    
    async def _validate_policy(self, policy: Policy) -> bool:
        """Validate policy structure and rules"""
        try:
            # Basic validation
            if not policy.policy_id or not policy.name:
                logger.error("Policy must have ID and name")
                return False
            
            if not policy.rules:
                logger.error("Policy must have at least one rule")
                return False
            
            # Validate rules
            for rule in policy.rules:
                if not await self._validate_rule(rule):
                    return False
            
            # Check for duplicate rule IDs
            rule_ids = [rule.rule_id for rule in policy.rules]
            if len(rule_ids) != len(set(rule_ids)):
                logger.error("Duplicate rule IDs found in policy")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Policy validation failed: {e}")
            return False
    
    async def _validate_rule(self, rule: PolicyRule) -> bool:
        """Validate rule structure and conditions"""
        try:
            if not rule.rule_id or not rule.name:
                logger.error("Rule must have ID and name")
                return False
            
            # Validate conditions
            for condition in rule.conditions:
                if not await self._validate_condition(condition):
                    return False
            
            # Validate logical operator
            if rule.logical_operator.upper() not in ['AND', 'OR']:
                logger.error(f"Invalid logical operator: {rule.logical_operator}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Rule validation failed: {e}")
            return False
    
    async def _validate_condition(self, condition: PolicyCondition) -> bool:
        """Validate condition structure"""
        try:
            if not condition.field or not condition.operator:
                logger.error("Condition must have field and operator")
                return False
            
            if condition.operator not in self.operators:
                logger.error(f"Unknown operator: {condition.operator}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Condition validation failed: {e}")
            return False
    
    async def _detect_conflicts(self, new_policy: Policy) -> List[PolicyConflict]:
        """Detect conflicts with existing policies"""
        try:
            conflicts = []
            
            for existing_id, existing_policy in self.policies.items():
                if existing_policy.scope != new_policy.scope:
                    continue
                
                if existing_policy.policy_type != new_policy.policy_type:
                    continue
                
                # Check for enforcement conflicts
                if (existing_policy.enforcement == PolicyEnforcement.BLOCK and
                    new_policy.enforcement == PolicyEnforcement.ENFORCE):
                    conflict = PolicyConflict(
                        conflict_id=f"{existing_id}_{new_policy.policy_id}_enforcement",
                        policy_ids=[existing_id, new_policy.policy_id],
                        conflict_type="enforcement",
                        description=f"Conflicting enforcement: {existing_policy.name} blocks while {new_policy.name} enforces",
                        severity="high",
                        resolution_strategy="priority",
                        auto_resolvable=True
                    )
                    conflicts.append(conflict)
                
                # Check for overlapping rules (simplified)
                if await self._rules_overlap(existing_policy.rules, new_policy.rules):
                    conflict = PolicyConflict(
                        conflict_id=f"{existing_id}_{new_policy.policy_id}_rules",
                        policy_ids=[existing_id, new_policy.policy_id],
                        conflict_type="action",
                        description=f"Overlapping rules between {existing_policy.name} and {new_policy.name}",
                        severity="medium",
                        resolution_strategy="merge",
                        auto_resolvable=False
                    )
                    conflicts.append(conflict)
            
            return conflicts
            
        except Exception as e:
            logger.error(f"Conflict detection failed: {e}")
            return []
    
    async def _rules_overlap(self, rules1: List[PolicyRule], rules2: List[PolicyRule]) -> bool:
        """Check if two sets of rules overlap (simplified implementation)"""
        try:
            # Simplified overlap detection - check if any conditions target the same fields
            fields1 = set()
            for rule in rules1:
                for condition in rule.conditions:
                    fields1.add(condition.field)
            
            fields2 = set()
            for rule in rules2:
                for condition in rule.conditions:
                    fields2.add(condition.field)
            
            return bool(fields1 & fields2)  # Intersection
            
        except Exception:
            return False
    
    async def _handle_violation(self, policy: Policy, evaluation: PolicyEvaluation):
        """Handle policy violation"""
        try:
            violation = PolicyViolation(
                violation_id=f"{policy.policy_id}_{int(time.time())}_{hash(str(evaluation.context)) % 10000}",
                policy_id=policy.policy_id,
                rule_id=evaluation.rule_id or "unknown",
                severity=policy.priority.name.lower(),
                description=f"Policy '{policy.name}' violated",
                context=evaluation.context,
                timestamp=time.time(),
                user_id=evaluation.context.get('user_id'),
                node_id=evaluation.context.get('node_id')
            )
            
            self.violations.append(violation)
            self.metrics['violations_detected'] += 1
            
            # Execute violation actions
            for action in evaluation.actions_triggered:
                await self._execute_action(action, evaluation.context)
            
            logger.warning(f"Policy violation: {violation.violation_id}")
            
        except Exception as e:
            logger.error(f"Violation handling failed: {e}")
    
    async def _execute_action(self, action: Dict[str, Any], context: Dict[str, Any]):
        """Execute a policy action"""
        try:
            action_type = action.get('type', 'unknown')
            
            if action_type in self.action_handlers:
                # Custom action handler
                await self.action_handlers[action_type](action, context)
            elif action_type == 'log':
                logger.info(f"Policy action log: {action.get('message', 'No message')}")
            elif action_type == 'warning':
                logger.warning(f"Policy warning: {action.get('message', 'No message')}")
            elif action_type == 'audit':
                logger.info(f"Policy audit: {action}")
            elif action_type == 'block':
                logger.error(f"Policy block: {action.get('message', 'Access blocked')}")
            else:
                logger.warning(f"Unknown action type: {action_type}")
                
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
    
    def register_action_handler(self, action_type: str, handler: Callable):
        """Register a custom action handler"""
        self.action_handlers[action_type] = handler
        logger.info(f"Registered action handler for {action_type}")
    
    def _generate_cache_key(
        self,
        context: Dict[str, Any],
        scope: Optional[PolicyScope],
        policy_types: Optional[List[PolicyType]]
    ) -> str:
        """Generate cache key for policy evaluation"""
        try:
            # Create deterministic hash of context and parameters
            context_str = json.dumps(context, sort_keys=True)
            scope_str = scope.value if scope else "all"
            types_str = "_".join(sorted([pt.value for pt in policy_types])) if policy_types else "all"
            
            combined = f"{context_str}_{scope_str}_{types_str}"
            return hashlib.md5(combined.encode()).hexdigest()
            
        except Exception:
            return str(time.time())  # Fallback to timestamp
    
    def _invalidate_cache(self, scope: PolicyScope):
        """Invalidate policy cache for given scope"""
        try:
            keys_to_remove = []
            for key in self.policy_cache.keys():
                # Simple cache invalidation - remove all entries for scope
                # In production, you'd have more sophisticated cache management
                keys_to_remove.append(key)
            
            for key in keys_to_remove[:100]:  # Limit to avoid performance issues
                del self.policy_cache[key]
                
        except Exception as e:
            logger.error(f"Cache invalidation failed: {e}")
    
    async def get_policies(
        self,
        scope: Optional[PolicyScope] = None,
        policy_type: Optional[PolicyType] = None,
        status: Optional[PolicyStatus] = None
    ) -> List[Policy]:
        """Get policies with optional filters"""
        try:
            filtered_policies = []
            
            for policy in self.policies.values():
                if scope and policy.scope != scope:
                    continue
                if policy_type and policy.policy_type != policy_type:
                    continue
                if status and policy.status != status:
                    continue
                
                filtered_policies.append(policy)
            
            return filtered_policies
            
        except Exception as e:
            logger.error(f"Failed to get policies: {e}")
            return []
    
    async def get_violations(
        self,
        policy_id: Optional[str] = None,
        user_id: Optional[str] = None,
        node_id: Optional[str] = None,
        resolved: Optional[bool] = None,
        limit: int = 100
    ) -> List[PolicyViolation]:
        """Get policy violations with optional filters"""
        try:
            filtered_violations = []
            
            for violation in self.violations:
                if policy_id and violation.policy_id != policy_id:
                    continue
                if user_id and violation.user_id != user_id:
                    continue
                if node_id and violation.node_id != node_id:
                    continue
                if resolved is not None and violation.resolved != resolved:
                    continue
                
                filtered_violations.append(violation)
                
                if len(filtered_violations) >= limit:
                    break
            
            return filtered_violations
            
        except Exception as e:
            logger.error(f"Failed to get violations: {e}")
            return []
    
    async def resolve_violation(self, violation_id: str, resolution_notes: str = "") -> bool:
        """Mark a violation as resolved"""
        try:
            for violation in self.violations:
                if violation.violation_id == violation_id:
                    violation.resolved = True
                    violation.resolution_notes = resolution_notes
                    logger.info(f"Violation {violation_id} resolved")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to resolve violation {violation_id}: {e}")
            return False
    
    async def import_policies_from_file(self, file_path: str) -> bool:
        """Import policies from YAML/JSON file"""
        try:
            with open(file_path, 'r') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            policies_data = data.get('policies', [])
            imported_count = 0
            
            for policy_data in policies_data:
                try:
                    policy = self._parse_policy_data(policy_data)
                    if await self.create_policy(policy):
                        imported_count += 1
                except Exception as e:
                    logger.error(f"Failed to import policy: {e}")
            
            logger.info(f"Imported {imported_count} policies from {file_path}")
            return imported_count > 0
            
        except Exception as e:
            logger.error(f"Failed to import policies from {file_path}: {e}")
            return False
    
    def _parse_policy_data(self, data: Dict[str, Any]) -> Policy:
        """Parse policy data from dictionary"""
        try:
            # Parse rules
            rules = []
            for rule_data in data.get('rules', []):
                conditions = []
                for cond_data in rule_data.get('conditions', []):
                    condition = PolicyCondition(
                        field=cond_data['field'],
                        operator=cond_data['operator'],
                        value=cond_data['value'],
                        case_sensitive=cond_data.get('case_sensitive', True)
                    )
                    conditions.append(condition)
                
                rule = PolicyRule(
                    rule_id=rule_data['rule_id'],
                    name=rule_data['name'],
                    description=rule_data.get('description', ''),
                    conditions=conditions,
                    actions=rule_data.get('actions', []),
                    logical_operator=rule_data.get('logical_operator', 'AND'),
                    weight=rule_data.get('weight', 1.0),
                    enabled=rule_data.get('enabled', True)
                )
                rules.append(rule)
            
            # Create policy
            policy = Policy(
                policy_id=data['policy_id'],
                name=data['name'],
                description=data.get('description', ''),
                policy_type=PolicyType(data['policy_type']),
                scope=PolicyScope(data['scope']),
                enforcement=PolicyEnforcement(data['enforcement']),
                priority=PolicyPriority(data['priority']),
                status=PolicyStatus(data.get('status', 'active')),
                rules=rules,
                version=data.get('version', '1.0'),
                created_by=data.get('created_by', 'import'),
                tags=data.get('tags', []),
                metadata=data.get('metadata', {})
            )
            
            return policy
            
        except Exception as e:
            logger.error(f"Failed to parse policy data: {e}")
            raise
    
    async def export_policies_to_file(self, file_path: str, policy_ids: Optional[List[str]] = None) -> bool:
        """Export policies to YAML/JSON file"""
        try:
            policies_to_export = []
            
            for policy_id, policy in self.policies.items():
                if policy_ids and policy_id not in policy_ids:
                    continue
                
                policy_dict = {
                    'policy_id': policy.policy_id,
                    'name': policy.name,
                    'description': policy.description,
                    'policy_type': policy.policy_type.value,
                    'scope': policy.scope.value,
                    'enforcement': policy.enforcement.value,
                    'priority': policy.priority.value,
                    'status': policy.status.value,
                    'version': policy.version,
                    'created_by': policy.created_by,
                    'tags': policy.tags,
                    'metadata': policy.metadata,
                    'rules': []
                }
                
                for rule in policy.rules:
                    rule_dict = {
                        'rule_id': rule.rule_id,
                        'name': rule.name,
                        'description': rule.description,
                        'logical_operator': rule.logical_operator,
                        'weight': rule.weight,
                        'enabled': rule.enabled,
                        'actions': rule.actions,
                        'conditions': []
                    }
                    
                    for condition in rule.conditions:
                        cond_dict = {
                            'field': condition.field,
                            'operator': condition.operator,
                            'value': condition.value,
                            'case_sensitive': condition.case_sensitive
                        }
                        rule_dict['conditions'].append(cond_dict)
                    
                    policy_dict['rules'].append(rule_dict)
                
                policies_to_export.append(policy_dict)
            
            export_data = {'policies': policies_to_export}
            
            with open(file_path, 'w') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    yaml.dump(export_data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported {len(policies_to_export)} policies to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export policies to {file_path}: {e}")
            return False
    
    async def get_policy_metrics(self) -> Dict[str, Any]:
        """Get comprehensive policy engine metrics"""
        try:
            # Calculate additional metrics
            active_policies_by_type = defaultdict(int)
            active_policies_by_scope = defaultdict(int)
            
            for policy in self.policies.values():
                if policy.status == PolicyStatus.ACTIVE:
                    active_policies_by_type[policy.policy_type.value] += 1
                    active_policies_by_scope[policy.scope.value] += 1
            
            unresolved_violations = len([v for v in self.violations if not v.resolved])
            recent_evaluations = len([e for e in self.evaluation_history if time.time() - e.evaluation_time < 3600])
            
            return {
                'policy_overview': {
                    'total_policies': len(self.policies),
                    'active_policies': sum(len(policies) for policies in self.active_policies.values()),
                    'policies_by_type': dict(active_policies_by_type),
                    'policies_by_scope': dict(active_policies_by_scope)
                },
                'evaluation_performance': {
                    'total_evaluations': self.metrics['total_evaluations'],
                    'policy_matches': self.metrics['policy_matches'],
                    'cache_hit_rate': self.metrics['cache_hits'] / max(self.metrics['total_evaluations'], 1),
                    'average_evaluation_time': self.metrics['average_evaluation_time'],
                    'evaluations_last_hour': recent_evaluations
                },
                'violations_and_conflicts': {
                    'total_violations': len(self.violations),
                    'unresolved_violations': unresolved_violations,
                    'conflicts_detected': len(self.conflicts),
                    'violations_last_hour': len([v for v in self.violations if time.time() - v.timestamp < 3600])
                },
                'system_health': {
                    'cache_entries': len(self.policy_cache),
                    'evaluation_history_size': len(self.evaluation_history),
                    'registered_action_handlers': len(self.action_handlers)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating policy metrics: {e}")
            return {}

# Global policy engine instance
_policy_engine_instance = None

def get_policy_engine() -> PolicyEngine:
    """Get or create global policy engine instance"""
    global _policy_engine_instance
    if _policy_engine_instance is None:
        _policy_engine_instance = PolicyEngine()
    return _policy_engine_instance

async def initialize_policy_engine(config: Dict[str, Any] = None):
    """Initialize the global policy engine"""
    global _policy_engine_instance
    _policy_engine_instance = PolicyEngine(config)
    logger.info("Global policy engine initialized")
    return _policy_engine_instance
