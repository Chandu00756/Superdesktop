"""
Omega Super Desktop Console v2.0 - Advanced RBAC Matrix Engine
Enterprise-grade role-based access control with dynamic policies, inheritance, and compliance
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Set, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PermissionLevel(Enum):
    NONE = 0
    READ = 1
    WRITE = 2
    EXECUTE = 4
    ADMIN = 8
    OWNER = 16

class ResourceType(Enum):
    NODE = "node"
    CLUSTER = "cluster"
    SERVICE = "service"
    PLUGIN = "plugin"
    STREAM = "stream"
    MEMORY = "memory"
    NETWORK = "network"
    POLICY = "policy"
    USER = "user"
    ROLE = "role"

class AccessDecision(Enum):
    ALLOW = "allow"
    DENY = "deny"
    AUDIT = "audit"
    CONDITIONAL = "conditional"

@dataclass
class Permission:
    """Individual permission definition"""
    permission_id: str
    name: str
    description: str
    resource_type: ResourceType
    level: PermissionLevel
    conditions: List[str] = field(default_factory=list)
    time_based: bool = False
    location_based: bool = False
    risk_level: int = 1  # 1-10 scale
    compliance_tags: Set[str] = field(default_factory=set)

@dataclass
class Role:
    """Role definition with advanced features"""
    role_id: str
    name: str
    description: str
    permissions: Set[str] = field(default_factory=set)
    parent_roles: Set[str] = field(default_factory=set)
    child_roles: Set[str] = field(default_factory=set)
    constraints: Dict[str, Any] = field(default_factory=dict)
    auto_assign_rules: List[str] = field(default_factory=list)
    max_sessions: int = -1  # -1 = unlimited
    session_timeout: int = 28800  # 8 hours default
    mfa_required: bool = False
    ip_restrictions: List[str] = field(default_factory=list)
    time_restrictions: List[str] = field(default_factory=list)
    temporary_until: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    created_by: str = "system"
    compliance_level: str = "standard"

@dataclass
class RoleAssignment:
    """User role assignment with context"""
    assignment_id: str
    user_id: str
    role_id: str
    assigned_by: str
    assigned_at: float
    expires_at: Optional[float] = None
    conditions: Dict[str, Any] = field(default_factory=dict)
    justification: str = ""
    approval_required: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[float] = None
    active: bool = True

@dataclass
class AccessRequest:
    """Access request for audit and policy evaluation"""
    request_id: str
    user_id: str
    resource_type: ResourceType
    resource_id: str
    action: str
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class AccessResult:
    """Result of access control evaluation"""
    request_id: str
    decision: AccessDecision
    allowed_permissions: Set[PermissionLevel]
    denied_reasons: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    audit_required: bool = False
    risk_score: int = 0
    policy_matches: List[str] = field(default_factory=list)
    evaluation_time_ms: float = 0.0

class RoleHierarchyManager:
    """Manages role inheritance and hierarchy"""
    
    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self.inheritance_cache: Dict[str, Set[str]] = {}
        
    async def add_role(self, role: Role) -> bool:
        """Add a role to the hierarchy"""
        try:
            # Check for circular dependencies
            if await self._would_create_cycle(role.role_id, role.parent_roles):
                raise ValueError("Role assignment would create circular dependency")
                
            self.roles[role.role_id] = role
            
            # Update parent-child relationships
            for parent_id in role.parent_roles:
                if parent_id in self.roles:
                    self.roles[parent_id].child_roles.add(role.role_id)
                    
            # Clear cache for affected roles
            self._clear_inheritance_cache(role.role_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add role {role.role_id}: {e}")
            return False
            
    async def get_effective_permissions(self, role_id: str) -> Set[str]:
        """Get all permissions including inherited ones"""
        try:
            if role_id not in self.roles:
                return set()
                
            # Check cache first
            if role_id in self.inheritance_cache:
                inherited_roles = self.inheritance_cache[role_id]
            else:
                inherited_roles = await self._compute_inherited_roles(role_id)
                self.inheritance_cache[role_id] = inherited_roles
                
            # Collect all permissions
            all_permissions = set()
            for role_id_in_hierarchy in inherited_roles:
                if role_id_in_hierarchy in self.roles:
                    all_permissions.update(self.roles[role_id_in_hierarchy].permissions)
                    
            return all_permissions
            
        except Exception as e:
            logger.error(f"Failed to get effective permissions for role {role_id}: {e}")
            return set()
            
    async def _compute_inherited_roles(self, role_id: str) -> Set[str]:
        """Compute all inherited roles (including self)"""
        visited = set()
        to_visit = {role_id}
        
        while to_visit:
            current = to_visit.pop()
            if current in visited or current not in self.roles:
                continue
                
            visited.add(current)
            role = self.roles[current]
            to_visit.update(role.parent_roles - visited)
            
        return visited
        
    async def _would_create_cycle(self, role_id: str, parent_roles: Set[str]) -> bool:
        """Check if adding parent roles would create a cycle"""
        for parent_id in parent_roles:
            if parent_id == role_id:
                return True
                
            # Check if role_id is already an ancestor of parent_id
            ancestors = await self._compute_inherited_roles(parent_id)
            if role_id in ancestors:
                return True
                
        return False
        
    def _clear_inheritance_cache(self, role_id: str):
        """Clear inheritance cache for role and its descendants"""
        to_clear = {role_id}
        
        # Find all roles that might be affected
        for rid, role in self.roles.items():
            if role_id in role.parent_roles:
                to_clear.add(rid)
                
        # Clear cache entries
        for rid in to_clear:
            self.inheritance_cache.pop(rid, None)

class PolicyEngine:
    """Advanced policy evaluation engine"""
    
    def __init__(self):
        self.policies: List[Dict[str, Any]] = []
        self.condition_evaluators: Dict[str, Callable] = {}
        
    async def evaluate_access(self, request: AccessRequest, user_roles: Set[str], 
                            permissions: Set[str]) -> AccessResult:
        """Evaluate access request against policies"""
        try:
            start_time = time.time()
            result = AccessResult(
                request_id=request.request_id,
                decision=AccessDecision.DENY,
                allowed_permissions=set()
            )
            
            # Base permission check
            required_permission = await self._get_required_permission(request)
            if required_permission in permissions:
                result.decision = AccessDecision.ALLOW
                result.allowed_permissions.add(required_permission)
            else:
                result.denied_reasons.append(f"Missing required permission: {required_permission}")
                
            # Apply policies
            for policy in self.policies:
                policy_result = await self._evaluate_policy(policy, request, user_roles)
                if policy_result['matches']:
                    result.policy_matches.append(policy['id'])
                    
                    if policy_result['decision'] == AccessDecision.DENY:
                        result.decision = AccessDecision.DENY
                        result.denied_reasons.extend(policy_result['reasons'])
                        break
                    elif policy_result['decision'] == AccessDecision.AUDIT:
                        result.audit_required = True
                        
            # Calculate risk score
            result.risk_score = await self._calculate_risk_score(request, user_roles)
            
            # High-risk operations require audit
            if result.risk_score > 7:
                result.audit_required = True
                
            result.evaluation_time_ms = (time.time() - start_time) * 1000
            return result
            
        except Exception as e:
            logger.error(f"Policy evaluation failed: {e}")
            return AccessResult(
                request_id=request.request_id,
                decision=AccessDecision.DENY,
                allowed_permissions=set(),
                denied_reasons=[f"Evaluation error: {str(e)}"]
            )
            
    async def _get_required_permission(self, request: AccessRequest) -> PermissionLevel:
        """Determine required permission level for request"""
        action_permissions = {
            'read': PermissionLevel.READ,
            'list': PermissionLevel.READ,
            'view': PermissionLevel.READ,
            'create': PermissionLevel.WRITE,
            'update': PermissionLevel.WRITE,
            'modify': PermissionLevel.WRITE,
            'delete': PermissionLevel.WRITE,
            'execute': PermissionLevel.EXECUTE,
            'run': PermissionLevel.EXECUTE,
            'admin': PermissionLevel.ADMIN,
            'manage': PermissionLevel.ADMIN
        }
        
        return action_permissions.get(request.action.lower(), PermissionLevel.READ)
        
    async def _evaluate_policy(self, policy: Dict[str, Any], request: AccessRequest, 
                             user_roles: Set[str]) -> Dict[str, Any]:
        """Evaluate a single policy"""
        try:
            # Check if policy applies
            if not await self._policy_applies(policy, request, user_roles):
                return {'matches': False, 'decision': AccessDecision.ALLOW, 'reasons': []}
                
            # Evaluate conditions
            conditions_met = True
            reasons = []
            
            for condition in policy.get('conditions', []):
                if not await self._evaluate_condition(condition, request):
                    conditions_met = False
                    reasons.append(f"Policy condition not met: {condition}")
                    
            decision = AccessDecision.ALLOW if conditions_met else AccessDecision.DENY
            if policy.get('audit_required', False):
                decision = AccessDecision.AUDIT
                
            return {
                'matches': True,
                'decision': decision,
                'reasons': reasons
            }
            
        except Exception as e:
            logger.error(f"Policy evaluation error: {e}")
            return {'matches': False, 'decision': AccessDecision.DENY, 'reasons': [str(e)]}
            
    async def _policy_applies(self, policy: Dict[str, Any], request: AccessRequest, 
                            user_roles: Set[str]) -> bool:
        """Check if policy applies to this request"""
        # Check resource type
        if 'resource_types' in policy:
            if request.resource_type.value not in policy['resource_types']:
                return False
                
        # Check user roles
        if 'roles' in policy:
            if not user_roles.intersection(set(policy['roles'])):
                return False
                
        # Check actions
        if 'actions' in policy:
            if request.action not in policy['actions']:
                return False
                
        return True
        
    async def _evaluate_condition(self, condition: str, request: AccessRequest) -> bool:
        """Evaluate a policy condition"""
        # Simplified condition evaluation
        # In production, this would be a full expression evaluator
        if condition.startswith('time_range:'):
            return await self._evaluate_time_condition(condition, request)
        elif condition.startswith('ip_range:'):
            return await self._evaluate_ip_condition(condition, request)
        elif condition.startswith('mfa_required'):
            return request.context.get('mfa_verified', False)
            
        return True
        
    async def _evaluate_time_condition(self, condition: str, request: AccessRequest) -> bool:
        """Evaluate time-based condition"""
        # Simplified time range check
        return True  # Always allow for demo
        
    async def _evaluate_ip_condition(self, condition: str, request: AccessRequest) -> bool:
        """Evaluate IP-based condition"""
        # Simplified IP range check
        return True  # Always allow for demo
        
    async def _calculate_risk_score(self, request: AccessRequest, user_roles: Set[str]) -> int:
        """Calculate risk score for the request"""
        risk_score = 0
        
        # Base risk by resource type
        resource_risks = {
            ResourceType.CLUSTER: 8,
            ResourceType.NODE: 6,
            ResourceType.POLICY: 7,
            ResourceType.USER: 5,
            ResourceType.ROLE: 6,
            ResourceType.SERVICE: 4,
            ResourceType.PLUGIN: 5,
            ResourceType.STREAM: 3,
            ResourceType.MEMORY: 4,
            ResourceType.NETWORK: 5
        }
        
        risk_score += resource_risks.get(request.resource_type, 3)
        
        # Action risk
        action_risks = {
            'delete': 3,
            'admin': 3,
            'execute': 2,
            'write': 1,
            'read': 0
        }
        
        risk_score += action_risks.get(request.action.lower(), 1)
        
        # Time-based risk (higher risk outside business hours)
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:
            risk_score += 2
            
        return min(risk_score, 10)

class AdvancedRBACMatrix:
    """Enterprise-grade RBAC matrix with advanced features"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.db_path = self.config.get('db_path', 'backend/rbac_matrix.db')
        
        # Core components
        self.permissions: Dict[str, Permission] = {}
        self.hierarchy_manager = RoleHierarchyManager()
        self.policy_engine = PolicyEngine()
        self.role_assignments: Dict[str, List[RoleAssignment]] = {}
        
        # Caching and performance
        self.permission_cache: Dict[str, Set[str]] = {}
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutes
        self.cache_timestamps: Dict[str, float] = {}
        
        # Auditing
        self.access_log: List[Dict[str, Any]] = []
        self.audit_enabled = self.config.get('audit_enabled', True)
        
        # Metrics
        self.metrics = {
            'total_permissions': 0,
            'total_roles': 0,
            'total_assignments': 0,
            'access_checks': 0,
            'access_denied': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
    async def initialize(self) -> bool:
        """Initialize the RBAC matrix"""
        try:
            logger.info("Initializing Advanced RBAC Matrix...")
            
            # Setup database
            await self._setup_database()
            
            # Load built-in permissions and roles
            await self._load_builtin_permissions()
            await self._load_builtin_roles()
            
            # Setup policies
            await self._setup_default_policies()
            
            logger.info("Advanced RBAC Matrix initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"RBAC matrix initialization failed: {e}")
            return False
            
    async def create_permission(self, permission: Permission) -> bool:
        """Create a new permission"""
        try:
            self.permissions[permission.permission_id] = permission
            self.metrics['total_permissions'] += 1
            
            # Save to database
            await self._save_permission(permission)
            
            logger.info(f"Permission created: {permission.permission_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create permission: {e}")
            return False
            
    async def create_role(self, role: Role) -> bool:
        """Create a new role"""
        try:
            success = await self.hierarchy_manager.add_role(role)
            if success:
                self.metrics['total_roles'] += 1
                await self._save_role(role)
                logger.info(f"Role created: {role.role_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"Failed to create role: {e}")
            return False
            
    async def assign_role(self, assignment: RoleAssignment) -> bool:
        """Assign a role to a user"""
        try:
            if assignment.user_id not in self.role_assignments:
                self.role_assignments[assignment.user_id] = []
                
            self.role_assignments[assignment.user_id].append(assignment)
            self.metrics['total_assignments'] += 1
            
            # Clear user's permission cache
            self._clear_user_cache(assignment.user_id)
            
            # Save to database
            await self._save_assignment(assignment)
            
            logger.info(f"Role {assignment.role_id} assigned to user {assignment.user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to assign role: {e}")
            return False
            
    async def check_access(self, request: AccessRequest) -> AccessResult:
        """Check if user has access to perform the requested action"""
        try:
            self.metrics['access_checks'] += 1
            
            # Get user roles
            user_roles = await self._get_user_roles(request.user_id)
            if not user_roles:
                self.metrics['access_denied'] += 1
                return AccessResult(
                    request_id=request.request_id,
                    decision=AccessDecision.DENY,
                    allowed_permissions=set(),
                    denied_reasons=["No roles assigned to user"]
                )
                
            # Get effective permissions
            permissions = await self._get_user_permissions(request.user_id)
            
            # Evaluate access using policy engine
            result = await self.policy_engine.evaluate_access(request, user_roles, permissions)
            
            # Log access attempt
            if self.audit_enabled:
                await self._log_access_attempt(request, result)
                
            if result.decision == AccessDecision.DENY:
                self.metrics['access_denied'] += 1
                
            return result
            
        except Exception as e:
            logger.error(f"Access check failed: {e}")
            self.metrics['access_denied'] += 1
            return AccessResult(
                request_id=request.request_id,
                decision=AccessDecision.DENY,
                allowed_permissions=set(),
                denied_reasons=[f"Access check error: {str(e)}"]
            )
            
    async def get_user_effective_permissions(self, user_id: str) -> Set[str]:
        """Get all effective permissions for a user"""
        return await self._get_user_permissions(user_id)
        
    async def get_rbac_metrics(self) -> Dict[str, Any]:
        """Get comprehensive RBAC metrics"""
        try:
            active_assignments = 0
            expired_assignments = 0
            current_time = time.time()
            
            for assignments in self.role_assignments.values():
                for assignment in assignments:
                    if assignment.active:
                        if assignment.expires_at and assignment.expires_at < current_time:
                            expired_assignments += 1
                        else:
                            active_assignments += 1
                            
            return {
                'rbac_status': {
                    'total_permissions': len(self.permissions),
                    'total_roles': len(self.hierarchy_manager.roles),
                    'total_users_with_roles': len(self.role_assignments),
                    'active_assignments': active_assignments,
                    'expired_assignments': expired_assignments
                },
                'performance': {
                    'total_access_checks': self.metrics['access_checks'],
                    'access_denied_count': self.metrics['access_denied'],
                    'cache_hit_ratio': (self.metrics['cache_hits'] / 
                                      max(self.metrics['cache_hits'] + self.metrics['cache_misses'], 1)),
                    'avg_evaluation_time': 0.0  # Would calculate from recent evaluations
                },
                'security': {
                    'audit_enabled': self.audit_enabled,
                    'audit_log_size': len(self.access_log),
                    'high_risk_operations': sum(1 for log in self.access_log[-1000:] 
                                              if log.get('risk_score', 0) > 7)
                },
                'hierarchy': {
                    'max_role_depth': await self._calculate_max_role_depth(),
                    'orphaned_roles': await self._count_orphaned_roles(),
                    'circular_dependencies': 0  # Would detect circular deps
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get RBAC metrics: {e}")
            return {}
            
    async def _get_user_roles(self, user_id: str) -> Set[str]:
        """Get active roles for a user"""
        if user_id not in self.role_assignments:
            return set()
            
        active_roles = set()
        current_time = time.time()
        
        for assignment in self.role_assignments[user_id]:
            if (assignment.active and 
                (not assignment.expires_at or assignment.expires_at > current_time)):
                active_roles.add(assignment.role_id)
                
        return active_roles
        
    async def _get_user_permissions(self, user_id: str) -> Set[str]:
        """Get effective permissions for a user with caching"""
        # Check cache
        cache_key = f"permissions:{user_id}"
        if (cache_key in self.permission_cache and 
            cache_key in self.cache_timestamps and
            time.time() - self.cache_timestamps[cache_key] < self.cache_ttl):
            self.metrics['cache_hits'] += 1
            return self.permission_cache[cache_key]
            
        self.metrics['cache_misses'] += 1
        
        # Calculate permissions
        user_roles = await self._get_user_roles(user_id)
        all_permissions = set()
        
        for role_id in user_roles:
            role_permissions = await self.hierarchy_manager.get_effective_permissions(role_id)
            all_permissions.update(role_permissions)
            
        # Cache result
        self.permission_cache[cache_key] = all_permissions
        self.cache_timestamps[cache_key] = time.time()
        
        return all_permissions
        
    def _clear_user_cache(self, user_id: str):
        """Clear cached permissions for a user"""
        cache_key = f"permissions:{user_id}"
        self.permission_cache.pop(cache_key, None)
        self.cache_timestamps.pop(cache_key, None)
        
    async def _setup_database(self):
        """Setup SQLite database for persistence"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS permissions (
                permission_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                resource_type TEXT NOT NULL,
                level INTEGER NOT NULL,
                conditions TEXT,
                risk_level INTEGER DEFAULT 1,
                compliance_tags TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            );
            
            CREATE TABLE IF NOT EXISTS roles (
                role_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                permissions TEXT,
                parent_roles TEXT,
                constraints TEXT,
                max_sessions INTEGER DEFAULT -1,
                session_timeout INTEGER DEFAULT 28800,
                mfa_required BOOLEAN DEFAULT FALSE,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                created_by TEXT DEFAULT 'system'
            );
            
            CREATE TABLE IF NOT EXISTS role_assignments (
                assignment_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                role_id TEXT NOT NULL,
                assigned_by TEXT NOT NULL,
                assigned_at REAL DEFAULT (strftime('%s', 'now')),
                expires_at REAL,
                active BOOLEAN DEFAULT TRUE,
                justification TEXT
            );
            
            CREATE TABLE IF NOT EXISTS access_log (
                log_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                action TEXT NOT NULL,
                decision TEXT NOT NULL,
                risk_score INTEGER,
                timestamp REAL DEFAULT (strftime('%s', 'now')),
                context TEXT
            );
        """)
        
        conn.commit()
        conn.close()
        
    async def _load_builtin_permissions(self):
        """Load built-in system permissions"""
        builtin_permissions = [
            Permission("nodes:view", "View Nodes", "View node information", ResourceType.NODE, PermissionLevel.READ),
            Permission("nodes:manage", "Manage Nodes", "Manage node configuration", ResourceType.NODE, PermissionLevel.ADMIN),
            Permission("cluster:view", "View Cluster", "View cluster status", ResourceType.CLUSTER, PermissionLevel.READ),
            Permission("cluster:admin", "Cluster Admin", "Full cluster administration", ResourceType.CLUSTER, PermissionLevel.ADMIN),
            Permission("services:view", "View Services", "View service status", ResourceType.SERVICE, PermissionLevel.READ),
            Permission("services:manage", "Manage Services", "Manage services", ResourceType.SERVICE, PermissionLevel.WRITE),
            Permission("plugins:view", "View Plugins", "View installed plugins", ResourceType.PLUGIN, PermissionLevel.READ),
            Permission("plugins:install", "Install Plugins", "Install new plugins", ResourceType.PLUGIN, PermissionLevel.WRITE),
            Permission("streams:view", "View Streams", "View active streams", ResourceType.STREAM, PermissionLevel.READ),
            Permission("streams:create", "Create Streams", "Create new streams", ResourceType.STREAM, PermissionLevel.WRITE),
            Permission("users:view", "View Users", "View user information", ResourceType.USER, PermissionLevel.READ),
            Permission("users:admin", "User Admin", "Full user administration", ResourceType.USER, PermissionLevel.ADMIN),
            Permission("roles:view", "View Roles", "View role definitions", ResourceType.ROLE, PermissionLevel.READ),
            Permission("roles:admin", "Role Admin", "Full role administration", ResourceType.ROLE, PermissionLevel.ADMIN),
        ]
        
        for permission in builtin_permissions:
            await self.create_permission(permission)
            
    async def _load_builtin_roles(self):
        """Load built-in system roles"""
        builtin_roles = [
            Role(
                role_id="system_admin",
                name="System Administrator",
                description="Full system access",
                permissions={
                    "nodes:manage", "cluster:admin", "services:manage", 
                    "plugins:install", "streams:create", "users:admin", "roles:admin"
                },
                mfa_required=True,
                compliance_level="high"
            ),
            Role(
                role_id="cluster_operator",
                name="Cluster Operator",
                description="Cluster operations access",
                permissions={
                    "nodes:view", "cluster:view", "services:manage", "streams:view"
                },
                parent_roles=set(),
                compliance_level="standard"
            ),
            Role(
                role_id="service_manager",
                name="Service Manager",
                description="Service management access",
                permissions={
                    "services:view", "services:manage", "plugins:view"
                },
                compliance_level="standard"
            ),
            Role(
                role_id="user_manager",
                name="User Manager",
                description="User management access",
                permissions={
                    "users:view", "users:admin", "roles:view"
                },
                mfa_required=True,
                compliance_level="high"
            ),
            Role(
                role_id="viewer",
                name="Viewer",
                description="Read-only access",
                permissions={
                    "nodes:view", "cluster:view", "services:view", 
                    "plugins:view", "streams:view", "users:view", "roles:view"
                },
                compliance_level="standard"
            )
        ]
        
        for role in builtin_roles:
            await self.create_role(role)
            
    async def _setup_default_policies(self):
        """Setup default security policies"""
        default_policies = [
            {
                'id': 'mfa_required_admin',
                'name': 'MFA Required for Admin Actions',
                'resource_types': ['cluster', 'user', 'role'],
                'actions': ['admin', 'delete', 'create'],
                'conditions': ['mfa_required'],
                'audit_required': True
            },
            {
                'id': 'time_restricted_admin',
                'name': 'Time Restricted Admin Actions',
                'resource_types': ['cluster'],
                'actions': ['admin'],
                'conditions': ['time_range:business_hours'],
                'audit_required': True
            },
            {
                'id': 'high_risk_audit',
                'name': 'High Risk Action Auditing',
                'resource_types': ['cluster', 'node', 'policy'],
                'actions': ['delete', 'admin'],
                'audit_required': True
            }
        ]
        
        self.policy_engine.policies = default_policies
        
    async def _save_permission(self, permission: Permission):
        """Save permission to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO permissions 
            (permission_id, name, description, resource_type, level, conditions, risk_level, compliance_tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            permission.permission_id,
            permission.name,
            permission.description,
            permission.resource_type.value,
            permission.level.value,
            json.dumps(permission.conditions),
            permission.risk_level,
            json.dumps(list(permission.compliance_tags))
        ))
        
        conn.commit()
        conn.close()
        
    async def _save_role(self, role: Role):
        """Save role to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO roles 
            (role_id, name, description, permissions, parent_roles, constraints, 
             max_sessions, session_timeout, mfa_required, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            role.role_id,
            role.name,
            role.description,
            json.dumps(list(role.permissions)),
            json.dumps(list(role.parent_roles)),
            json.dumps(role.constraints),
            role.max_sessions,
            role.session_timeout,
            role.mfa_required,
            role.created_by
        ))
        
        conn.commit()
        conn.close()
        
    async def _save_assignment(self, assignment: RoleAssignment):
        """Save role assignment to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO role_assignments 
            (assignment_id, user_id, role_id, assigned_by, assigned_at, expires_at, active, justification)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            assignment.assignment_id,
            assignment.user_id,
            assignment.role_id,
            assignment.assigned_by,
            assignment.assigned_at,
            assignment.expires_at,
            assignment.active,
            assignment.justification
        ))
        
        conn.commit()
        conn.close()
        
    async def _log_access_attempt(self, request: AccessRequest, result: AccessResult):
        """Log access attempt for auditing"""
        log_entry = {
            'log_id': str(uuid.uuid4()),
            'user_id': request.user_id,
            'resource_type': request.resource_type.value,
            'resource_id': request.resource_id,
            'action': request.action,
            'decision': result.decision.value,
            'risk_score': result.risk_score,
            'timestamp': request.timestamp,
            'context': request.context
        }
        
        self.access_log.append(log_entry)
        
        # Keep only recent entries in memory
        if len(self.access_log) > 10000:
            self.access_log = self.access_log[-5000:]
            
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO access_log 
            (log_id, user_id, resource_type, resource_id, action, decision, risk_score, timestamp, context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            log_entry['log_id'],
            log_entry['user_id'],
            log_entry['resource_type'],
            log_entry['resource_id'],
            log_entry['action'],
            log_entry['decision'],
            log_entry['risk_score'],
            log_entry['timestamp'],
            json.dumps(log_entry['context'])
        ))
        
        conn.commit()
        conn.close()
        
    async def _calculate_max_role_depth(self) -> int:
        """Calculate maximum role hierarchy depth"""
        max_depth = 0
        
        for role_id in self.hierarchy_manager.roles:
            depth = await self._calculate_role_depth(role_id, set())
            max_depth = max(max_depth, depth)
            
        return max_depth
        
    async def _calculate_role_depth(self, role_id: str, visited: Set[str]) -> int:
        """Calculate depth for a specific role"""
        if role_id in visited or role_id not in self.hierarchy_manager.roles:
            return 0
            
        visited.add(role_id)
        role = self.hierarchy_manager.roles[role_id]
        
        if not role.parent_roles:
            return 1
            
        max_parent_depth = 0
        for parent_id in role.parent_roles:
            parent_depth = await self._calculate_role_depth(parent_id, visited.copy())
            max_parent_depth = max(max_parent_depth, parent_depth)
            
        return max_parent_depth + 1
        
    async def _count_orphaned_roles(self) -> int:
        """Count roles with no parent or child relationships"""
        orphaned = 0
        
        for role in self.hierarchy_manager.roles.values():
            if not role.parent_roles and not role.child_roles:
                orphaned += 1
                
        return orphaned

# Global instance
_rbac_matrix: Optional[AdvancedRBACMatrix] = None

async def initialize_rbac_matrix(config: Optional[Dict[str, Any]] = None) -> bool:
    """Initialize the global RBAC matrix"""
    global _rbac_matrix
    try:
        _rbac_matrix = AdvancedRBACMatrix(config)
        return await _rbac_matrix.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize RBAC matrix: {e}")
        return False

def get_rbac_matrix() -> AdvancedRBACMatrix:
    """Get the global RBAC matrix instance"""
    global _rbac_matrix
    if _rbac_matrix is None:
        raise RuntimeError("RBAC matrix not initialized. Call initialize_rbac_matrix() first.")
    return _rbac_matrix

async def shutdown_rbac_matrix():
    """Shutdown the global RBAC matrix"""
    global _rbac_matrix
    if _rbac_matrix:
        _rbac_matrix = None
