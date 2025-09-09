"""
Omega Super Desktop Console v2.0 - Compliance Framework Engine
Enterprise compliance management with regulatory reporting and audit automation
"""

import asyncio
import json
import logging
import time
import uuid
import hashlib
from typing import Dict, List, Optional, Set, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
from pathlib import Path
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)

class ComplianceStandard(Enum):
    SOX = "sox"              # Sarbanes-Oxley Act
    GDPR = "gdpr"            # General Data Protection Regulation
    HIPAA = "hipaa"          # Health Insurance Portability and Accountability Act
    PCI_DSS = "pci_dss"      # Payment Card Industry Data Security Standard
    ISO27001 = "iso27001"    # ISO/IEC 27001
    NIST = "nist"            # NIST Cybersecurity Framework
    CCPA = "ccpa"            # California Consumer Privacy Act
    SOC2 = "soc2"            # SOC 2 Type II
    FISMA = "fisma"          # Federal Information Security Management Act

class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    PENDING_REVIEW = "pending_review"

class AuditEventType(Enum):
    LOGIN = "login"
    LOGOUT = "logout"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    PERMISSION_CHANGE = "permission_change"
    CONFIGURATION_CHANGE = "configuration_change"
    SYSTEM_ACCESS = "system_access"
    FILE_ACCESS = "file_access"
    ADMIN_ACTION = "admin_action"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ReportFormat(Enum):
    PDF = "pdf"
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    HTML = "html"

@dataclass
class CompliancePolicy:
    """Compliance policy definition"""
    policy_id: str
    name: str
    description: str
    standard: ComplianceStandard
    requirements: List[str]
    controls: List[Dict[str, Any]]
    automated_checks: List[Dict[str, Any]]
    manual_checks: List[Dict[str, Any]]
    risk_level: RiskLevel
    owner: str
    review_frequency: int = 90  # days
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    last_review: Optional[float] = None

@dataclass
class AuditEvent:
    """Audit trail event"""
    event_id: str
    event_type: AuditEventType
    user_id: str
    user_name: str
    timestamp: float
    resource: str
    action: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    risk_score: float = 0.0
    compliance_relevant: bool = True

@dataclass
class ComplianceAssessment:
    """Compliance assessment result"""
    assessment_id: str
    policy_id: str
    standard: ComplianceStandard
    status: ComplianceStatus
    score: float  # 0-100
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    evidence: List[Dict[str, Any]]
    assessor: str
    timestamp: float = field(default_factory=time.time)
    next_assessment: Optional[float] = None

@dataclass
class ComplianceReport:
    """Compliance report"""
    report_id: str
    name: str
    description: str
    standards: List[ComplianceStandard]
    time_range: Tuple[float, float]
    generated_by: str
    format: ReportFormat
    content: Dict[str, Any]
    file_path: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None

class PolicyManager:
    """Manage compliance policies"""
    
    def __init__(self):
        self.policies: Dict[str, CompliancePolicy] = {}
        self.policy_templates: Dict[ComplianceStandard, Dict[str, Any]] = {}
        
    async def initialize(self):
        """Initialize with standard policy templates"""
        try:
            await self._load_policy_templates()
            logger.info("Policy manager initialized with standard templates")
            
        except Exception as e:
            logger.error(f"Policy manager initialization failed: {e}")
            
    async def _load_policy_templates(self):
        """Load standard compliance policy templates"""
        try:
            # GDPR template
            self.policy_templates[ComplianceStandard.GDPR] = {
                'data_protection': {
                    'name': 'Data Protection',
                    'requirements': [
                        'Implement appropriate technical and organizational measures',
                        'Ensure data processing transparency',
                        'Provide data subject rights mechanisms',
                        'Maintain data processing records'
                    ],
                    'controls': [
                        {'control_id': 'gdpr_001', 'description': 'Data encryption at rest and in transit'},
                        {'control_id': 'gdpr_002', 'description': 'Access control and authentication'},
                        {'control_id': 'gdpr_003', 'description': 'Data retention policies'},
                        {'control_id': 'gdpr_004', 'description': 'Consent management'}
                    ]
                },
                'privacy_by_design': {
                    'name': 'Privacy by Design',
                    'requirements': [
                        'Data protection by design and by default',
                        'Privacy impact assessments',
                        'Data minimization principles'
                    ]
                }
            }
            
            # SOX template
            self.policy_templates[ComplianceStandard.SOX] = {
                'financial_controls': {
                    'name': 'Financial Controls',
                    'requirements': [
                        'Establish internal control framework',
                        'Document financial processes',
                        'Implement segregation of duties',
                        'Maintain audit trails'
                    ],
                    'controls': [
                        {'control_id': 'sox_001', 'description': 'Financial data access controls'},
                        {'control_id': 'sox_002', 'description': 'Change management procedures'},
                        {'control_id': 'sox_003', 'description': 'Backup and recovery controls'},
                        {'control_id': 'sox_004', 'description': 'Audit logging and monitoring'}
                    ]
                }
            }
            
            # ISO 27001 template
            self.policy_templates[ComplianceStandard.ISO27001] = {
                'information_security': {
                    'name': 'Information Security Management',
                    'requirements': [
                        'Establish ISMS framework',
                        'Conduct risk assessments',
                        'Implement security controls',
                        'Monitor and review effectiveness'
                    ],
                    'controls': [
                        {'control_id': 'iso_001', 'description': 'Information security policies'},
                        {'control_id': 'iso_002', 'description': 'Asset management'},
                        {'control_id': 'iso_003', 'description': 'Access control management'},
                        {'control_id': 'iso_004', 'description': 'Incident management'}
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to load policy templates: {e}")
            
    async def create_policy_from_template(self, standard: ComplianceStandard, 
                                        template_name: str, owner: str) -> Optional[str]:
        """Create policy from standard template"""
        try:
            if standard not in self.policy_templates:
                return None
                
            template = self.policy_templates[standard].get(template_name)
            if not template:
                return None
                
            policy_id = str(uuid.uuid4())
            
            policy = CompliancePolicy(
                policy_id=policy_id,
                name=template['name'],
                description=f"Auto-generated {standard.value.upper()} policy for {template_name}",
                standard=standard,
                requirements=template.get('requirements', []),
                controls=template.get('controls', []),
                automated_checks=template.get('automated_checks', []),
                manual_checks=template.get('manual_checks', []),
                risk_level=RiskLevel.MEDIUM,
                owner=owner
            )
            
            self.policies[policy_id] = policy
            logger.info(f"Created policy {policy.name} from template")
            
            return policy_id
            
        except Exception as e:
            logger.error(f"Failed to create policy from template: {e}")
            return None
            
    async def add_policy(self, policy: CompliancePolicy) -> bool:
        """Add custom compliance policy"""
        try:
            self.policies[policy.policy_id] = policy
            logger.info(f"Added compliance policy: {policy.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add policy: {e}")
            return False
            
    async def get_policies_by_standard(self, standard: ComplianceStandard) -> List[CompliancePolicy]:
        """Get all policies for a specific standard"""
        return [policy for policy in self.policies.values() if policy.standard == standard]
        
    async def get_overdue_reviews(self) -> List[CompliancePolicy]:
        """Get policies that need review"""
        try:
            current_time = time.time()
            overdue = []
            
            for policy in self.policies.values():
                if not policy.enabled:
                    continue
                    
                if policy.last_review is None:
                    overdue.append(policy)
                else:
                    next_review = policy.last_review + (policy.review_frequency * 86400)
                    if current_time > next_review:
                        overdue.append(policy)
                        
            return overdue
            
        except Exception as e:
            logger.error(f"Failed to get overdue reviews: {e}")
            return []

class AuditTrail:
    """Comprehensive audit trail management"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.events: List[AuditEvent] = []
        self.event_buffer: List[AuditEvent] = []
        self.buffer_size = 1000
        
    async def log_event(self, event: AuditEvent) -> bool:
        """Log audit event"""
        try:
            # Calculate risk score
            event.risk_score = await self._calculate_risk_score(event)
            
            # Add to buffer
            self.event_buffer.append(event)
            
            # Flush buffer if full
            if len(self.event_buffer) >= self.buffer_size:
                await self._flush_buffer()
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            return False
            
    async def _calculate_risk_score(self, event: AuditEvent) -> float:
        """Calculate risk score for audit event"""
        try:
            base_score = 0.0
            
            # Risk based on event type
            risk_weights = {
                AuditEventType.LOGIN: 0.1,
                AuditEventType.LOGOUT: 0.0,
                AuditEventType.DATA_ACCESS: 0.3,
                AuditEventType.DATA_MODIFICATION: 0.7,
                AuditEventType.DATA_DELETION: 0.9,
                AuditEventType.PERMISSION_CHANGE: 0.8,
                AuditEventType.CONFIGURATION_CHANGE: 0.6,
                AuditEventType.SYSTEM_ACCESS: 0.5,
                AuditEventType.FILE_ACCESS: 0.2,
                AuditEventType.ADMIN_ACTION: 0.8
            }
            
            base_score = risk_weights.get(event.event_type, 0.3)
            
            # Increase risk for sensitive resources
            sensitive_keywords = ['password', 'key', 'secret', 'credential', 'admin', 'root']
            for keyword in sensitive_keywords:
                if keyword.lower() in event.resource.lower():
                    base_score += 0.2
                    break
                    
            # Increase risk for off-hours access
            hour = time.localtime(event.timestamp).tm_hour
            if hour < 6 or hour > 22:  # Outside business hours
                base_score += 0.1
                
            return min(1.0, base_score)
            
        except Exception as e:
            logger.error(f"Risk score calculation failed: {e}")
            return 0.0
            
    async def _flush_buffer(self):
        """Flush event buffer to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for event in self.event_buffer:
                cursor.execute("""
                    INSERT INTO audit_events 
                    (event_id, event_type, user_id, user_name, timestamp, resource, 
                     action, details, ip_address, user_agent, session_id, risk_score, 
                     compliance_relevant)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id, event.event_type.value, event.user_id, event.user_name,
                    event.timestamp, event.resource, event.action, json.dumps(event.details),
                    event.ip_address, event.user_agent, event.session_id, event.risk_score,
                    event.compliance_relevant
                ))
                
            conn.commit()
            conn.close()
            
            logger.debug(f"Flushed {len(self.event_buffer)} audit events to database")
            self.event_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush audit buffer: {e}")
            
    async def search_events(self, criteria: Dict[str, Any], 
                          limit: int = 1000) -> List[AuditEvent]:
        """Search audit events with criteria"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build query
            where_clauses = []
            params = []
            
            if 'user_id' in criteria:
                where_clauses.append("user_id = ?")
                params.append(criteria['user_id'])
                
            if 'event_type' in criteria:
                where_clauses.append("event_type = ?")
                params.append(criteria['event_type'])
                
            if 'start_time' in criteria:
                where_clauses.append("timestamp >= ?")
                params.append(criteria['start_time'])
                
            if 'end_time' in criteria:
                where_clauses.append("timestamp <= ?")
                params.append(criteria['end_time'])
                
            if 'resource' in criteria:
                where_clauses.append("resource LIKE ?")
                params.append(f"%{criteria['resource']}%")
                
            if 'min_risk_score' in criteria:
                where_clauses.append("risk_score >= ?")
                params.append(criteria['min_risk_score'])
                
            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            query = f"""
                SELECT event_id, event_type, user_id, user_name, timestamp, resource,
                       action, details, ip_address, user_agent, session_id, risk_score,
                       compliance_relevant
                FROM audit_events 
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            """
            
            params.append(limit)
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            events = []
            for row in rows:
                event = AuditEvent(
                    event_id=row[0],
                    event_type=AuditEventType(row[1]),
                    user_id=row[2],
                    user_name=row[3],
                    timestamp=row[4],
                    resource=row[5],
                    action=row[6],
                    details=json.loads(row[7]) if row[7] else {},
                    ip_address=row[8],
                    user_agent=row[9],
                    session_id=row[10],
                    risk_score=row[11],
                    compliance_relevant=row[12]
                )
                events.append(event)
                
            conn.close()
            return events
            
        except Exception as e:
            logger.error(f"Audit event search failed: {e}")
            return []
            
    async def get_audit_statistics(self, time_range: Tuple[float, float]) -> Dict[str, Any]:
        """Get audit trail statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            start_time, end_time = time_range
            
            # Total events
            cursor.execute("""
                SELECT COUNT(*) FROM audit_events 
                WHERE timestamp BETWEEN ? AND ?
            """, (start_time, end_time))
            total_events = cursor.fetchone()[0]
            
            # Events by type
            cursor.execute("""
                SELECT event_type, COUNT(*) FROM audit_events 
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY event_type
            """, (start_time, end_time))
            events_by_type = dict(cursor.fetchall())
            
            # High risk events
            cursor.execute("""
                SELECT COUNT(*) FROM audit_events 
                WHERE timestamp BETWEEN ? AND ? AND risk_score >= 0.7
            """, (start_time, end_time))
            high_risk_events = cursor.fetchone()[0]
            
            # Unique users
            cursor.execute("""
                SELECT COUNT(DISTINCT user_id) FROM audit_events 
                WHERE timestamp BETWEEN ? AND ?
            """, (start_time, end_time))
            unique_users = cursor.fetchone()[0]
            
            # Compliance relevant events
            cursor.execute("""
                SELECT COUNT(*) FROM audit_events 
                WHERE timestamp BETWEEN ? AND ? AND compliance_relevant = 1
            """, (start_time, end_time))
            compliance_events = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_events': total_events,
                'events_by_type': events_by_type,
                'high_risk_events': high_risk_events,
                'unique_users': unique_users,
                'compliance_events': compliance_events,
                'time_range': time_range
            }
            
        except Exception as e:
            logger.error(f"Failed to get audit statistics: {e}")
            return {}

class ComplianceAssessor:
    """Automated compliance assessment engine"""
    
    def __init__(self):
        self.assessments: Dict[str, ComplianceAssessment] = {}
        self.automated_checks: Dict[str, Callable] = {}
        
    async def register_check(self, check_name: str, check_function: Callable):
        """Register automated compliance check"""
        self.automated_checks[check_name] = check_function
        
    async def assess_policy(self, policy: CompliancePolicy, 
                          audit_trail: AuditTrail) -> ComplianceAssessment:
        """Perform compliance assessment for policy"""
        try:
            assessment_id = str(uuid.uuid4())
            findings = []
            evidence = []
            recommendations = []
            
            # Run automated checks
            for check_config in policy.automated_checks:
                check_name = check_config.get('name')
                if check_name in self.automated_checks:
                    try:
                        result = await self.automated_checks[check_name](policy, audit_trail)
                        if result:
                            findings.extend(result.get('findings', []))
                            evidence.extend(result.get('evidence', []))
                            recommendations.extend(result.get('recommendations', []))
                    except Exception as e:
                        logger.error(f"Automated check {check_name} failed: {e}")
                        
            # Calculate compliance score
            total_checks = len(policy.automated_checks) + len(policy.manual_checks)
            passed_checks = len([f for f in findings if f.get('status') == 'passed'])
            score = (passed_checks / total_checks * 100) if total_checks > 0 else 0
            
            # Determine status
            if score >= 95:
                status = ComplianceStatus.COMPLIANT
            elif score >= 70:
                status = ComplianceStatus.PARTIAL
            else:
                status = ComplianceStatus.NON_COMPLIANT
                
            assessment = ComplianceAssessment(
                assessment_id=assessment_id,
                policy_id=policy.policy_id,
                standard=policy.standard,
                status=status,
                score=score,
                findings=findings,
                recommendations=recommendations,
                evidence=evidence,
                assessor="automated_system",
                next_assessment=time.time() + (policy.review_frequency * 86400)
            )
            
            self.assessments[assessment_id] = assessment
            logger.info(f"Completed assessment for policy {policy.name}: {score:.1f}% compliant")
            
            return assessment
            
        except Exception as e:
            logger.error(f"Policy assessment failed: {e}")
            return ComplianceAssessment(
                assessment_id=str(uuid.uuid4()),
                policy_id=policy.policy_id,
                standard=policy.standard,
                status=ComplianceStatus.UNKNOWN,
                score=0.0,
                findings=[],
                recommendations=[],
                evidence=[],
                assessor="automated_system"
            )
            
    async def _default_access_control_check(self, policy: CompliancePolicy, 
                                          audit_trail: AuditTrail) -> Dict[str, Any]:
        """Default access control compliance check"""
        try:
            # Check for unauthorized access attempts
            end_time = time.time()
            start_time = end_time - 86400  # Last 24 hours
            
            failed_logins = await audit_trail.search_events({
                'event_type': AuditEventType.LOGIN.value,
                'start_time': start_time,
                'end_time': end_time,
                'action': 'failed'
            })
            
            findings = []
            evidence = []
            recommendations = []
            
            if len(failed_logins) > 100:  # Threshold for concern
                findings.append({
                    'check': 'access_control',
                    'status': 'failed',
                    'message': f'High number of failed login attempts: {len(failed_logins)}'
                })
                recommendations.append('Implement account lockout policies')
                recommendations.append('Review authentication mechanisms')
            else:
                findings.append({
                    'check': 'access_control',
                    'status': 'passed',
                    'message': 'Normal login failure rates observed'
                })
                
            evidence.append({
                'type': 'audit_events',
                'count': len(failed_logins),
                'period': '24_hours'
            })
            
            return {
                'findings': findings,
                'evidence': evidence,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Access control check failed: {e}")
            return {'findings': [], 'evidence': [], 'recommendations': []}

class ReportGenerator:
    """Compliance report generation"""
    
    def __init__(self):
        self.reports: Dict[str, ComplianceReport] = {}
        
    async def generate_compliance_report(self, standards: List[ComplianceStandard],
                                       assessments: List[ComplianceAssessment],
                                       audit_stats: Dict[str, Any],
                                       format: ReportFormat = ReportFormat.JSON) -> ComplianceReport:
        """Generate comprehensive compliance report"""
        try:
            report_id = str(uuid.uuid4())
            
            # Organize assessments by standard
            assessments_by_standard = defaultdict(list)
            for assessment in assessments:
                assessments_by_standard[assessment.standard].append(assessment)
                
            # Calculate overall compliance score
            total_score = sum(a.score for a in assessments)
            overall_score = total_score / len(assessments) if assessments else 0
            
            # Determine overall status
            if overall_score >= 95:
                overall_status = ComplianceStatus.COMPLIANT
            elif overall_score >= 70:
                overall_status = ComplianceStatus.PARTIAL
            else:
                overall_status = ComplianceStatus.NON_COMPLIANT
                
            # Build report content
            content = {
                'executive_summary': {
                    'overall_score': overall_score,
                    'overall_status': overall_status.value,
                    'total_standards': len(standards),
                    'total_assessments': len(assessments),
                    'report_period': audit_stats.get('time_range', [])
                },
                'standards_compliance': {},
                'key_findings': [],
                'recommendations': [],
                'audit_summary': audit_stats,
                'detailed_assessments': []
            }
            
            # Process each standard
            for standard in standards:
                standard_assessments = assessments_by_standard.get(standard, [])
                
                if standard_assessments:
                    standard_score = sum(a.score for a in standard_assessments) / len(standard_assessments)
                    standard_findings = []
                    standard_recommendations = set()
                    
                    for assessment in standard_assessments:
                        standard_findings.extend(assessment.findings)
                        standard_recommendations.update(assessment.recommendations)
                        
                        content['detailed_assessments'].append({
                            'assessment_id': assessment.assessment_id,
                            'policy_id': assessment.policy_id,
                            'standard': assessment.standard.value,
                            'score': assessment.score,
                            'status': assessment.status.value,
                            'findings_count': len(assessment.findings),
                            'recommendations_count': len(assessment.recommendations),
                            'timestamp': assessment.timestamp
                        })
                        
                    content['standards_compliance'][standard.value] = {
                        'score': standard_score,
                        'status': ComplianceStatus.COMPLIANT.value if standard_score >= 95 else 
                                 ComplianceStatus.PARTIAL.value if standard_score >= 70 else 
                                 ComplianceStatus.NON_COMPLIANT.value,
                        'assessments_count': len(standard_assessments),
                        'findings_count': len(standard_findings),
                        'recommendations_count': len(standard_recommendations)
                    }
                    
                    # Add top findings and recommendations
                    failed_findings = [f for f in standard_findings if f.get('status') == 'failed']
                    content['key_findings'].extend(failed_findings[:5])  # Top 5
                    content['recommendations'].extend(list(standard_recommendations)[:5])  # Top 5
                    
            # Create report object
            report = ComplianceReport(
                report_id=report_id,
                name=f"Compliance Report - {', '.join([s.value.upper() for s in standards])}",
                description=f"Automated compliance assessment report for {len(standards)} standards",
                standards=standards,
                time_range=(audit_stats.get('time_range', [0, time.time()])),
                generated_by="automated_system",
                format=format,
                content=content,
                expires_at=time.time() + (90 * 86400)  # 90 days
            )
            
            self.reports[report_id] = report
            logger.info(f"Generated compliance report {report_id}")
            
            return report
            
        except Exception as e:
            logger.error(f"Compliance report generation failed: {e}")
            return ComplianceReport(
                report_id=str(uuid.uuid4()),
                name="Error Report",
                description="Report generation failed",
                standards=standards,
                time_range=(0, time.time()),
                generated_by="automated_system",
                format=format,
                content={}
            )
            
    async def export_report(self, report_id: str, file_path: str) -> bool:
        """Export report to file"""
        try:
            if report_id not in self.reports:
                return False
                
            report = self.reports[report_id]
            
            if report.format == ReportFormat.JSON:
                with open(file_path, 'w') as f:
                    json.dump(report.content, f, indent=2, default=str)
            elif report.format == ReportFormat.CSV:
                # Convert to CSV format (simplified)
                import csv
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Metric', 'Value'])
                    
                    # Write summary data
                    summary = report.content.get('executive_summary', {})
                    for key, value in summary.items():
                        writer.writerow([key, value])
                        
            # Update report with file path
            report.file_path = file_path
            
            logger.info(f"Exported report {report_id} to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Report export failed: {e}")
            return False

class ComplianceFramework:
    """Main compliance framework engine"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.db_path = self.config.get('db_path', 'backend/compliance.db')
        
        # Core components
        self.policy_manager = PolicyManager()
        self.audit_trail = AuditTrail(self.db_path)
        self.assessor = ComplianceAssessor()
        self.report_generator = ReportGenerator()
        
        # Monitoring
        self.running = False
        
        # Metrics
        self.metrics = {
            'total_policies': 0,
            'active_assessments': 0,
            'compliance_score': 0.0,
            'audit_events': 0,
            'generated_reports': 0
        }
        
    async def initialize(self) -> bool:
        """Initialize the compliance framework"""
        try:
            logger.info("Initializing Compliance Framework...")
            
            # Setup database
            await self._setup_database()
            
            # Initialize components
            await self.policy_manager.initialize()
            
            # Register default checks
            await self.assessor.register_check('access_control', 
                                             self.assessor._default_access_control_check)
            
            # Start background tasks
            self.running = True
            asyncio.create_task(self._compliance_monitor())
            asyncio.create_task(self._audit_processor())
            
            logger.info("Compliance Framework initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Compliance framework initialization failed: {e}")
            return False
            
    async def shutdown(self):
        """Shutdown the compliance framework"""
        try:
            logger.info("Shutting down Compliance Framework...")
            self.running = False
            
            # Flush any pending audit events
            await self.audit_trail._flush_buffer()
            
            logger.info("Compliance Framework shutdown complete")
            
        except Exception as e:
            logger.error(f"Compliance framework shutdown error: {e}")
            
    async def create_policy(self, standard: ComplianceStandard, template_name: str, 
                          owner: str) -> Optional[str]:
        """Create compliance policy from template"""
        try:
            policy_id = await self.policy_manager.create_policy_from_template(
                standard, template_name, owner
            )
            
            if policy_id:
                self.metrics['total_policies'] += 1
                
            return policy_id
            
        except Exception as e:
            logger.error(f"Failed to create policy: {e}")
            return None
            
    async def log_audit_event(self, event: AuditEvent) -> bool:
        """Log compliance audit event"""
        try:
            success = await self.audit_trail.log_event(event)
            if success:
                self.metrics['audit_events'] += 1
                
            return success
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            return False
            
    async def run_assessment(self, policy_id: str) -> Optional[ComplianceAssessment]:
        """Run compliance assessment for policy"""
        try:
            policy = self.policy_manager.policies.get(policy_id)
            if not policy:
                return None
                
            assessment = await self.assessor.assess_policy(policy, self.audit_trail)
            self.metrics['active_assessments'] += 1
            
            return assessment
            
        except Exception as e:
            logger.error(f"Assessment failed: {e}")
            return None
            
    async def generate_report(self, standards: List[ComplianceStandard],
                            time_range: Optional[Tuple[float, float]] = None) -> Optional[ComplianceReport]:
        """Generate compliance report"""
        try:
            if time_range is None:
                end_time = time.time()
                start_time = end_time - (30 * 86400)  # Last 30 days
                time_range = (start_time, end_time)
                
            # Get assessments for standards
            assessments = []
            for standard in standards:
                policies = await self.policy_manager.get_policies_by_standard(standard)
                for policy in policies:
                    assessment = await self.run_assessment(policy.policy_id)
                    if assessment:
                        assessments.append(assessment)
                        
            # Get audit statistics
            audit_stats = await self.audit_trail.get_audit_statistics(time_range)
            
            # Generate report
            report = await self.report_generator.generate_compliance_report(
                standards, assessments, audit_stats
            )
            
            self.metrics['generated_reports'] += 1
            
            # Update compliance score
            if assessments:
                self.metrics['compliance_score'] = sum(a.score for a in assessments) / len(assessments)
                
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return None
            
    async def get_compliance_status(self) -> Dict[str, Any]:
        """Get comprehensive compliance status"""
        try:
            # Get overdue policy reviews
            overdue_reviews = await self.policy_manager.get_overdue_reviews()
            
            # Get recent audit statistics
            end_time = time.time()
            start_time = end_time - 86400  # Last 24 hours
            audit_stats = await self.audit_trail.get_audit_statistics((start_time, end_time))
            
            # Policy statistics
            policies_by_standard = defaultdict(int)
            for policy in self.policy_manager.policies.values():
                policies_by_standard[policy.standard.value] += 1
                
            return {
                'framework_status': {
                    'running': self.running,
                    'total_policies': len(self.policy_manager.policies),
                    'total_assessments': len(self.assessor.assessments),
                    'total_reports': len(self.report_generator.reports)
                },
                'compliance_metrics': {
                    'overall_score': self.metrics['compliance_score'],
                    'audit_events_24h': audit_stats.get('total_events', 0),
                    'high_risk_events_24h': audit_stats.get('high_risk_events', 0),
                    'overdue_reviews': len(overdue_reviews)
                },
                'policy_distribution': dict(policies_by_standard),
                'audit_summary': audit_stats,
                'system_metrics': self.metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to get compliance status: {e}")
            return {}
            
    async def _compliance_monitor(self):
        """Background compliance monitoring"""
        while self.running:
            try:
                # Check for overdue assessments
                current_time = time.time()
                
                for policy in self.policy_manager.policies.values():
                    if not policy.enabled:
                        continue
                        
                    # Check if assessment is due
                    if (policy.last_review is None or 
                        current_time - policy.last_review > (policy.review_frequency * 86400)):
                        
                        logger.info(f"Running scheduled assessment for policy: {policy.name}")
                        await self.run_assessment(policy.policy_id)
                        policy.last_review = current_time
                        
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Compliance monitoring error: {e}")
                await asyncio.sleep(3600)
                
    async def _audit_processor(self):
        """Background audit processing"""
        while self.running:
            try:
                # Periodic buffer flush
                if len(self.audit_trail.event_buffer) > 0:
                    await self.audit_trail._flush_buffer()
                    
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Audit processing error: {e}")
                await asyncio.sleep(300)
                
    async def _setup_database(self):
        """Setup SQLite database for compliance data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS compliance_policies (
                policy_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                standard TEXT NOT NULL,
                requirements TEXT,
                controls TEXT,
                automated_checks TEXT,
                manual_checks TEXT,
                risk_level TEXT NOT NULL,
                owner TEXT NOT NULL,
                review_frequency INTEGER DEFAULT 90,
                enabled BOOLEAN DEFAULT TRUE,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                last_updated REAL DEFAULT (strftime('%s', 'now')),
                last_review REAL
            );
            
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                user_id TEXT NOT NULL,
                user_name TEXT NOT NULL,
                timestamp REAL NOT NULL,
                resource TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT,
                ip_address TEXT,
                user_agent TEXT,
                session_id TEXT,
                risk_score REAL DEFAULT 0.0,
                compliance_relevant BOOLEAN DEFAULT TRUE
            );
            
            CREATE TABLE IF NOT EXISTS compliance_assessments (
                assessment_id TEXT PRIMARY KEY,
                policy_id TEXT NOT NULL,
                standard TEXT NOT NULL,
                status TEXT NOT NULL,
                score REAL NOT NULL,
                findings TEXT,
                recommendations TEXT,
                evidence TEXT,
                assessor TEXT NOT NULL,
                timestamp REAL DEFAULT (strftime('%s', 'now')),
                next_assessment REAL
            );
            
            CREATE TABLE IF NOT EXISTS compliance_reports (
                report_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                standards TEXT NOT NULL,
                time_range TEXT NOT NULL,
                generated_by TEXT NOT NULL,
                format TEXT NOT NULL,
                content TEXT NOT NULL,
                file_path TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                expires_at REAL
            );
            
            CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit_events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_audit_events_user ON audit_events(user_id);
            CREATE INDEX IF NOT EXISTS idx_audit_events_type ON audit_events(event_type);
            CREATE INDEX IF NOT EXISTS idx_audit_events_risk ON audit_events(risk_score);
        """)
        
        conn.commit()
        conn.close()

# Global instance
_compliance_framework: Optional[ComplianceFramework] = None

async def initialize_compliance_framework(config: Optional[Dict[str, Any]] = None) -> bool:
    """Initialize the global compliance framework"""
    global _compliance_framework
    try:
        _compliance_framework = ComplianceFramework(config)
        return await _compliance_framework.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize compliance framework: {e}")
        return False

def get_compliance_framework() -> ComplianceFramework:
    """Get the global compliance framework instance"""
    global _compliance_framework
    if _compliance_framework is None:
        raise RuntimeError("Compliance framework not initialized. Call initialize_compliance_framework() first.")
    return _compliance_framework

async def shutdown_compliance_framework():
    """Shutdown the global compliance framework"""
    global _compliance_framework
    if _compliance_framework:
        await _compliance_framework.shutdown()
        _compliance_framework = None
