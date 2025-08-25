#!/usr/bin/env python3
"""
Configuration Validator Module for NeuralSync2
Advanced configuration validation, conflict detection, and auto-resolution
"""

import os
import yaml
import json
import hashlib
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import socket
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ConflictType(Enum):
    """Configuration conflict types"""
    PORT_CONFLICT = "port_conflict"
    PATH_CONFLICT = "path_conflict"
    DUPLICATE_SERVICE = "duplicate_service"
    INVALID_VALUE = "invalid_value"
    MISSING_DEPENDENCY = "missing_dependency"
    PERMISSION_ISSUE = "permission_issue"

@dataclass
class ValidationIssue:
    """Configuration validation issue"""
    severity: ValidationSeverity
    conflict_type: ConflictType
    component: str
    field: str
    current_value: Any
    expected_value: Optional[Any]
    description: str
    auto_fixable: bool
    fix_action: Optional[str]
    dependencies: List[str] = field(default_factory=list)

@dataclass
class ConfigurationProfile:
    """Configuration profile for different environments"""
    name: str
    description: str
    base_config: Dict[str, Any]
    overrides: Dict[str, Any] = field(default_factory=dict)
    required_services: List[str] = field(default_factory=list)
    optional_services: List[str] = field(default_factory=list)
    environment_vars: Dict[str, str] = field(default_factory=dict)

class ConfigurationValidator:
    """Advanced configuration validation and management"""
    
    def __init__(self, config_dir: Path):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration paths
        self.main_config_path = self.config_dir / "config.yaml"
        self.profiles_path = self.config_dir / "profiles.json"
        self.validation_cache_path = self.config_dir / "validation_cache.json"
        
        # Validation rules
        self.validation_rules = self._load_validation_rules()
        self.profiles = self._load_profiles()
        
        # Runtime state
        self.validation_cache: Dict[str, Any] = self._load_validation_cache()
        self.active_profile: Optional[str] = None
        
        # Locking
        self._validation_lock = threading.RLock()
        
        # Executor for concurrent validation
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="config_validator")
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load configuration validation rules"""
        return {
            'neuralsync_server': {
                'required_fields': ['site_id', 'bind_host', 'bind_port'],
                'field_types': {
                    'site_id': str,
                    'bind_host': str,
                    'bind_port': int,
                    'db_path': str,
                    'oplog_path': str,
                    'vector_dim': int,
                    'token': str
                },
                'field_constraints': {
                    'bind_port': {'min': 1024, 'max': 65535},
                    'vector_dim': {'min': 128, 'max': 2048},
                    'bind_host': {'pattern': r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$|^localhost$|^127\.0\.0\.1$'}
                },
                'dependencies': ['db_path', 'oplog_path']
            },
            'daemon_manager': {
                'required_fields': ['pid_dir', 'log_dir'],
                'field_types': {
                    'pid_dir': str,
                    'log_dir': str,
                    'startup_timeout': int,
                    'health_check_interval': int,
                    'max_restart_attempts': int
                },
                'field_constraints': {
                    'startup_timeout': {'min': 10, 'max': 300},
                    'health_check_interval': {'min': 5, 'max': 300},
                    'max_restart_attempts': {'min': 0, 'max': 10}
                },
                'dependencies': []
            },
            'cli_integration': {
                'required_fields': ['cli_tools'],
                'field_types': {
                    'personality_data': str,
                    'cli_tools': list
                },
                'field_constraints': {},
                'dependencies': []
            }
        }
    
    def _load_profiles(self) -> Dict[str, ConfigurationProfile]:
        """Load configuration profiles"""
        profiles = {}
        
        # Default profiles
        default_profiles = {
            'development': ConfigurationProfile(
                name='development',
                description='Development environment with verbose logging and fast restarts',
                base_config={
                    'bind_host': '127.0.0.1',
                    'bind_port': 8373,
                    'vector_dim': 512,
                    'startup_timeout': 30,
                    'health_check_interval': 10,
                    'max_restart_attempts': 5
                },
                required_services=['neuralsync-server'],
                optional_services=['neuralsync-broker'],
                environment_vars={'NS_LOG_LEVEL': 'DEBUG'}
            ),
            'production': ConfigurationProfile(
                name='production',
                description='Production environment with stability and performance optimizations',
                base_config={
                    'bind_host': '127.0.0.1',
                    'bind_port': 8373,
                    'vector_dim': 1024,
                    'startup_timeout': 60,
                    'health_check_interval': 30,
                    'max_restart_attempts': 3
                },
                required_services=['neuralsync-server', 'neuralsync-broker'],
                optional_services=[],
                environment_vars={'NS_LOG_LEVEL': 'INFO'}
            ),
            'testing': ConfigurationProfile(
                name='testing',
                description='Testing environment with isolated resources',
                base_config={
                    'bind_host': '127.0.0.1',
                    'bind_port': 8374,  # Different port
                    'vector_dim': 256,  # Smaller for faster tests
                    'startup_timeout': 15,
                    'health_check_interval': 5,
                    'max_restart_attempts': 1
                },
                required_services=['neuralsync-server'],
                optional_services=[],
                environment_vars={'NS_LOG_LEVEL': 'DEBUG', 'NS_TEST_MODE': '1'}
            )
        }
        
        # Load custom profiles if they exist
        if self.profiles_path.exists():
            try:
                with open(self.profiles_path, 'r') as f:
                    custom_profiles_data = json.load(f)
                
                for profile_name, profile_data in custom_profiles_data.items():
                    profiles[profile_name] = ConfigurationProfile(**profile_data)
                    
            except Exception as e:
                logger.warning(f"Failed to load custom profiles: {e}")
        
        # Merge with defaults
        for name, profile in default_profiles.items():
            if name not in profiles:
                profiles[name] = profile
        
        return profiles
    
    def _load_validation_cache(self) -> Dict[str, Any]:
        """Load validation cache"""
        if self.validation_cache_path.exists():
            try:
                with open(self.validation_cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.debug(f"Failed to load validation cache: {e}")
        
        return {
            'last_validation': 0,
            'config_hash': '',
            'issues': [],
            'profile_checksums': {}
        }
    
    def _save_validation_cache(self):
        """Save validation cache"""
        try:
            with open(self.validation_cache_path, 'w') as f:
                json.dump(self.validation_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save validation cache: {e}")
    
    def _calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """Calculate hash of configuration for change detection"""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def validate_configuration(
        self, 
        config: Dict[str, Any],
        profile_name: Optional[str] = None,
        force_revalidation: bool = False
    ) -> List[ValidationIssue]:
        """Comprehensive configuration validation"""
        
        with self._validation_lock:
            # Check if validation is needed
            config_hash = self._calculate_config_hash(config)
            
            if not force_revalidation and config_hash == self.validation_cache.get('config_hash', ''):
                # Return cached issues if configuration unchanged
                cached_issues = self.validation_cache.get('issues', [])
                return [ValidationIssue(**issue_data) for issue_data in cached_issues]
            
            issues = []
            
            # Apply profile if specified
            if profile_name and profile_name in self.profiles:
                config = self._apply_profile(config, profile_name)
            
            # Basic structure validation
            issues.extend(self._validate_structure(config))
            
            # Field type and constraint validation
            issues.extend(self._validate_fields(config))
            
            # Dependency validation
            issues.extend(self._validate_dependencies(config))
            
            # Resource availability validation
            issues.extend(self._validate_resources(config))
            
            # Port conflict detection
            issues.extend(self._validate_port_conflicts(config))
            
            # Path validation
            issues.extend(self._validate_paths(config))
            
            # Service configuration validation
            issues.extend(self._validate_services(config, profile_name))
            
            # Update cache
            # Convert issues to serializable format
            serializable_issues = []
            for issue in issues:
                issue_dict = asdict(issue)
                issue_dict['severity'] = issue.severity.value
                issue_dict['conflict_type'] = issue.conflict_type.value
                serializable_issues.append(issue_dict)
            
            self.validation_cache.update({
                'last_validation': time.time() if 'time' in globals() else 0,
                'config_hash': config_hash,
                'issues': serializable_issues,
                'profile_checksums': {
                    profile_name: self._calculate_config_hash(profile.base_config)
                    for profile_name, profile in self.profiles.items()
                }
            })
            
            self._save_validation_cache()
            
            return issues
    
    def _apply_profile(self, config: Dict[str, Any], profile_name: str) -> Dict[str, Any]:
        """Apply configuration profile"""
        if profile_name not in self.profiles:
            return config
        
        profile = self.profiles[profile_name]
        
        # Create merged configuration
        merged_config = config.copy()
        
        # Apply profile base config
        for key, value in profile.base_config.items():
            if key not in merged_config:
                merged_config[key] = value
        
        # Apply profile overrides
        for key, value in profile.overrides.items():
            merged_config[key] = value
        
        return merged_config
    
    def _validate_structure(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate basic configuration structure"""
        issues = []
        
        for component, rules in self.validation_rules.items():
            required_fields = rules.get('required_fields', [])
            
            for field in required_fields:
                if field not in config:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        conflict_type=ConflictType.MISSING_DEPENDENCY,
                        component=component,
                        field=field,
                        current_value=None,
                        expected_value="<required>",
                        description=f"Required field '{field}' is missing for component '{component}'",
                        auto_fixable=True,
                        fix_action=f"add_default_value:{field}"
                    ))
        
        return issues
    
    def _validate_fields(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate field types and constraints"""
        issues = []
        
        for component, rules in self.validation_rules.items():
            field_types = rules.get('field_types', {})
            field_constraints = rules.get('field_constraints', {})
            
            for field, expected_type in field_types.items():
                if field in config:
                    current_value = config[field]
                    
                    # Type validation
                    if not isinstance(current_value, expected_type):
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            conflict_type=ConflictType.INVALID_VALUE,
                            component=component,
                            field=field,
                            current_value=current_value,
                            expected_value=expected_type.__name__,
                            description=f"Field '{field}' should be of type {expected_type.__name__}, got {type(current_value).__name__}",
                            auto_fixable=True,
                            fix_action=f"convert_type:{field}:{expected_type.__name__}"
                        ))
                        continue
                    
                    # Constraint validation
                    if field in field_constraints:
                        constraints = field_constraints[field]
                        
                        # Numeric constraints
                        if isinstance(current_value, (int, float)):
                            if 'min' in constraints and current_value < constraints['min']:
                                issues.append(ValidationIssue(
                                    severity=ValidationSeverity.WARNING,
                                    conflict_type=ConflictType.INVALID_VALUE,
                                    component=component,
                                    field=field,
                                    current_value=current_value,
                                    expected_value=f">= {constraints['min']}",
                                    description=f"Field '{field}' value {current_value} is below minimum {constraints['min']}",
                                    auto_fixable=True,
                                    fix_action=f"set_value:{field}:{constraints['min']}"
                                ))
                            
                            if 'max' in constraints and current_value > constraints['max']:
                                issues.append(ValidationIssue(
                                    severity=ValidationSeverity.WARNING,
                                    conflict_type=ConflictType.INVALID_VALUE,
                                    component=component,
                                    field=field,
                                    current_value=current_value,
                                    expected_value=f"<= {constraints['max']}",
                                    description=f"Field '{field}' value {current_value} is above maximum {constraints['max']}",
                                    auto_fixable=True,
                                    fix_action=f"set_value:{field}:{constraints['max']}"
                                ))
                        
                        # Pattern validation for strings
                        if isinstance(current_value, str) and 'pattern' in constraints:
                            import re
                            pattern = constraints['pattern']
                            if not re.match(pattern, current_value):
                                issues.append(ValidationIssue(
                                    severity=ValidationSeverity.ERROR,
                                    conflict_type=ConflictType.INVALID_VALUE,
                                    component=component,
                                    field=field,
                                    current_value=current_value,
                                    expected_value=f"matching pattern: {pattern}",
                                    description=f"Field '{field}' value '{current_value}' does not match required pattern",
                                    auto_fixable=False,
                                    fix_action=None
                                ))
        
        return issues
    
    def _validate_dependencies(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate configuration dependencies"""
        issues = []
        
        for component, rules in self.validation_rules.items():
            dependencies = rules.get('dependencies', [])
            
            for dependency in dependencies:
                if dependency in config:
                    dep_value = config[dependency]
                    
                    # Path dependency validation
                    if isinstance(dep_value, str) and ('path' in dependency.lower() or 'dir' in dependency.lower()):
                        path = Path(dep_value)
                        
                        # Check if parent directory exists
                        if not path.parent.exists():
                            issues.append(ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                conflict_type=ConflictType.PATH_CONFLICT,
                                component=component,
                                field=dependency,
                                current_value=str(path),
                                expected_value="existing path",
                                description=f"Parent directory for '{dependency}' does not exist: {path.parent}",
                                auto_fixable=True,
                                fix_action=f"create_directory:{path.parent}"
                            ))
                        
                        # Check write permissions
                        if path.parent.exists():
                            try:
                                test_file = path.parent / f".test_write_{os.getpid()}"
                                test_file.touch()
                                test_file.unlink()
                            except (PermissionError, OSError):
                                issues.append(ValidationIssue(
                                    severity=ValidationSeverity.ERROR,
                                    conflict_type=ConflictType.PERMISSION_ISSUE,
                                    component=component,
                                    field=dependency,
                                    current_value=str(path),
                                    expected_value="writable path",
                                    description=f"No write permission for '{dependency}' path: {path.parent}",
                                    auto_fixable=False,
                                    fix_action=None
                                ))
        
        return issues
    
    def _validate_resources(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate resource availability"""
        issues = []
        
        # Memory validation
        import psutil
        available_memory_mb = psutil.virtual_memory().available / 1024 / 1024
        
        if available_memory_mb < 256:  # Minimum 256MB
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                conflict_type=ConflictType.INVALID_VALUE,
                component="system",
                field="available_memory",
                current_value=f"{available_memory_mb:.0f}MB",
                expected_value=">= 256MB",
                description=f"Low available memory: {available_memory_mb:.0f}MB. Consider reducing cache sizes.",
                auto_fixable=False,
                fix_action=None
            ))
        
        # Disk space validation
        if 'db_path' in config:
            db_path = Path(config['db_path'])
            if db_path.parent.exists():
                disk_usage = shutil.disk_usage(db_path.parent)
                free_space_mb = disk_usage.free / 1024 / 1024
                
                if free_space_mb < 100:  # Minimum 100MB
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        conflict_type=ConflictType.INVALID_VALUE,
                        component="system",
                        field="disk_space",
                        current_value=f"{free_space_mb:.0f}MB",
                        expected_value=">= 100MB",
                        description=f"Insufficient disk space for database: {free_space_mb:.0f}MB available",
                        auto_fixable=False,
                        fix_action=None
                    ))
        
        return issues
    
    def _validate_port_conflicts(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate port availability"""
        issues = []
        
        if 'bind_port' in config:
            port = config['bind_port']
            
            # Check if port is available
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.settimeout(1)
                result = sock.connect_ex(('127.0.0.1', port))
                
                if result == 0:  # Port is occupied
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        conflict_type=ConflictType.PORT_CONFLICT,
                        component="neuralsync_server",
                        field="bind_port",
                        current_value=port,
                        expected_value="available port",
                        description=f"Port {port} is already in use",
                        auto_fixable=True,
                        fix_action=f"find_alternative_port:{port}"
                    ))
            except Exception:
                pass
            finally:
                sock.close()
        
        return issues
    
    def _validate_paths(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate file and directory paths"""
        issues = []
        
        path_fields = ['db_path', 'oplog_path', 'pid_dir', 'log_dir']
        
        for field in path_fields:
            if field in config:
                path = Path(config[field])
                
                # Check if path is absolute (recommended for production)
                if not path.is_absolute():
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        conflict_type=ConflictType.INVALID_VALUE,
                        component="paths",
                        field=field,
                        current_value=str(path),
                        expected_value="absolute path",
                        description=f"Relative path detected for '{field}'. Absolute paths recommended for production.",
                        auto_fixable=True,
                        fix_action=f"make_absolute:{field}"
                    ))
                
                # Check for path traversal issues
                if '..' in str(path):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        conflict_type=ConflictType.INVALID_VALUE,
                        component="paths",
                        field=field,
                        current_value=str(path),
                        expected_value="clean path",
                        description=f"Path traversal detected in '{field}'. Consider using absolute paths.",
                        auto_fixable=True,
                        fix_action=f"clean_path:{field}"
                    ))
        
        return issues
    
    def _validate_services(self, config: Dict[str, Any], profile_name: Optional[str]) -> List[ValidationIssue]:
        """Validate service configuration"""
        issues = []
        
        if profile_name and profile_name in self.profiles:
            profile = self.profiles[profile_name]
            
            # Check required services
            for service in profile.required_services:
                if service not in config.get('services', {}):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        conflict_type=ConflictType.MISSING_DEPENDENCY,
                        component="services",
                        field=service,
                        current_value=None,
                        expected_value="configured service",
                        description=f"Required service '{service}' is not configured for profile '{profile_name}'",
                        auto_fixable=True,
                        fix_action=f"add_service:{service}"
                    ))
        
        return issues
    
    def auto_fix_issues(self, config: Dict[str, Any], issues: List[ValidationIssue]) -> Tuple[Dict[str, Any], List[str]]:
        """Automatically fix configuration issues where possible"""
        fixed_config = config.copy()
        fix_log = []
        
        for issue in issues:
            if not issue.auto_fixable or not issue.fix_action:
                continue
            
            try:
                action_parts = issue.fix_action.split(':', 2)
                action = action_parts[0]
                field = action_parts[1] if len(action_parts) > 1 else None
                value = action_parts[2] if len(action_parts) > 2 else None
                
                if action == 'add_default_value':
                    fixed_config[field] = self._get_default_value(issue.component, field)
                    fix_log.append(f"Added default value for {field}")
                
                elif action == 'convert_type':
                    if value == 'int':
                        fixed_config[field] = int(fixed_config.get(field, 0))
                    elif value == 'str':
                        fixed_config[field] = str(fixed_config.get(field, ''))
                    elif value == 'float':
                        fixed_config[field] = float(fixed_config.get(field, 0.0))
                    fix_log.append(f"Converted {field} to {value}")
                
                elif action == 'set_value':
                    fixed_config[field] = type(fixed_config.get(field))(value)
                    fix_log.append(f"Set {field} to {value}")
                
                elif action == 'find_alternative_port':
                    original_port = int(value)
                    alternative_port = self._find_alternative_port(original_port)
                    if alternative_port:
                        fixed_config[field] = alternative_port
                        fix_log.append(f"Changed port from {original_port} to {alternative_port}")
                
                elif action == 'create_directory':
                    Path(value).mkdir(parents=True, exist_ok=True)
                    fix_log.append(f"Created directory: {value}")
                
                elif action == 'make_absolute':
                    relative_path = fixed_config.get(field, '')
                    absolute_path = str(Path(relative_path).resolve())
                    fixed_config[field] = absolute_path
                    fix_log.append(f"Made {field} absolute: {absolute_path}")
                
                elif action == 'clean_path':
                    dirty_path = fixed_config.get(field, '')
                    clean_path = str(Path(dirty_path).resolve())
                    fixed_config[field] = clean_path
                    fix_log.append(f"Cleaned path {field}: {clean_path}")
                
            except Exception as e:
                logger.warning(f"Failed to auto-fix issue for {issue.field}: {e}")
        
        return fixed_config, fix_log
    
    def _get_default_value(self, component: str, field: str) -> Any:
        """Get default value for a configuration field"""
        defaults = {
            'neuralsync_server': {
                'site_id': lambda: __import__('uuid').uuid4().hex,
                'bind_host': '127.0.0.1',
                'bind_port': 8373,
                'db_path': str(self.config_dir / 'memory.db'),
                'oplog_path': str(self.config_dir / 'oplog.jsonl'),
                'vector_dim': 512,
                'token': ''
            },
            'daemon_manager': {
                'pid_dir': str(self.config_dir / 'pids'),
                'log_dir': str(self.config_dir / 'logs'),
                'startup_timeout': 30,
                'health_check_interval': 10,
                'max_restart_attempts': 5
            }
        }
        
        component_defaults = defaults.get(component, {})
        default_value = component_defaults.get(field)
        
        if callable(default_value):
            return default_value()
        return default_value
    
    def _find_alternative_port(self, original_port: int) -> Optional[int]:
        """Find an alternative available port"""
        for offset in range(1, 100):  # Try up to 100 alternatives
            for candidate_port in [original_port + offset, original_port - offset]:
                if 1024 <= candidate_port <= 65535:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    try:
                        sock.settimeout(1)
                        result = sock.connect_ex(('127.0.0.1', candidate_port))
                        if result != 0:  # Port is available
                            return candidate_port
                    except Exception:
                        pass
                    finally:
                        sock.close()
        
        return None
    
    def create_profile(self, profile: ConfigurationProfile) -> bool:
        """Create a new configuration profile"""
        try:
            self.profiles[profile.name] = profile
            
            # Save profiles to file
            profiles_data = {
                name: asdict(prof) for name, prof in self.profiles.items()
                if name not in ['development', 'production', 'testing']  # Don't save built-ins
            }
            
            with open(self.profiles_path, 'w') as f:
                json.dump(profiles_data, f, indent=2)
            
            logger.info(f"Created configuration profile: {profile.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create profile {profile.name}: {e}")
            return False
    
    def apply_profile_to_config(self, config: Dict[str, Any], profile_name: str) -> Dict[str, Any]:
        """Apply a configuration profile to existing config"""
        return self._apply_profile(config, profile_name)
    
    def get_validation_summary(self, issues: List[ValidationIssue]) -> Dict[str, Any]:
        """Get validation summary statistics"""
        severity_counts = {severity.value: 0 for severity in ValidationSeverity}
        conflict_type_counts = {ctype.value: 0 for ctype in ConflictType}
        auto_fixable_count = 0
        
        for issue in issues:
            severity_counts[issue.severity.value] += 1
            conflict_type_counts[issue.conflict_type.value] += 1
            if issue.auto_fixable:
                auto_fixable_count += 1
        
        return {
            'total_issues': len(issues),
            'severity_breakdown': severity_counts,
            'conflict_type_breakdown': conflict_type_counts,
            'auto_fixable_count': auto_fixable_count,
            'manual_fix_required': len(issues) - auto_fixable_count
        }
    
    def export_configuration(self, config: Dict[str, Any], output_path: Path, format: str = 'yaml') -> bool:
        """Export configuration to file"""
        try:
            if format.lower() == 'yaml':
                with open(output_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                with open(output_path, 'w') as f:
                    json.dump(config, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Configuration exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False
    
    def shutdown(self):
        """Shutdown configuration validator"""
        self.executor.shutdown(wait=True)

# Testing function
def test_configuration_validator():
    """Test configuration validator functionality"""
    validator = ConfigurationValidator(Path.home() / ".neuralsync")
    
    # Test configuration
    test_config = {
        'site_id': 'test-site',
        'bind_host': '127.0.0.1',
        'bind_port': 8373,
        'db_path': '/tmp/test.db',
        'oplog_path': '/tmp/test.jsonl',
        'vector_dim': 512
    }
    
    try:
        # Validate configuration
        print("Testing configuration validation...")
        issues = validator.validate_configuration(test_config, 'development')
        
        print(f"Found {len(issues)} issues:")
        for issue in issues[:5]:  # Show first 5
            print(f"  {issue.severity.value}: {issue.description}")
        
        # Test auto-fix
        print("\nTesting auto-fix...")
        fixed_config, fix_log = validator.auto_fix_issues(test_config, issues)
        
        print(f"Applied {len(fix_log)} fixes:")
        for fix in fix_log:
            print(f"  - {fix}")
        
        # Test validation summary
        summary = validator.get_validation_summary(issues)
        print(f"\nValidation summary: {summary}")
        
        # Test profile creation
        test_profile = ConfigurationProfile(
            name='custom_test',
            description='Custom test profile',
            base_config={'bind_port': 9999, 'vector_dim': 128}
        )
        
        success = validator.create_profile(test_profile)
        print(f"\nProfile creation: {'success' if success else 'failed'}")
        
    finally:
        validator.shutdown()

if __name__ == "__main__":
    import time
    test_configuration_validator()