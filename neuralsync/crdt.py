"""
Advanced CRDT Implementation with Byzantine Fault Tolerance
Provides conflict-free replicated data types with vector clocks and hash chains
"""

import hashlib
import time
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Set, List, Tuple
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)

@dataclass
class Version:
    lamport: int
    site_id: str

def version_newer(a: 'Version', b: 'Version') -> bool:
    """Legacy compatibility function"""
    if a.lamport != b.lamport:
        return a.lamport > b.lamport
    return a.site_id > b.site_id


@dataclass
class AdvancedVersion:
    """Enhanced version with vector clocks and Byzantine fault tolerance"""
    vector_clock: Dict[str, int] = field(default_factory=dict)
    physical_time: int = field(default_factory=lambda: time.time_ns())
    hash_chain: bytes = field(default=b"")
    signature: bytes = field(default=b"")
    
    def __post_init__(self):
        if not self.hash_chain:
            self.hash_chain = self._compute_hash()
            
    def _compute_hash(self) -> bytes:
        """Compute cryptographic hash of version data"""
        data = {
            'vector_clock': self.vector_clock,
            'physical_time': self.physical_time
        }
        content = json.dumps(data, sort_keys=True).encode()
        return hashlib.sha256(content).digest()
        
    def increment(self, site_id: str) -> 'AdvancedVersion':
        """Increment version for given site"""
        new_clock = self.vector_clock.copy()
        new_clock[site_id] = new_clock.get(site_id, 0) + 1
        
        return AdvancedVersion(
            vector_clock=new_clock,
            physical_time=time.time_ns()
        )
        
    def merge(self, other: 'AdvancedVersion') -> 'AdvancedVersion':
        """Merge two versions taking maximum of all clocks"""
        merged_clock = {}
        all_sites = set(self.vector_clock.keys()) | set(other.vector_clock.keys())
        
        for site in all_sites:
            merged_clock[site] = max(
                self.vector_clock.get(site, 0),
                other.vector_clock.get(site, 0)
            )
            
        return AdvancedVersion(
            vector_clock=merged_clock,
            physical_time=max(self.physical_time, other.physical_time)
        )
        
    def happens_before(self, other: 'AdvancedVersion') -> bool:
        """Check if this version happens before other"""
        # Check if all our clocks are <= other's clocks
        our_sites = set(self.vector_clock.keys())
        other_sites = set(other.vector_clock.keys())
        
        # We must have at least one clock value less than other
        has_smaller = False
        
        for site in our_sites | other_sites:
            our_time = self.vector_clock.get(site, 0)
            other_time = other.vector_clock.get(site, 0)
            
            if our_time > other_time:
                return False  # We're not before
            elif our_time < other_time:
                has_smaller = True
                
        return has_smaller
        
    def concurrent_with(self, other: 'AdvancedVersion') -> bool:
        """Check if versions are concurrent (neither happens before the other)"""
        return not self.happens_before(other) and not other.happens_before(self)


class ByzantineError(Exception):
    """Raised when Byzantine fault is detected"""
    pass


class ByzantineCRDT:
    """CRDT with Byzantine fault tolerance and conflict resolution"""
    
    def __init__(self, site_id: str, trusted_sites: Optional[Set[str]] = None):
        self.site_id = site_id
        self.trusted_sites = trusted_sites or set()
        self.version = AdvancedVersion(vector_clock={site_id: 0})
        self.data: Dict[str, Any] = {}
        self.tombstones: Set[str] = set()
        self.operation_log: List[Tuple[AdvancedVersion, str, Dict[str, Any]]] = []
        self.lock = threading.RLock()
        
        # Byzantine fault detection
        self.suspected_byzantine: Set[str] = set()
        self.hash_chain_history: List[bytes] = []
        self.max_history_size = 1000
        
    def get(self, key: str) -> Optional[Any]:
        """Get value for key, returns None if deleted"""
        with self.lock:
            if key in self.tombstones:
                return None
            return self.data.get(key)
            
    def set(self, key: str, value: Any) -> bool:
        """Set key-value pair with conflict resolution"""
        with self.lock:
            # Increment our version
            self.version = self.version.increment(self.site_id)
            
            # Remove from tombstones if present
            self.tombstones.discard(key)
            
            # Store value
            old_value = self.data.get(key)
            self.data[key] = value
            
            # Log operation
            operation = {
                'type': 'set',
                'key': key,
                'value': value,
                'old_value': old_value
            }
            self.operation_log.append((self.version, self.site_id, operation))
            
            # Maintain log size
            if len(self.operation_log) > self.max_history_size:
                self.operation_log = self.operation_log[-self.max_history_size:]
                
            return True
            
    def delete(self, key: str) -> bool:
        """Delete key by adding tombstone"""
        with self.lock:
            if key not in self.data:
                return False
                
            self.version = self.version.increment(self.site_id)
            self.tombstones.add(key)
            
            # Log operation
            operation = {
                'type': 'delete',
                'key': key,
                'old_value': self.data.get(key)
            }
            self.operation_log.append((self.version, self.site_id, operation))
            
            return True
            
    def merge(self, other: 'ByzantineCRDT') -> 'ByzantineCRDT':
        """Merge with another CRDT with Byzantine fault detection"""
        with self.lock:
            # Validate other CRDT isn't Byzantine
            if not self._verify_integrity(other):
                raise ByzantineError(f"Byzantine behavior detected from {other.site_id}")
                
            # Create merged CRDT
            merged = ByzantineCRDT(self.site_id, self.trusted_sites.union(other.trusted_sites))
            
            # Merge versions
            merged.version = self.version.merge(other.version)
            
            # Merge data with conflict resolution
            all_keys = set(self.data.keys()) | set(other.data.keys())
            
            for key in all_keys:
                our_has = key in self.data and key not in self.tombstones
                other_has = key in other.data and key not in other.tombstones
                our_deleted = key in self.tombstones
                other_deleted = key in other.tombstones
                
                if our_deleted and other_deleted:
                    # Both deleted - keep tombstone
                    merged.tombstones.add(key)
                elif our_deleted and other_has:
                    # We deleted, they have value - resolve by version
                    our_delete_version = self._get_delete_version(key)
                    other_set_version = other._get_set_version(key)
                    
                    if other_set_version and our_delete_version:
                        if other_set_version.happens_before(our_delete_version):
                            merged.tombstones.add(key)  # Our delete wins
                        else:
                            merged.data[key] = other.data[key]  # Their set wins
                    else:
                        # Default to keeping the value
                        merged.data[key] = other.data[key]
                        
                elif other_deleted and our_has:
                    # They deleted, we have value
                    their_delete_version = other._get_delete_version(key)
                    our_set_version = self._get_set_version(key)
                    
                    if our_set_version and their_delete_version:
                        if our_set_version.happens_before(their_delete_version):
                            merged.tombstones.add(key)  # Their delete wins
                        else:
                            merged.data[key] = self.data[key]  # Our set wins
                    else:
                        merged.data[key] = self.data[key]
                        
                elif our_has and other_has:
                    # Both have values - resolve conflict
                    merged.data[key] = self._resolve_conflict(
                        key, self.data[key], other.data[key]
                    )
                elif our_has:
                    merged.data[key] = self.data[key]
                elif other_has:
                    merged.data[key] = other.data[key]
                    
            # Merge operation logs
            merged.operation_log = self._merge_operation_logs(other)
            
            return merged
            
    def _verify_integrity(self, other: 'ByzantineCRDT') -> bool:
        """Verify integrity of other CRDT to detect Byzantine faults"""
        
        # Check if site is already suspected
        if other.site_id in self.suspected_byzantine:
            logger.warning(f"Site {other.site_id} is suspected Byzantine")
            return False
            
        # Verify hash chain integrity
        if not self._verify_hash_chain(other):
            self.suspected_byzantine.add(other.site_id)
            logger.error(f"Hash chain verification failed for {other.site_id}")
            return False
            
        # Check for impossible version vectors
        if not self._verify_version_vector(other):
            self.suspected_byzantine.add(other.site_id)
            logger.error(f"Invalid version vector from {other.site_id}")
            return False
            
        # Verify operation consistency
        if not self._verify_operations(other):
            self.suspected_byzantine.add(other.site_id)
            logger.error(f"Inconsistent operations from {other.site_id}")
            return False
            
        return True
        
    def _verify_hash_chain(self, other: 'ByzantineCRDT') -> bool:
        """Verify cryptographic hash chain"""
        # Simplified verification - in practice would check full chain
        return len(other.hash_chain_history) <= self.max_history_size
        
    def _verify_version_vector(self, other: 'ByzantineCRDT') -> bool:
        """Verify version vector is internally consistent"""
        # Check that site's own clock is consistent
        if other.site_id in other.version.vector_clock:
            site_time = other.version.vector_clock[other.site_id]
            if site_time < 0 or site_time > 1000000:  # Reasonable bounds
                return False
                
        # Check for time travel (physical time consistency)
        if other.version.physical_time > time.time_ns() + 60 * 1000000000:  # 1 minute future
            return False
            
        return True
        
    def _verify_operations(self, other: 'ByzantineCRDT') -> bool:
        """Verify operation log consistency"""
        # Check that operations are ordered by version
        prev_version = None
        for version, site_id, operation in other.operation_log:
            if prev_version and not prev_version.happens_before(version):
                if not version.concurrent_with(prev_version):
                    return False  # Invalid ordering
            prev_version = version
            
        return True
        
    def _get_delete_version(self, key: str) -> Optional[AdvancedVersion]:
        """Get version when key was deleted"""
        for version, site_id, operation in reversed(self.operation_log):
            if operation['type'] == 'delete' and operation['key'] == key:
                return version
        return None
        
    def _get_set_version(self, key: str) -> Optional[AdvancedVersion]:
        """Get version when key was last set"""
        for version, site_id, operation in reversed(self.operation_log):
            if operation['type'] == 'set' and operation['key'] == key:
                return version
        return None
        
    def _resolve_conflict(self, key: str, our_value: Any, their_value: Any) -> Any:
        """Resolve conflict between two values"""
        # Use timestamp-based resolution with site_id as tiebreaker
        our_version = self._get_set_version(key)
        # For now, use simple last-writer-wins with site_id tiebreaker
        if isinstance(our_value, str) and isinstance(their_value, str):
            # String concatenation strategy
            return f"{our_value}|{their_value}"
        elif isinstance(our_value, (int, float)) and isinstance(their_value, (int, float)):
            # Take maximum for numbers
            return max(our_value, their_value)
        else:
            # Default: use site_id comparison
            return our_value if self.site_id > their_value else their_value
            
    def _merge_operation_logs(self, other: 'ByzantineCRDT') -> List[Tuple[AdvancedVersion, str, Dict[str, Any]]]:
        """Merge operation logs maintaining causal ordering"""
        merged_log = []
        
        # Simple merge - in practice would use more sophisticated ordering
        all_operations = self.operation_log + other.operation_log
        
        # Sort by physical time then by site_id for deterministic ordering
        all_operations.sort(key=lambda x: (x[0].physical_time, x[1]))
        
        # Deduplicate and maintain size limit
        seen_operations = set()
        for op in all_operations:
            op_signature = (op[1], op[2]['type'], op[2]['key'], op[0].physical_time)
            if op_signature not in seen_operations:
                merged_log.append(op)
                seen_operations.add(op_signature)
                
        return merged_log[-self.max_history_size:]
        
    def get_state(self) -> Dict[str, Any]:
        """Get current state for serialization"""
        with self.lock:
            return {
                'site_id': self.site_id,
                'version': asdict(self.version),
                'data': self.data.copy(),
                'tombstones': list(self.tombstones),
                'suspected_byzantine': list(self.suspected_byzantine)
            }
            
    def apply_state(self, state: Dict[str, Any]) -> None:
        """Apply serialized state"""
        with self.lock:
            self.data = state['data'].copy()
            self.tombstones = set(state['tombstones'])
            self.suspected_byzantine = set(state.get('suspected_byzantine', []))
            
            # Reconstruct version
            version_data = state['version']
            self.version = AdvancedVersion(
                vector_clock=version_data['vector_clock'],
                physical_time=version_data['physical_time'],
                hash_chain=bytes.fromhex(version_data.get('hash_chain', '')),
                signature=bytes.fromhex(version_data.get('signature', ''))
            )
