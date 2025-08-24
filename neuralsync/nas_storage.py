#!/usr/bin/env python3
"""
NAS Storage Integration for NeuralSync2
Cold storage system for long-term memory archival
"""

import asyncio
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import hashlib
import tarfile
import zipfile
import aiofiles
import aiohttp
from urllib.parse import urlparse

@dataclass
class StorageLocation:
    """NAS storage location configuration"""
    name: str
    type: str  # 'nfs', 'smb', 'ftp', 'local'
    host: str
    path: str
    username: Optional[str] = None
    password: Optional[str] = None
    mount_point: Optional[str] = None
    enabled: bool = True

class NASStorageManager:
    """High-performance NAS storage manager for cold archival"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or str(Path.home() / ".neuralsync" / "nas_config.json")
        self.storage_locations: Dict[str, StorageLocation] = {}
        self.local_cache_dir = Path.home() / ".neuralsync" / "nas_cache"
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        self.archive_threshold_days = 30  # Archive memories older than 30 days
        self.compression_enabled = True
        
        self._load_config()
    
    def _load_config(self):
        """Load NAS storage configuration"""
        if Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    
                for loc_data in config.get('storage_locations', []):
                    location = StorageLocation(**loc_data)
                    self.storage_locations[location.name] = location
                    
                self.archive_threshold_days = config.get('archive_threshold_days', 30)
                self.compression_enabled = config.get('compression_enabled', True)
                    
            except Exception as e:
                print(f"❌ Error loading NAS config: {e}")
        else:
            # Create default config
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default NAS configuration"""
        default_config = {
            'storage_locations': [
                {
                    'name': 'local_backup',
                    'type': 'local',
                    'host': 'localhost',
                    'path': str(Path.home() / 'neuralsync_backup'),
                    'enabled': True
                }
            ],
            'archive_threshold_days': 30,
            'compression_enabled': True
        }
        
        # Ensure parent directory exists
        Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
            
        self._load_config()
    
    async def add_storage_location(self, location: StorageLocation) -> bool:
        """Add new storage location"""
        try:
            # Test connection
            if await self._test_connection(location):
                self.storage_locations[location.name] = location
                await self._save_config()
                print(f"✅ Added NAS storage location: {location.name}")
                return True
            else:
                print(f"❌ Failed to connect to storage location: {location.name}")
                return False
        except Exception as e:
            print(f"❌ Error adding storage location: {e}")
            return False
    
    async def _test_connection(self, location: StorageLocation) -> bool:
        """Test connection to storage location"""
        if location.type == 'local':
            try:
                Path(location.path).mkdir(parents=True, exist_ok=True)
                test_file = Path(location.path) / '.neuralsync_test'
                test_file.write_text('test')
                test_file.unlink()
                return True
            except Exception:
                return False
        elif location.type == 'nfs':
            # Test NFS mount
            return await self._test_nfs_connection(location)
        elif location.type == 'smb':
            # Test SMB connection
            return await self._test_smb_connection(location)
        else:
            print(f"⚠️ Unsupported storage type: {location.type}")
            return False
    
    async def _test_nfs_connection(self, location: StorageLocation) -> bool:
        """Test NFS connection"""
        try:
            # This is a simplified test - in production you'd use proper NFS libraries
            mount_cmd = f"mount -t nfs {location.host}:{location.path} {location.mount_point}"
            result = await asyncio.create_subprocess_shell(
                f"timeout 10 {mount_cmd}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, _ = await result.communicate()
            return result.returncode == 0
        except Exception:
            return False
    
    async def _test_smb_connection(self, location: StorageLocation) -> bool:
        """Test SMB connection"""
        try:
            # This would use smbclient or similar in production
            return True  # Simplified for demo
        except Exception:
            return False
    
    async def archive_memories(self, memory_db_path: str, cutoff_days: int = None) -> Dict[str, Any]:
        """Archive old memories to cold storage"""
        cutoff_days = cutoff_days or self.archive_threshold_days
        cutoff_time = time.time() - (cutoff_days * 24 * 3600)
        
        archive_stats = {
            'archived_entries': 0,
            'total_size_mb': 0,
            'storage_locations': [],
            'errors': []
        }
        
        try:
            # Export old memories
            export_path = self.local_cache_dir / f"archive_{int(time.time())}.json"
            
            from .storage import connect
            con = connect(memory_db_path)
            
            # Get old entries
            old_entries = []
            for row in con.execute('SELECT * FROM items WHERE created_at < ?', (cutoff_time * 1000,)):
                # Convert row to dict
                cols = [desc[0] for desc in con.description]
                entry = dict(zip(cols, row))
                old_entries.append(entry)
            
            con.close()
            
            if not old_entries:
                print("📁 No memories to archive")
                return archive_stats
            
            # Save to export file
            with open(export_path, 'w') as f:
                json.dump({
                    'metadata': {
                        'export_time': time.time(),
                        'cutoff_time': cutoff_time,
                        'total_entries': len(old_entries)
                    },
                    'entries': old_entries
                }, f, indent=2, default=str)
            
            # Compress if enabled
            if self.compression_enabled:
                compressed_path = export_path.with_suffix('.tar.gz')
                with tarfile.open(compressed_path, 'w:gz') as tar:
                    tar.add(export_path, arcname=export_path.name)
                export_path.unlink()  # Remove uncompressed version
                export_path = compressed_path
            
            file_size_mb = export_path.stat().st_size / (1024 * 1024)
            archive_stats['total_size_mb'] = file_size_mb
            archive_stats['archived_entries'] = len(old_entries)
            
            # Upload to storage locations
            for location in self.storage_locations.values():
                if location.enabled:
                    success = await self._upload_archive(export_path, location)
                    if success:
                        archive_stats['storage_locations'].append(location.name)
                    else:
                        archive_stats['errors'].append(f"Failed to upload to {location.name}")
            
            # Clean up local files after successful upload
            if archive_stats['storage_locations']:
                # Remove archived entries from main database
                con = connect(memory_db_path)
                con.execute('DELETE FROM items WHERE created_at < ?', (cutoff_time * 1000,))
                con.commit()
                con.close()
                
                export_path.unlink()  # Remove local archive
                print(f"🗄️ Archived {len(old_entries)} memories ({file_size_mb:.2f} MB)")
            
        except Exception as e:
            archive_stats['errors'].append(str(e))
            print(f"❌ Archive error: {e}")
        
        return archive_stats
    
    async def _upload_archive(self, archive_path: Path, location: StorageLocation) -> bool:
        """Upload archive to storage location"""
        try:
            if location.type == 'local':
                dest_dir = Path(location.path)
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / archive_path.name
                shutil.copy2(archive_path, dest_path)
                return True
            elif location.type == 'nfs':
                # Copy to NFS mount point
                if location.mount_point and Path(location.mount_point).exists():
                    dest_path = Path(location.mount_point) / archive_path.name
                    shutil.copy2(archive_path, dest_path)
                    return True
            # Add other storage types as needed
            return False
        except Exception as e:
            print(f"❌ Upload error to {location.name}: {e}")
            return False
    
    async def restore_memories(self, archive_name: str, memory_db_path: str) -> Dict[str, Any]:
        """Restore memories from cold storage"""
        restore_stats = {
            'restored_entries': 0,
            'errors': [],
            'source_location': None
        }
        
        try:
            # Find archive in storage locations
            archive_path = None
            source_location = None
            
            for location in self.storage_locations.values():
                if location.enabled:
                    potential_path = await self._find_archive(archive_name, location)
                    if potential_path:
                        archive_path = potential_path
                        source_location = location
                        break
            
            if not archive_path:
                restore_stats['errors'].append(f"Archive {archive_name} not found in any storage location")
                return restore_stats
            
            restore_stats['source_location'] = source_location.name
            
            # Download archive to local cache
            local_archive = self.local_cache_dir / archive_name
            await self._download_archive(archive_path, local_archive, source_location)
            
            # Extract if compressed
            if archive_name.endswith('.tar.gz'):
                with tarfile.open(local_archive, 'r:gz') as tar:
                    tar.extractall(self.local_cache_dir)
                json_file = self.local_cache_dir / archive_name.replace('.tar.gz', '.json')
            else:
                json_file = local_archive
            
            # Load archived data
            with open(json_file, 'r') as f:
                archive_data = json.load(f)
            
            entries = archive_data.get('entries', [])
            
            # Restore to database
            from .storage import connect, upsert_item
            con = connect(memory_db_path)
            
            for entry in entries:
                upsert_item(con, entry)
                restore_stats['restored_entries'] += 1
            
            con.close()
            
            # Clean up local files
            local_archive.unlink(missing_ok=True)
            json_file.unlink(missing_ok=True)
            
            print(f"📥 Restored {restore_stats['restored_entries']} memories from {source_location.name}")
            
        except Exception as e:
            restore_stats['errors'].append(str(e))
            print(f"❌ Restore error: {e}")
        
        return restore_stats
    
    async def _find_archive(self, archive_name: str, location: StorageLocation) -> Optional[Path]:
        """Find archive in storage location"""
        try:
            if location.type == 'local':
                archive_path = Path(location.path) / archive_name
                return archive_path if archive_path.exists() else None
            elif location.type == 'nfs' and location.mount_point:
                archive_path = Path(location.mount_point) / archive_name
                return archive_path if archive_path.exists() else None
            return None
        except Exception:
            return None
    
    async def _download_archive(self, remote_path: Path, local_path: Path, location: StorageLocation):
        """Download archive from storage location"""
        if location.type in ['local', 'nfs']:
            shutil.copy2(remote_path, local_path)
    
    async def list_archives(self) -> Dict[str, List[Dict[str, Any]]]:
        """List available archives across all storage locations"""
        archives = {}
        
        for location in self.storage_locations.values():
            if location.enabled:
                location_archives = await self._list_location_archives(location)
                archives[location.name] = location_archives
        
        return archives
    
    async def _list_location_archives(self, location: StorageLocation) -> List[Dict[str, Any]]:
        """List archives in specific storage location"""
        archives = []
        
        try:
            if location.type == 'local':
                archive_dir = Path(location.path)
                if archive_dir.exists():
                    for file_path in archive_dir.glob('archive_*.json*'):
                        stat = file_path.stat()
                        archives.append({
                            'name': file_path.name,
                            'size_mb': stat.st_size / (1024 * 1024),
                            'created': stat.st_ctime,
                            'compressed': file_path.suffix == '.gz'
                        })
            elif location.type == 'nfs' and location.mount_point:
                mount_dir = Path(location.mount_point)
                if mount_dir.exists():
                    for file_path in mount_dir.glob('archive_*.json*'):
                        stat = file_path.stat()
                        archives.append({
                            'name': file_path.name,
                            'size_mb': stat.st_size / (1024 * 1024),
                            'created': stat.st_ctime,
                            'compressed': file_path.suffix == '.gz'
                        })
        except Exception as e:
            print(f"❌ Error listing archives for {location.name}: {e}")
        
        return sorted(archives, key=lambda x: x['created'], reverse=True)
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        stats = {
            'total_locations': len(self.storage_locations),
            'enabled_locations': len([l for l in self.storage_locations.values() if l.enabled]),
            'locations': {},
            'total_archives': 0,
            'total_size_mb': 0
        }
        
        archives_by_location = await self.list_archives()
        
        for location_name, archives in archives_by_location.items():
            location_stats = {
                'archive_count': len(archives),
                'total_size_mb': sum(a['size_mb'] for a in archives),
                'oldest_archive': min([a['created'] for a in archives]) if archives else None,
                'newest_archive': max([a['created'] for a in archives]) if archives else None
            }
            
            stats['locations'][location_name] = location_stats
            stats['total_archives'] += location_stats['archive_count']
            stats['total_size_mb'] += location_stats['total_size_mb']
        
        return stats
    
    async def _save_config(self):
        """Save configuration to disk"""
        config = {
            'storage_locations': [asdict(loc) for loc in self.storage_locations.values()],
            'archive_threshold_days': self.archive_threshold_days,
            'compression_enabled': self.compression_enabled
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    async def cleanup_local_cache(self, max_age_hours: int = 24) -> int:
        """Clean up old files in local cache"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        cleaned_count = 0
        
        for file_path in self.local_cache_dir.iterdir():
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                cleaned_count += 1
        
        return cleaned_count

# High-level API
class NASStorageAPI:
    """High-level NAS storage API"""
    
    def __init__(self):
        self.manager = NASStorageManager()
    
    async def setup_local_storage(self, path: str = None) -> bool:
        """Setup local backup storage"""
        backup_path = path or str(Path.home() / "neuralsync_backup")
        
        location = StorageLocation(
            name="local_backup",
            type="local",
            host="localhost",
            path=backup_path,
            enabled=True
        )
        
        return await self.manager.add_storage_location(location)
    
    async def setup_nfs_storage(self, host: str, remote_path: str, mount_point: str) -> bool:
        """Setup NFS storage"""
        location = StorageLocation(
            name=f"nfs_{host}",
            type="nfs",
            host=host,
            path=remote_path,
            mount_point=mount_point,
            enabled=True
        )
        
        return await self.manager.add_storage_location(location)
    
    async def auto_archive(self, memory_db_path: str) -> Dict[str, Any]:
        """Automatically archive old memories"""
        return await self.manager.archive_memories(memory_db_path)
    
    async def list_available_archives(self) -> Dict[str, List[Dict[str, Any]]]:
        """List all available archives"""
        return await self.manager.list_archives()
    
    async def restore_from_archive(self, archive_name: str, memory_db_path: str) -> Dict[str, Any]:
        """Restore memories from archive"""
        return await self.manager.restore_memories(archive_name, memory_db_path)

# CLI integration example
async def integrate_nas_with_neuralsync():
    """Example integration with NeuralSync"""
    nas_api = NASStorageAPI()
    
    # Setup local backup
    await nas_api.setup_local_storage()
    
    # Auto-archive old memories (run this periodically)
    memory_db_path = str(Path.home() / ".neuralsync" / "memory.db")
    archive_result = await nas_api.auto_archive(memory_db_path)
    
    print(f"Archived {archive_result['archived_entries']} memories")
    print(f"Storage locations: {archive_result['storage_locations']}")
    
    return nas_api

if __name__ == "__main__":
    async def test_nas_system():
        api = NASStorageAPI()
        
        # Setup local storage
        await api.setup_local_storage()
        
        # Test archiving (would need real memory database)
        # result = await api.auto_archive("test.db")
        # print(f"Archive result: {result}")
        
        # List archives
        archives = await api.list_available_archives()
        print(f"Available archives: {archives}")
    
    asyncio.run(test_nas_system())
