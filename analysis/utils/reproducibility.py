"""
Reproducibility Utilities

Ensures experiment reproducibility through:
- Seed control for all random operations
- Configuration versioning and saving
- Data checksums
- Environment tracking

Author: Claude Code
Date: 2026-01-15
"""

import json
import random
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import numpy as np

# Global experiment configuration
EXPERIMENT_CONFIG = {
    'random_state': 42,
    'seeds_set': False
}


def set_all_seeds(seed: int = 42):
    """
    Set random seeds for all libraries to ensure reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    EXPERIMENT_CONFIG['random_state'] = seed
    EXPERIMENT_CONFIG['seeds_set'] = True

    # Try to set sklearn seed via environment
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"[Reproducibility] All seeds set to {seed}")


def get_git_commit_hash() -> Optional[str]:
    """Get current git commit hash for version tracking."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except:
        pass
    return None


def get_python_versions() -> Dict[str, str]:
    """Get versions of key Python packages."""
    versions = {}

    try:
        import sys
        versions['python'] = sys.version.split()[0]
    except:
        pass

    packages = ['numpy', 'scipy', 'sklearn', 'pandas', 'matplotlib']
    for pkg in packages:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, '__version__', 'unknown')
        except:
            versions[pkg] = 'not installed'

    return versions


def compute_file_checksum(filepath: Path) -> str:
    """Compute MD5 checksum of a file."""
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)
    return md5.hexdigest()


def compute_data_checksums(data_paths: List[Path]) -> Dict[str, str]:
    """Compute checksums for multiple data files."""
    checksums = {}
    for path in data_paths:
        if path.exists():
            if path.is_file():
                checksums[str(path)] = compute_file_checksum(path)
            elif path.is_dir():
                # For directories, checksum first few files
                for f in sorted(path.glob('*.jsonl'))[:3]:
                    checksums[str(f)] = compute_file_checksum(f)
    return checksums


def save_experiment_config(
    output_path: Path,
    experiment_name: str,
    data_config: Dict,
    model_config: Dict,
    cv_config: Dict,
    additional_config: Optional[Dict] = None
):
    """
    Save complete experiment configuration for reproducibility.

    Args:
        output_path: Path to save configuration JSON
        experiment_name: Name of the experiment
        data_config: Data source configuration
        model_config: Model hyperparameters
        cv_config: Cross-validation configuration
        additional_config: Any additional parameters
    """
    config = {
        'experiment_name': experiment_name,
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'git_commit': get_git_commit_hash(),

        'environment': {
            'python_versions': get_python_versions(),
            'random_state': EXPERIMENT_CONFIG['random_state'],
            'seeds_set': EXPERIMENT_CONFIG['seeds_set']
        },

        'data': data_config,
        'model': model_config,
        'cross_validation': cv_config
    }

    if additional_config:
        config['additional'] = additional_config

    # Compute data checksums if paths provided
    if 'source_paths' in data_config:
        paths = [Path(p) for p in data_config['source_paths']]
        config['data_checksums'] = compute_data_checksums(paths)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)

    print(f"[Reproducibility] Configuration saved to {output_path}")
    return config


def verify_reproducibility(config_path: Path) -> Dict:
    """
    Verify that current environment matches saved configuration.

    Args:
        config_path: Path to saved configuration

    Returns:
        Dict with verification results
    """
    with open(config_path, 'r') as f:
        saved_config = json.load(f)

    current_versions = get_python_versions()
    saved_versions = saved_config.get('environment', {}).get('python_versions', {})

    mismatches = []
    for pkg, saved_ver in saved_versions.items():
        current_ver = current_versions.get(pkg, 'not installed')
        if current_ver != saved_ver:
            mismatches.append({
                'package': pkg,
                'saved': saved_ver,
                'current': current_ver
            })

    return {
        'config_path': str(config_path),
        'reproducible': len(mismatches) == 0,
        'mismatches': mismatches,
        'saved_timestamp': saved_config.get('timestamp'),
        'saved_git_commit': saved_config.get('git_commit')
    }


class ExperimentTracker:
    """
    Track experiment progress and results for reproducibility.
    """

    def __init__(self, experiment_name: str, output_dir: Path):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.start_time = datetime.now()
        self.results = {}
        self.logs = []

        output_dir.mkdir(parents=True, exist_ok=True)

    def log(self, message: str):
        """Log a message with timestamp."""
        timestamp = datetime.now().isoformat()
        entry = f"[{timestamp}] {message}"
        self.logs.append(entry)
        print(entry)

    def add_result(self, key: str, value):
        """Add a result to track."""
        self.results[key] = value

    def save(self):
        """Save all tracked data."""
        output = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'results': self.results,
            'logs': self.logs
        }

        output_path = self.output_dir / f'{self.experiment_name}_tracking.json'
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        self.log(f"Tracking data saved to {output_path}")
        return output_path
