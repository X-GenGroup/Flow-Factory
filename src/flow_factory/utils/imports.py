# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# src/flow_factory/utils/version.py

"""
Version comparison utilities.
"""
from functools import lru_cache
from packaging import version
import importlib.metadata
import importlib.util

def compare_lib_version(lib_name: str, target_version: str) -> int:
    """
    Compare the version of given lib and target version
    
    Args:
        lib_name: Name of the library to check
        target_version: Target version string to compare against
    Returns:
        1 : installed version  > target version
        0 : installed version == target version
        -1 : installed version  < target version
        None: Not installed
    """
    try:
        installed_ver = importlib.metadata.version(lib_name)
    except importlib.metadata.PackageNotFoundError:
        return None

    # Parser version index
    v_installed = version.parse(installed_ver)
    v_target = version.parse(target_version)

    if v_installed > v_target:
        return 1
    elif v_installed < v_target:
        return -1
    else:
        return 0

def is_version_at_least(lib_name: str, min_version: str) -> bool:
    res = compare_lib_version(lib_name, min_version)
    return res is not None and res >= 0


def _is_package_available(pkg_name: str, metadata_name: str = None) -> bool:
    """Check if a package is installed and importable."""
    if importlib.util.find_spec(pkg_name) is None:
        return False
    try:
        importlib.metadata.metadata(metadata_name or pkg_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


@lru_cache
def is_flash_attn_available(min_version: str = None) -> bool:
    """
    Check if flash-attn is installed and meets minimum version requirement.
    
    Args:
        min_version: Optional minimum version (e.g., "2.0.0")
        
    Returns:
        True if flash-attn is available (and meets version requirement if specified)
    """
    if not _is_package_available("flash_attn", "flash-attn"):
        return False
    
    if min_version is not None:
        return is_version_at_least("flash-attn", min_version)
    
    return True


@lru_cache
def get_flash_attn_version() -> str | None:
    """Return installed flash-attn version, or None if not installed."""
    try:
        return importlib.metadata.version("flash-attn")
    except importlib.metadata.PackageNotFoundError:
        return None