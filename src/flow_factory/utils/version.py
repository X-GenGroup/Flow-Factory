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

from packaging import version
import importlib.metadata

def compare_lib_version(lib_name: str, target_version: str) -> int:
    """
    Compare the version of given lib and target version
    
    返回结果:
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