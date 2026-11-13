"""PlatformIO pre-script: refresh build_version.h on every build.

PlatformIO only re-runs the CMake configure step (where main/CMakeLists.txt
generates the header) when build files change, so a new commit alone would
leave a stale revision embedded in the firmware.
"""
import os
import subprocess
import sys

Import("env")  # noqa: F821  (SCons global)

project = env.subst("$PROJECT_DIR")  # noqa: F821
header = os.path.join(env.subst("$BUILD_DIR"), "esp-idf", "main", "generated", "build_version.h")  # noqa: F821
os.makedirs(os.path.dirname(header), exist_ok=True)
subprocess.check_call([sys.executable, os.path.join(project, "tools", "generate_version.py"),
                       "--root", project, "--profile", env.BoardConfig().get("build.mcu"),  # noqa: F821
                       "--output", header])
