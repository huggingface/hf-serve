import importlib
import os
import platform
import shutil
import subprocess
import sys
from typing import List, Optional

from hf_inference_sdk.logging import logger


class DynamicInstaller:
    def __init__(self):
        self.system = platform.system().lower()

    def apt(self, packages: List[str], update: bool = True) -> bool:
        if self.system != "linux":
            logger.warning(f"apt-get not available on {self.system}")
            return False

        if not shutil.which("apt-get"):
            logger.error("apt-get not found")
            return False

        try:
            if update:
                update_cmd = ["apt-get", "update", "-y"]
                if os.getuid() == 0:
                    update_cmd = ["sudo"] + update_cmd
                subprocess.run(update_cmd, check=True, capture_output=True)

            install_cmd = ["apt-get", "install", "-y"]
            if os.getuid() == 0:
                install_cmd = ["sudo"] + install_cmd
            subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            logger.info(f"Successfully installed apt packages: {packages}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install apt packages {packages}: {e.stderr}")
            return False

    def brew(self, packages: List[str]) -> bool:
        if self.system != "darwin":
            logger.warning(f"brew not available on {self.system}")
            return False

        if not shutil.which("brew"):
            logger.error("brew not found")
            return False

        try:
            for package in packages:
                cmd = ["brew", "install", package]
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Successfully installed brew packages: {packages}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install brew packages {packages}: {e.stderr}")
            return False

    def pip(self, packages: List[str], reload_modules: Optional[List[str]] = None) -> bool:
        installer = "uv" if shutil.which("uv") else "pip"

        try:
            if installer == "uv":
                cmd = ["uv", "pip", "install"] + packages
            else:
                cmd = [sys.executable, "-m", "pip", "install"] + packages

            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Successfully installed Python packages via {installer}: {packages}")

            # Reload specified modules to capture new installations
            if reload_modules:
                for module_name in reload_modules:
                    try:
                        if module_name in sys.modules:
                            importlib.reload(sys.modules[module_name])
                        else:
                            importlib.import_module(module_name)
                        logger.info(f"Reloaded module: {module_name}")
                    except ImportError as e:
                        logger.warning(f"Could not reload module {module_name}: {e}")

            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install Python packages {packages} via {installer}: {e.stderr}")
            return False
