import os
import importlib
import sys
import subprocess

def install_if_missing(package):
    try:
        importlib.import_module(package)
        print(f"Package {package} already installed.")
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Installing package {package}...")

def install_missing_packages(packages):
    for package in packages:
        install_if_missing(package)

def to_float(s):
    return float(s.replace(",", "."))