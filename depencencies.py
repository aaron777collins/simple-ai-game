import subprocess
import ensurepip
from sys import platform
import sys
import os.path
from os import path


INSTALL_FILE_NAME = "installed.txt"


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def checkLinux():
    return platform == "linux" or platform == "linux2"


def installAll():
    # ensuring pip is installed

    if(path.exists(INSTALL_FILE_NAME) ):
        return

    if checkLinux():
        subprocess.check_call(["sudo", "apt-get", "install", "python3-pip"])
    else:
        ensurepip.bootstrap()

    install("pygame")
    install("tensorflow")
    install("tf-agents")
    install("imageio==2.4.0")
    install("pyvirtualdisplay")
    install("ipython")
    install("matplotlib")

    print("\n\n########################################\nAll Dependencies Installed.\n")

    # marks as installed by creating dummy file
    open(INSTALL_FILE_NAME, 'a').close()