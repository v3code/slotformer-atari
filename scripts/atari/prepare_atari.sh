#!/bin/bash
pip install gym[atari]
pip install autorom[accept-rom-license]
mkdir ./Roms
AutoROM --accept-license --install-dir ./Roms
