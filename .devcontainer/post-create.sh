#!/usr/bin/env bash

# Stop immediately if any command fails so a broken development environment
# is not silently treated as successfully configured.
set -e

# Install the exact uv version used by this project.
# Pinning the version keeps future Codespaces reproducible instead of
# unexpectedly installing a newer uv release with different behavior.
curl -LsSf https://astral.sh/uv/0.12.17/install.sh | sh

# Make uv available in every future bash terminal opened by this user.
# The grep check prevents duplicate PATH entries when the script is run again.
grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" || \
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
