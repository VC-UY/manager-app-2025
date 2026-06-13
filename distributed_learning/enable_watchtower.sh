#!/bin/bash
set -e

# Start the volunteer service normally, then start optional Watchtower auto-update.
# Run this on the volunteer machine.

docker compose up -d

docker compose --profile watchtower up -d

echo "Watchtower auto-update enabled. It will check for new images every 5 minutes."
