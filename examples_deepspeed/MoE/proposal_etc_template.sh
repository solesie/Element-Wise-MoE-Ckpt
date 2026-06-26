#!/bin/bash
# Deprecated wrapper. The old ETC experiment template has been retired.
# Use proposal_test_template.sh for proposal test/debug runs.
set -euo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
exec bash "$DIR/proposal_test_template.sh" "$@"
