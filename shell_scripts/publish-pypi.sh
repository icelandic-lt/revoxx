#!/bin/bash
# Publish to PyPI (Production)

set -e

# Parse command line arguments
FORCE=false
while [[ $# -gt 0 ]]; do
	case $1 in
	-f | --force)
		FORCE=true
		shift
		;;
	*)
		echo "Unknown option: $1"
		echo "Usage: $0 [-f|--force]"
		exit 1
		;;
	esac
done

echo "Publishing to PyPI (Production)..."
echo "=================================="

# Load tokens from .secrets file if it exists
if [ -f ".secrets" ]; then
	source .secrets
else
	echo "Warning: .secrets file not found!"
	echo "Please copy .secrets.example to .secrets and add your tokens."
	echo ""
fi

# Check if dist directory exists and has files
if [ ! -d "dist" ] || [ -z "$(ls -A dist)" ]; then
	echo "Building package first..."
	uv build
fi

# Check for token
if [ -z "$PYPI_TOKEN" ]; then
	echo "Error: PYPI_TOKEN not found!"
	echo ""
	echo "Please add your token to .secrets file:"
	echo "  cp .secrets.example .secrets"
	echo "  chmod 600 .secrets"
	echo "  # Edit .secrets and add your token"
	echo ""
	echo "Get your token from: https://pypi.org/manage/account/token/"
	exit 1
fi

# Confirmation prompt
echo ""
echo "Files to upload:"
ls -lh dist/
echo ""

if [ "$FORCE" = false ]; then
	read -p "Are you sure you want to upload to PRODUCTION PyPI? (yes/no): " confirm

	if [ "$confirm" != "yes" ]; then
		echo "Upload cancelled."
		exit 0
	fi
fi

echo ""
echo "Uploading to PyPI..."
uv publish --token "$PYPI_TOKEN"

echo ""
echo "Success! Package uploaded to PyPI"
echo ""
echo "Install with:"
echo "  pip install revoxx"
echo "  pip install revoxx[vad]  # with VAD support"
echo ""
echo "View package at: https://pypi.org/project/revoxx/"
