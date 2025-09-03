#!/bin/bash
# Publish to TestPyPI

set -e

echo "Publishing to TestPyPI..."
echo "========================"

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
if [ -z "$TESTPYPI_TOKEN" ]; then
	echo "Error: TESTPYPI_TOKEN not found!"
	echo ""
	echo "Please add your token to .secrets file:"
	echo "  cp .secrets.example .secrets"
	echo "  chmod 600 .secrets"
	echo "  # Edit .secrets and add your token"
	echo ""
	echo "Get your token from: https://test.pypi.org/manage/account/token/"
	exit 1
fi

# Show what will be uploaded
echo ""
echo "Files to upload:"
ls -lh dist/

echo ""
echo "Uploading to TestPyPI..."
uv publish --publish-url https://test.pypi.org/legacy/ --token "$TESTPYPI_TOKEN"

echo ""
echo "Success! Package uploaded to TestPyPI"
echo ""
echo "Test installation with:"
echo "  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ revoxx"
echo ""
echo "View package at: https://test.pypi.org/project/revoxx/"
