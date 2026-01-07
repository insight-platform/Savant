#!/bin/bash

VERSION=$1

if [ -z "$VERSION" ]; then
    # No version specified, pull latest and savant-latest tags
    echo "Updating images with 'latest' and 'savant-latest' tags..."
    
    IMAGES=$(docker images | grep ghcr.io/insight-platform | grep -E "(latest|savant-latest)" | awk '{print $1}')
    
    if [ -z "$IMAGES" ]; then
        echo "No images found with 'latest' or 'savant-latest' tags"
        exit 1
    fi
    
    echo "Found images:"
    echo "$IMAGES"
    echo ""
    
    echo "$IMAGES" | xargs -n 1 docker pull
else
    # Version specified, pull vX.Y.Z tags
    echo "Updating images with version v$VERSION..."
    
    IMAGES=$(docker images | grep ghcr.io/insight-platform | grep -E ":(v)?$VERSION" | awk '{print $1}')
    
    if [ -z "$IMAGES" ]; then
        echo "No images found with version v$VERSION"
        exit 1
    fi
    
    echo "Found images:"
    echo "$IMAGES"
    echo ""
    
    echo "$IMAGES" | xargs -n 1 docker pull
fi

echo "Image update completed!"