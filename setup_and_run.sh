#!/bin/bash

# Update dependencies
echo "Updating langchain-community and chromadb..."
pip3 install --upgrade langchain-community chromadb

# Path to your Streamlit app script
SCRIPT_PATH="/Users/markevans/rag_project/streamlit_rag.py"

# Backup the original file
cp "$SCRIPT_PATH" "${SCRIPT_PATH}.bak"
echo "Created a backup of the original Streamlit app script at ${SCRIPT_PATH}.bak"

# Update the script: Remove `fetch_k=k*2` argument from similarity_search call
sed -i "" "s/, fetch_k=k*2//g" "$SCRIPT_PATH"
echo "Updated the Streamlit app script to remove fetch_k argument."

# Run the Streamlit app
echo "Starting Streamlit app..."
streamlit run "$SCRIPT_PATH"

