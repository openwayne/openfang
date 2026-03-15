#!/bin/bash
# Mock copilot script to generate sample JSON output
if [[ "$*" == *"--output-format json"* ]]; then
  echo '{"message": "I am a helpful assistant.", "usage": {"input_tokens": 10, "output_tokens": 20}}'
else
  echo "I am a helpful assistant."
fi
