#!/bin/bash

OUTPUT="dump.txt"
> "$OUTPUT"

while IFS= read -r file; do
    echo "================ $file ================" >> "$OUTPUT"
    cat "$file" >> "$OUTPUT"
    echo -e "\n\n" >> "$OUTPUT"
done < <(find . -name "*.py" \
    -not -path "./.venv/*" \
    -not -path "./.git/*" \
    -not -path "*/__pycache__/*" \
    | sort)

echo "Done → $OUTPUT"