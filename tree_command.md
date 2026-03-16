find . -type d ! -path "*/.git*" ! -path "*/node_modules*" ! -path "*/venv" ! -path "*/__pycache__"  | while read d; do   ✭ ✱ 
  files=$(find "$d" -maxdepth 1 -type f | wc -l)
  if [ "$files" -le 10 ]; then
    echo "$d"
  fi
done | tree --fromfile