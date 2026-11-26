---
name: file-manager
description: "Read, write, list, move, and delete files inside the file root"
min_capability: tier-1
requires:
  connectors: [file]
---
# File Manager

You can read, write, list, move, and delete files inside one configured root.

## How to handle file requests

When a user asks to:
- "Read the file config.yaml"
- "Create a file called notes.txt with this content"
- "List files in the data directory"
- "Move notes.txt to archive/notes.txt"
- "Delete the old export"

Do the following:
1. Identify the operation (read, write, list, move, or delete)
2. Identify the path relative to the file root
3. Use file__read, file__write, file__list, file__move, or file__delete
4. Present the result clearly

## Safety rules

- Every path stays inside the file root (`LUMEN_FILE_ROOT`, or the module runtime `files` directory when that is unset). Do not try to leave it. `..`, absolute paths outside the root, and symlinks are refused.
- file__write can replace an existing file and the confirmation gate does not stop it. Ask the user before replacing a file that already exists. The result field `overwrote` says whether that happened.
- file__move renames inside the root and does not replace a destination that already exists.
- file__delete is destructive. The confirmation gate asks before it runs. Delete a non-empty directory only when the user asked for the whole tree, and pass `recursive=true`. A tree that contains a symlink is refused.
- For large files, summarize content rather than showing everything
- Warn the user if a file path looks sensitive (.env, credentials, keys)

## Examples

User: "Read my config file"
→ file__read(path="config.yaml")
→ Display the content in a code block

User: "Save this as notes.txt: Meeting at 3pm with design team"
→ file__write(path="notes.txt", content="Meeting at 3pm with design team")
→ "Saved to notes.txt"

User: "Delete the exports directory and everything in it"
→ file__delete(path="exports", recursive=true)
→ Wait for the confirmation gate, then say what was removed
