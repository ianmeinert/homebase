# cleanup_test_artifacts.ps1
# Removes registry items that are not part of the original 30 seeded items

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$dbPath = Join-Path $scriptDir "data\homebase.db"
$seedPath = Join-Path $scriptDir "data\registry.json"
$helperScript = Join-Path $scriptDir "_cleanup_helper.py"

# Write the Python helper to a temp file
@"
import sqlite3, json, sys

db_path = sys.argv[1]
seed_path = sys.argv[2]

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

seeded = {i['id'] for i in json.load(open(seed_path))}
all_items = conn.execute('SELECT id, title FROM registry ORDER BY id').fetchall()
artifacts = [r for r in all_items if r['id'] not in seeded]

if not artifacts:
    print('No test artifacts found. Registry is clean.')
else:
    print(f'Found {len(artifacts)} artifact(s):')
    for r in artifacts:
        print(f'  [{r["id"]}] {r["title"]}')
    confirm = input('Delete these items? (y/N): ').strip().lower()
    if confirm == 'y':
        for r in artifacts:
            conn.execute('DELETE FROM registry WHERE id = ?', (r['id'],))
        conn.commit()
        print(f'Deleted {len(artifacts)} item(s).')
    else:
        print('Cancelled.')

conn.close()
"@ | Out-File -FilePath $helperScript -Encoding utf8

& ".venv\Scripts\python.exe" $helperScript $dbPath $seedPath

Remove-Item $helperScript -Force
