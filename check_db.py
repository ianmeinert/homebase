import sqlite3

conn = sqlite3.connect("data/homebase.db")
rows = conn.execute("SELECT id, title, status FROM registry ORDER BY id").fetchall()
import json
seeded = {i["id"] for i in json.load(open("data/registry.json"))}
print("HV items in DB:")
for r in rows:
    if r[0].startswith("HV-"):
        flag = "" if r[0] in seeded else " <-- ARTIFACT"
        print(f"  {r[0]}  {r[1]}  [{r[2]}]{flag}")
conn.close()
