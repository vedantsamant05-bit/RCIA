import urllib.request, json

# Fetch all items and pending material
all_items = json.loads(urllib.request.urlopen('http://127.0.0.1:8000/api/review').read().decode('utf-8'))
items = json.loads(urllib.request.urlopen('http://127.0.0.1:8000/api/review?status=pending_material').read().decode('utf-8'))

print("--- SUMMARY BAR STATS ---")
unique_clauses = len(set(i['regulation_clause_id'] for i in all_items))
total = len(all_items)
material = len([i for i in all_items if i['impact_type'] != 'no-op'])
dismissed = len([i for i in all_items if i['routing'] == 'auto_dismiss'])
print(f"Regulatory clauses analyzed: {unique_clauses}")
print(f"Candidates retrieved:        {total}")
print(f"Material impacts:            {material}")
print(f"Auto-dismissed:              {dismissed}")

print()
print("--- SAMPLE PENDING MATERIAL ITEM ---")
if items:
    item = items[0]
    print("Impact type:", item['impact_type'])
    print("Confidence:", f"{item['confidence']:.2f} / 1.00 (Heuristic)")
    print()
    print("REDLINE:")
    print(item['draft_redline'])
    print()
    print("BUSINESS ACTION:")
    print(item['business_action'])
else:
    print("No pending material items found (try ingesting a sample first)")
