import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

print(f"Checking URL: {url}")
try:
    client = create_client(url, key)
    # Try a simple public request
    res = client.table("users").select("*", count="exact").limit(1).execute()
    print("✅ Connection Success: Supabase URL and Key are correct.")
except Exception as e:
    print(f"❌ Connection Failed: {e}")
    if "sb_publishable" in str(key):
        print("   HINT: You are still using a Stripe key! Get the 'anon' key from Supabase Settings > API.")
