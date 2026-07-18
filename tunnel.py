"""
Run this script to get a public URL for FarmSense AI.
Your Flask server must already be running on port 5000.

STEP 1: Get a free token from https://dashboard.ngrok.com/signup
STEP 2: Paste your token below where it says YOUR_TOKEN_HERE
STEP 3: Run: python tunnel.py
"""
from pyngrok import ngrok, conf

# ── PASTE YOUR NGROK AUTH TOKEN HERE ──────────────────────────────────────────
NGROK_TOKEN = "YOUR_TOKEN_HERE"
# ──────────────────────────────────────────────────────────────────────────────

conf.get_default().auth_token = NGROK_TOKEN

# Open a tunnel to your running Flask server on port 5000
tunnel = ngrok.connect(5000, "http")

print("\n" + "="*60)
print("  ✅ FarmSense AI is now PUBLIC!")
print(f"  🌐 Your public URL: {tunnel.public_url}")
print(f"  📋 Dashboard link:  {tunnel.public_url}/dashboard.html")
print("="*60)
print("\n  Share the link above for evaluation.")
print("  Keep this window open — closing it stops the tunnel.")
print("  Press Ctrl+C to stop.\n")

# Keep running
try:
    ngrok.run()
except KeyboardInterrupt:
    print("\nTunnel closed.")
    ngrok.disconnect(tunnel.public_url)
