"""
Run this ONCE, before your first real session, to initialize Cognee's local
database schema. This is not something to re-run before every session -
prune_system()/prune_data() wipes stored memory, which would defeat the
entire point of session persistence.

Root cause this fixes: on a completely fresh install, Cognee's SQLite database
file/schema doesn't exist until something initializes it. Calling recall()
before that initialization has ever happened fails with
"sqlite3.OperationalError: unable to open database file" - not a permissions
issue, just a missing first-run step.

Usage:
    python scripts/init_cognee.py

Run this exactly once per machine/environment. If you ever want to
deliberately wipe memory and start over (e.g. testing session 1 fresh again),
re-run it - but know that it deletes everything remembered so far.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cognee
from memory.cognee_config import configure


async def main():
    configure()
    print("Initializing Cognee database schema (this wipes any existing memory)...")
    await cognee.prune_data()
    await cognee.prune_system(metadata=True)
    print("Done. Cognee's local database is now initialized and empty.")
    print("You can now run scripts/run_session.py --mode cold safely.")


if __name__ == "__main__":
    asyncio.run(main())