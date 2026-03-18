#!/usr/bin/env python3
"""Delete the Hot Aisle VM via the admin TUI.

Navigation sequence:
  1. SSH to admin.hotaisle.app (TUI loads)
  2. Wait for team list, press Enter to select team "Yaakoub"
  3. Press 'j' to navigate down to VM, press Enter
  4. Press 'G' to go to end of operations (Delete VM)
  5. Press Enter to open delete dialog
  6. Press 'y' to confirm normal delete
  7. Wait for completion, exit
"""

import sys
import time

try:
    import pexpect
except ImportError:
    print("ERROR: pexpect not available")
    sys.exit(1)


def delete_vm():
    print("Starting VM deletion via admin.hotaisle.app TUI...")

    child = pexpect.spawn(
        'ssh -tt admin.hotaisle.app',
        encoding='utf-8',
        timeout=60,
        maxread=200000,
    )

    # Step 1: Wait for TUI to load (team selection)
    print("  Waiting for TUI to load...")
    time.sleep(12)

    # Step 2: Select team (press Enter)
    print("  Selecting team...")
    child.send('\r')
    time.sleep(5)

    # Step 3: Navigate to VM and enter management
    print("  Navigating to VM...")
    child.send('j')  # down to VM
    time.sleep(1)
    child.send('\r')  # enter VM management
    time.sleep(5)

    # Step 4: Go to end of operations list (Delete VM)
    print("  Navigating to Delete VM option...")
    child.send('G')  # go to end
    time.sleep(2)

    # Step 5: Press Enter on Delete VM
    print("  Opening delete confirmation...")
    child.send('\r')
    time.sleep(5)

    # Step 6: Confirm with 'y' (normal delete)
    print("  Confirming deletion with 'y'...")
    child.send('y')
    time.sleep(10)

    # Read any final output
    try:
        output = child.read_nonblocking(size=100000, timeout=5)
    except (pexpect.TIMEOUT, pexpect.EOF):
        output = ""

    print("  Waiting for deletion to complete...")
    time.sleep(5)

    child.sendcontrol('c')
    time.sleep(2)

    try:
        child.close()
    except Exception:
        pass

    print("VM deletion sequence completed.")
    return 0


if __name__ == '__main__':
    sys.exit(delete_vm())
