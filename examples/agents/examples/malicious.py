def strategy(prices, position, cash):
    """DANGEROUS: This agent tries to escape the sandbox.

    In local mode, this just fails harmlessly.
    In cluster mode, Kata/MSHV isolation prevents any damage.
    """
    import os
    # Try to read sensitive files (should fail in sandbox)
    try:
        with open("/etc/shadow") as f:
            print(f"BREACH: {f.read()[:50]}")
    except PermissionError:
        print("Blocked: cannot read /etc/shadow")

    # Try to access the network (should be blocked by network policy)
    try:
        import urllib.request
        urllib.request.urlopen("http://169.254.169.254/metadata", timeout=2)
        print("BREACH: accessed metadata endpoint")
    except Exception:
        print("Blocked: cannot access metadata endpoint")

    # Try to consume all memory (should be killed by resource limits)
    # Uncomment to test: data = "x" * (10 ** 10)

    # Still return a valid action so trading continues
    return "hold"
