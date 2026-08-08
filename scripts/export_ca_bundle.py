"""Export the Windows certificate store to a PEM bundle.

``gee.use_system_certificates()`` fixes anything that goes through this
package's ``initialize()``. It cannot fix tools that do not: the ``earthengine``
CLI, ``pip``, ``curl``, or any script that imports ``requests`` directly. On a
machine where HTTPS is intercepted — AVG Web/Mail Shield here, but a campus
proxy or VPN does the same — those all fail with CERTIFICATE_VERIFY_FAILED
while the browser works perfectly.

This writes every root the OS already trusts, including the interceptor's, to a
single PEM. Pointing ``REQUESTS_CA_BUNDLE`` and ``SSL_CERT_FILE`` at it makes
those tools agree with the rest of the machine.

**This is not a verification bypass.** Certificates are still checked and still
rejected if invalid. The trust *source* changes from a static list shipped
inside ``certifi`` to the store the machine's own browser uses. The alternative
people usually reach for — ``verify=False``, or ``pip --trusted-host`` — accepts
any certificate from anyone, and is a genuinely bad idea.

Run with ``python scripts/export_ca_bundle.py``. It prints the two commands to
set the variables permanently.
"""

from __future__ import annotations

import ssl
from pathlib import Path

OUT = Path.home() / ".certs" / "windows-ca-bundle.pem"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    chunks: list[str] = []
    seen: set[bytes] = set()
    # ROOT holds the trust anchors; CA holds intermediates. An intercepting
    # proxy installs its root into ROOT, which is the one that matters here,
    # but intermediates are cheap to include and occasionally required.
    for store in ("ROOT", "CA"):
        for cert_bytes, encoding, trust in ssl.enum_certificates(store):
            # trust is True for "trusted for all purposes", or a set of OIDs.
            # False means explicitly distrusted - a revoked or blacklisted root,
            # which must not be copied into the bundle.
            if encoding != "x509_asn" or trust is False:
                continue
            if cert_bytes in seen:
                continue
            seen.add(cert_bytes)
            chunks.append(ssl.DER_cert_to_PEM_cert(cert_bytes))

    OUT.write_text("".join(chunks), encoding="ascii")
    print(f"Wrote {len(chunks)} certificates to {OUT}")

    print("\nSet these permanently (PowerShell, then restart your terminal):\n")
    print(f'  setx REQUESTS_CA_BUNDLE "{OUT}"')
    print(f'  setx SSL_CERT_FILE "{OUT}"')
    print("\nThen 'earthengine', 'pip' and anything using requests will work.")


if __name__ == "__main__":
    main()
