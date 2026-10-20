# Vendored read-only telemetry

Source: https://github.com/carpentry-liu/rl-physics-overlay

Revision: `8472a74a3c66473d885f9dce1aa1b5a3dfe6b347`, inspected 2026-09-03.

License: MIT, retained in LICENSE. Only five runtime modules are included;
no build scripts, demo assets, remote connections, or optional event adapters.
The upstream source files are unmodified.

Enable with `?telemetry=1`. The wrapper adds the separately measured payload
contact forces, penetration and applied-force audit to the diagnostic area.
The upstream adapter reports joint generalized forces, not spatial contact
wrenches. Unsupported wrench channels remain unavailable.
