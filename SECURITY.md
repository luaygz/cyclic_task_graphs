# Security

Do not commit `.env`, API keys, browser state, raw retrieved pages, MongoDB
dumps, or unsanitized trajectories. Service ports in `compose.yaml` bind only
to loopback. Rotate credentials immediately if a secret is ever committed,
including in private history.

Report vulnerabilities privately to the repository owner once the public
hosting location and contact channel are selected.

