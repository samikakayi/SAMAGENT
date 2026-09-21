# Security model

SAM is capable, but it is deliberately not an unrestricted administrator. Its
default policy protects the machine even when a model makes a mistake or reads
malicious instructions from a file or webpage.

## Default decisions

Automatically allowed:

- List, search, and read ordinary files inside the selected workspace.
- Create a new file inside the workspace and make bounded project edits.
- Inspect project status and run configured read-only checks.
- Read local memories and plans.
- Navigate to an ordinary `http` or `https` URL.

Always requires an explicit, single-use approval:

- Deleting, truncating, or overwriting an existing file.
- Writing or moving anything outside the workspace.
- Arbitrary PowerShell, terminal, Python, or project script execution when the
  action is not on the safe command list.
- Installing, upgrading, or removing software or packages.
- Registry, service, scheduled-task, firewall, ACL, or other system changes.
- Access to credential-like paths or secret material.
- Launching an unknown executable or using a non-web URL protocol.
- Uploading a file, submitting a form, sending a message, posting data, making a
  purchase, or any comparable external side effect.

Blocked by default, even if a model asks:

- UAC/admin elevation or `RunAs`.
- Encoded/obfuscated commands, `Invoke-Expression`, or an action whose nested
  behavior cannot be inspected.
- Credential dumping, security-control disabling, boot/disk/partition wiping,
  destructive shadow-copy changes, ransomware-like bulk mutation, hidden
  remote shells, or automating SAM's own approval controls.
- Live broker order placement. The MetaTrader adapter exports market data only;
  no order execution endpoint or tool is registered.

Computer Control and Screen Access are separate runtime permissions and both
start OFF. Enabling them does not bypass the action policy, single-use
approvals, protected-path rules, or verification requirements.

An approval is bound to the normalized tool name, arguments, working directory,
and session. Changing any of them invalidates it. There is no "approve all"
mode.

## Workspace and secrets

Paths are resolved before checking containment. Parent traversal, sibling-prefix
confusion, device paths, alternate data streams, and link/junction escapes are
rejected or escalated. Sensitive locations such as SSH keys, cloud credentials,
browser profiles, cookie stores, password databases, and files whose names look
like credentials are never sent to a cloud model automatically.

Use Windows Credential Manager or another OS-backed secret broker for real
credentials. `.env` is supported for local development, but it must remain
untracked and its values must never be pasted into chat.

## Audit trail

The audit log records the session, model/provider, requested tool, redacted
arguments, policy decision and reason, approval decision, timing, result status,
and bounded output metadata. Entries are linked with hashes so accidental or
casual tampering is detectable. SAM exposes audit data read-only; the model has
no tool for editing or deleting it.

## Important limitation

A child process running as the same Windows user is not a true operating-system
sandbox. Timeouts, environment scrubbing, working-directory restrictions, and
approval gates reduce risk, but an approved arbitrary script can still access
resources available to that user. Use Windows Sandbox, Hyper-V, a VM, or a
restricted container for untrusted repositories or scripts. Do not start SAM as
Administrator.

## Reporting problems

Preserve the relevant redacted audit event and the command/tool request. Never
attach secrets, raw authorization headers, cookies, or private keys to a bug
report.
