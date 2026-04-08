Source: https://github.com/ChatGPTNextWeb/NextChat
Snapshot commit: c3b8c1587c04fff05f7b42276a43016e87771527
Snapshot date: 2025-09-29

Local modifications:
- Removed the hard-coded Yarn registry override from `Dockerfile` to keep builds on the default registry.
- Forced OpenAI chat requests to use streaming for normal chat completions.
- Added `x-nanobot-session-key` propagation based on the active NextChat session id.
- Forwarded `x-nanobot-session-key` through NextChat's OpenAI proxy route into the upstream proxy.
