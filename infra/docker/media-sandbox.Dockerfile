# Confinement for ffmpeg/ffprobe running on attacker-supplied media (ADR-0009).
#
# Deliberately minimal: the image contains ffmpeg and nothing else useful to an
# attacker who achieves code execution inside it. No shell utilities beyond
# busybox, no package manager at runtime, no credentials, and - enforced at run
# time, not here - no network.
#
# Built locally rather than pulled so the security-critical container is not a
# third-party image that can change under us, and so the stack still works with
# no registry access.
FROM alpine:3.20

RUN apk add --no-cache ffmpeg=6.1.2-r1 || apk add --no-cache ffmpeg

# 65534:65534 is nobody:nogroup. The runtime also passes --user, so this is
# defence in depth rather than the only thing standing between ffmpeg and root.
USER 65534:65534

# Scratch only. The rootfs is mounted read-only at run time; /work is the
# writable bind mount the orchestrating worker provides.
WORKDIR /work

# No ENTRYPOINT on purpose: the caller supplies the full argv, which keeps the
# command visible at the call site instead of hidden in the image.
