# 19 — Testing Strategy

---

## 1. Shape

```
                ▲  Manual / exploratory       — device matrix, listening tests
               ╱ ╲
              ╱E2E╲                            ~40  Playwright, critical journeys
             ╱─────╲
            ╱  Integ ╲                        ~200  API+DB, worker+storage, contracts
           ╱──────────╲
          ╱    Unit    ╲                     ~1200  pure logic, fast
         ╱──────────────╲
        ╱  ML Evaluation ╲                   ◄── the axis a normal pyramid lacks
       ╱──────────────────╲                      quick (PR) + full (nightly)
```

The ML evaluation layer is the addition. Correctness for this product is not only "does the code
run" but "is the output good" — and that is a different kind of test, with different gates and a
different cadence.

---

## 2. Unit

| Area | Focus | Notes |
|---|---|---|
| API | Validation, authz predicates, state machine, quota | No DB — repositories mocked |
| Workers | Stage orchestration, retry, idempotency keys | Models mocked |
| ML | Loss functions, enrolment purity scoring, chunking/overlap-add, mask application | ⭐ Numerical assertions |
| Frontend | Stores, hooks, engine logic, caption index | `AudioContext` mocked |
| Shared | Timecode conversion, manifest parsing | Property-based |

Numerical tests worth writing explicitly, because they catch the errors that are otherwise found at
epoch 40:

```python
def test_si_sdr_perfect_reconstruction():
    s = torch.randn(16000)
    assert si_sdr(s, s) > 100                    # scale-invariant, so exact → huge

def test_si_sdr_scale_invariance():
    s = torch.randn(16000)
    assert torch.allclose(si_sdr(s * 3.7, s), si_sdr(s, s), atol=1e-3)

def test_suppression_loss_hinges():
    """Beyond tau, extra suppression must stop contributing gradient."""
    assert suppression_loss(very_clean, interferers, tau=-10.0) == 0.0

def test_overlap_add_reconstructs():
    """Chunking + crossfade must be lossless on a passthrough model."""
    x = torch.randn(160000)
    assert torch.allclose(overlap_add(chunk(x)), x, atol=1e-5)

def test_silence_loss_penalises_leakage():
    """The core product requirement, as a unit test."""
    silent_target = torch.zeros(16000)
    assert silence_loss(quiet_leakage, silent_target) > 0
```

`test_overlap_add_reconstructs` is the one that repays itself most: a subtly wrong crossfade window
produces a quiet periodic artifact that is nearly invisible in metrics and obvious to a listener.

Coverage: backend ≥ 80%, ML ≥ 70%, frontend ≥ 70% (NFR-MAINT-01). Coverage is a floor, not a goal.

---

## 3. Integration

Real Postgres, real Redis, real MinIO (testcontainers). Models mocked.

| Suite | Verifies |
|---|---|
| Auth flows | Register → verify → login → refresh → **reuse detection revokes family** |
| Upload | Presign → PUT → complete → job enqueued |
| **Ownership** | ⭐ Generated from the OpenAPI spec: User B gets 404 on every one of User A's resources |
| Job lifecycle | State transitions, retries, resume-from-stage, cancellation |
| Retention | Sweepers delete the right things and only those things |
| **Deletion** | ⭐ Project delete removes every object; receipt is accurate; keys stop resolving |
| Quotas | Enforced at the boundary; correct error shape |
| Contract | Generated spec matches committed spec |

The ownership suite is generated, not hand-written — so a new endpoint is covered the moment it
appears in the spec. This is the control that makes IDOR (the highest-impact threat in
[`15-security.md`](./15-security.md) §1) structurally hard to introduce.

---

## 4. End-to-end

Playwright, against a full stack with the **mock pipeline**.

| Journey | Assertions |
|---|---|
| Signup → verify → first upload | Email verification gate works |
| Upload → progress → ready | SSE events arrive; stages advance; ETA present |
| **Play → switch speaker → captions swap** | ⭐ The core product loop |
| Seek → sync holds | Audio and video aligned after seek |
| Rename speaker → propagates | Label updates in captions and exports |
| Download audio + captions | Correct files, correct filenames, mode encoded |
| Share link → open in a clean context → play | Read-only, no download unless allowed |
| Delete project → gone | Receipt available; media URLs 403 |
| Reduced motion | No WebGL context created |
| Keyboard-only | Full journey without a mouse |
| Mobile viewport | Layout and player usable at 360 px |

---

## 5. Audio-sync tests ⭐

The suite that measures the actual product claim. Detailed in
[`12-media-pipeline-and-sync.md`](./12-media-pipeline-and-sync.md) §7.

```ts
test('speaker switch is under 120ms and seamless', async ({ page }) => {
  await page.goto('/projects/fixture-3spk');
  await page.getByRole('button', { name: 'Play' }).click();

  const result = await page.evaluate(async () => {
    const analyser = window.__testHooks.attachAnalyser();
    const t0 = performance.now();
    await window.__testHooks.selectSpeaker('spk_2');
    const settled = await analyser.waitForTrackSignature('spk_2');
    return { latency: performance.now() - t0, minRms: analyser.minRmsDuringTransition() };
  });

  expect(result.latency).toBeLessThan(120);          // NFR-PERF-03
  expect(result.minRms).toBeGreaterThan(0.7);        // no >3dB dip → equal-power crossfade works
});

test('drift stays under 40ms over 10 minutes', async ({ page }) => {
  // ... plays at 8x with a fixture, samples getDrift() every 5s
  expect(Math.max(...drifts)).toBeLessThan(40);      // FR-PLAY-05
});
```

Fixtures include tracks with deliberately **mismatched sample counts** to prove invariant I3's
assertion actually fires. A test that only exercises correct inputs does not test an assertion.

---

## 6. ML evaluation tests

Two tiers:

| Tier | Scope | When | Gate |
|---|---|---|---|
| `CI-EVAL-QUICK` | 30 items, ~5 min | Every PR touching `ml/` | Blocking |
| `CI-EVAL-FULL` | AMI-Eval, ~2 h | Nightly + pre-release | Blocking on release |

Regression gates (from [`08-evaluation-protocol.md`](./08-evaluation-protocol.md) §6):

| Metric | Block if |
|---|---|
| SI-SDRi | drops > 0.5 dB |
| SIR | drops > 1.0 dB |
| WER | rises > 1.0 point |
| Hallucination rate | rises > 0.5 pt absolute |
| Trust ECE | > 0.05 |
| RTF | rises > 20% |

Plus **pipeline smoke tests** on 5 fixture videos covering: 2-speaker clean, 3-speaker overlapping,
no-face, audio-only, and a pathological input (near-silence, single speaker, extreme reverb). These
assert the pipeline *completes* and produces a schema-valid manifest — not that quality is good.
Fast, and they catch integration breakage that metric gates miss.

---

## 7. Security tests

| Test | Method |
|---|---|
| Cross-tenant access | Generated from spec (§3) |
| Auth bypass | Schemathesis with malformed/absent/expired/tampered tokens |
| **Media sandbox** | ⭐ Corpus of malicious media: decode bombs, polyglots, fuzzed containers, symlink escapes, path traversal in filenames |
| **Sandbox escape** | Attempt network access, file writes outside the scratch volume, fork bomb, credential access — all must fail |
| Rate limits | Burst tests |
| CSP | Automated header check; report-only monitoring in staging |
| Injection | SQLi/XSS payloads through every text field, including speaker labels and titles |
| Secrets | gitleaks over full history |
| Dependencies | Trivy, pip-audit, pnpm audit |

The malicious media corpus is the highest-value security test in the project, because the media
decoder is the highest-risk component. Sources: ffmpeg's own FATE fuzzing corpus, oss-fuzz artifacts,
and hand-crafted cases from the limits table in [`15-security.md`](./15-security.md) §4.

---

## 8. Performance tests

| Test | Target |
|---|---|
| Lighthouse CI (landing, player) | ≥ 95 all categories |
| Bundle size | Budget in [`13-frontend-architecture.md`](./13-frontend-architecture.md) §6 |
| API load (k6) | p95 < 200 ms at 100 rps |
| Pipeline throughput | RTF ≤ 2.0× |
| Concurrent jobs | 10 simultaneous without degradation |
| Hero FPS | ≥ 50 desktop / ≥ 30 mobile on a throttled profile |

---

## 9. Release gate

All must pass before production:

**Automated**
- [ ] All test suites green
- [ ] Coverage thresholds met
- [ ] `CI-EVAL-FULL` — no regression gate tripped
- [ ] Lighthouse ≥ 95, bundle within budget
- [ ] Zero axe violations
- [ ] Trivy: zero CRITICAL
- [ ] Contract check: no spec drift
- [ ] Migration up **and down** tested on a production-sized seed

**Manual**
- [ ] Device matrix: macOS/iOS Safari, Android Chrome, Windows Chrome/Firefox
- [ ] Audio verified on Bluetooth, wired, and laptop speakers
- [ ] Listening spot-check on 5 real clips — someone actually listens
- [ ] Security checklist ([`15-security.md`](./15-security.md) §12)
- [ ] Privacy checklist ([`16-privacy-compliance.md`](./16-privacy-compliance.md) §9)
- [ ] Model licence audit (the ⚠️ rows in [`05-ml-architecture.md`](./05-ml-architecture.md) §14)
- [ ] Runbook current
- [ ] Rollback rehearsed in staging
- [ ] **Limitations documented and accurate to the shipped build**

The last item is a real gate. If the model got worse at 3 speakers, the landing page and the docs
change before release — not after a user notices.

---

## 10. Test data

| Fixture | Purpose |
|---|---|
| `2spk_clean_30s.mp4` | Happy path |
| `3spk_overlap_60s.mp4` | Core case |
| `4spk_hard_60s.mp4` | Degradation |
| `noface_audio_only.mp4` | Modality fallback |
| `audio_only.wav` | No video path |
| `single_speaker.mp4` | Passthrough behaviour |
| `near_silence.mp4` | ⭐ Whisper hallucination guard |
| `extreme_reverb.mp4` | Front-end stress |
| `corrupt_header.mp4` | Rejection path |
| `decode_bomb.mp4` | Security |
| `mismatched_lengths/` | ⭐ Invariant I3 assertion |

Fixtures are small (< 5 MB each), committed via Git LFS, and generated by a reproducible script so
they can be regenerated rather than trusted.

`near_silence.mp4` earns its place: isolated tracks are mostly silence, Whisper hallucinates on
silence, and that combination is the most likely source of confidently wrong captions in production
([`05-ml-architecture.md`](./05-ml-architecture.md) §10).
