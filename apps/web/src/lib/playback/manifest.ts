/**
 * The playback view of the artifact manifest (contract v1.0).
 *
 * These types mirror `packages/contracts/schemas/manifest.schema.json`, which
 * is frozen. They are hand-written rather than generated because the manifest
 * is served as an opaque body by `GET /projects/{id}/manifest` — it is a CDN
 * document rather than an API response shape, so it never entered the OpenAPI
 * spec that produces `@visiovox/ts-client`. `test_manifest_schema.py` guards
 * the schema itself; these types have to be kept in step with it by hand.
 */

export type SpeakerId = string;

/**
 * The unprocessed original is always selectable, under a reserved id. It is
 * not a speaker, so it cannot live in `manifest.speakers`, but the player
 * treats it as just another track — which is what makes A/B comparison against
 * the mixture free rather than a special case.
 */
export const MIXED_TRACK = 'mixed';

/**
 * A speaker id, or the reserved `MIXED_TRACK`. Deliberately not written as a
 * union with the literal: speaker ids are opaque server-issued strings, so the
 * union collapses to `string` and only looks like it is checking something.
 */
export type TrackId = SpeakerId;

export type Modality = 'audiovisual' | 'audio_only' | 'visual_only';
export type AudioMode = 'faithful' | 'natural';
export type PlaybackHint = 'webaudio' | 'hls';

export interface AudioAsset {
  readonly url: string;
  readonly bytes: number;
}

export interface Speaker {
  readonly id: SpeakerId;
  readonly ordinal: number;
  readonly label: string;
  readonly color_token: string;
  readonly modality: Modality;
  readonly thumbnail_url?: string;
  readonly speaking_ratio: number;
  readonly mean_confidence: number;
  readonly extraction_ok: boolean;
  // `faithful` and `vtt` are required by the schema, not optional-in-practice:
  // Faithful is the default track and the one that gets transcribed, because
  // generative restoration can hallucinate words (invariant 6). Typing them as
  // present is what lets `tracksFor` guarantee every speaker reaches the rail.
  readonly audio: {
    readonly faithful: AudioAsset;
    readonly natural?: AudioAsset;
    readonly hls?: string;
  };
  readonly peaks_url?: string;
  readonly captions: {
    readonly vtt: string;
    readonly json?: string;
  };
}

export interface VideoTrack {
  readonly url: string;
  readonly width: number;
  readonly height: number;
}

export interface Manifest {
  readonly project_id: string;
  readonly manifest_version: string;
  readonly duration_ms: number;
  readonly has_video: boolean;
  readonly difficulty?: string;
  readonly overlap_ratio?: number;
  readonly video?: VideoTrack;
  readonly speakers: readonly Speaker[];
  readonly mixed: { readonly audio_url: string };
  readonly playback_hint: PlaybackHint;
  readonly warnings: readonly string[];
  readonly signed_until: string;
}

/** One decodable audio stream, resolved from the manifest for a given mode. */
export interface Track {
  readonly id: TrackId;
  readonly url: string;
  readonly label: string;
  readonly bytes: number;
}

/**
 * Resolve the playable tracks for an audio mode.
 *
 * Natural is optional per speaker (S6 is feature-flagged, so a project may
 * ship Faithful-only), and falling back silently is right here: the mode is a
 * listening preference, not a correctness property, and a missing Natural
 * render should not remove a speaker from the rail.
 */
export function tracksFor(manifest: Manifest, mode: AudioMode): Track[] {
  const tracks: Track[] = [
    { id: MIXED_TRACK, url: manifest.mixed.audio_url, label: 'Original mix', bytes: 0 },
  ];
  for (const speaker of manifest.speakers) {
    const asset =
      mode === 'natural'
        ? (speaker.audio.natural ?? speaker.audio.faithful)
        : speaker.audio.faithful;
    tracks.push({ id: speaker.id, url: asset.url, label: speaker.label, bytes: asset.bytes });
  }
  return tracks;
}

/** Whether any speaker in the manifest offers a Natural render. */
export function hasNaturalMode(manifest: Manifest): boolean {
  return manifest.speakers.some((s) => s.audio.natural !== undefined);
}
