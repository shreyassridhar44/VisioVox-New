'use client';

/**
 * Client-side session and API access.
 *
 * **Nothing is persisted.** The access token lives in memory for its ten
 * minutes and the refresh token is an httpOnly cookie this code cannot read
 * (docs/28 §W4). Storing either in localStorage would mean any XSS on any page
 * is a full account takeover that survives a reload — the exact thing the
 * cookie exists to prevent, undone by convenience.
 *
 * The cost is that a reload starts signed-out until the first refresh returns.
 * `hydrate()` performs that refresh, so the gap is one request rather than a
 * re-login.
 *
 * Zustand selectors are always narrow (repository convention). Subscribing to
 * the whole store re-renders every consumer on any change, which is the
 * documented cause of the player dropping from 60 fps to 10.
 */

import { VisioVoxClient, type Tokens } from '@visiovox/ts-client';
import { create } from 'zustand';

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

/** Memory only. Deliberately not localStorage, sessionStorage or a cookie. */
let inMemoryTokens: Tokens | null = null;

let clientSingleton: VisioVoxClient | null = null;

export function api(): VisioVoxClient {
  clientSingleton ??= new VisioVoxClient({
    baseUrl: API_BASE,
    onTokens: (tokens) => {
      inMemoryTokens = tokens;
      useSession.setState({ signedIn: tokens !== null });
    },
  });
  return clientSingleton;
}

/**
 * The current access token, for the one caller that cannot use a header.
 *
 * `EventSource` has no way to set Authorization, so the progress stream takes
 * the token on the query string. Access tokens only: ten minutes and revocable,
 * which is what makes that acceptable. The refresh token is not reachable from
 * here at all, by construction.
 */
export function accessToken(): string | null {
  return inMemoryTokens?.accessToken ?? null;
}

interface SessionState {
  signedIn: boolean;
  email: string | null;
  ready: boolean;
  setSignedIn: (signedIn: boolean, email?: string | null) => void;
  hydrate: () => Promise<void>;
  signOut: () => Promise<void>;
}

export const useSession = create<SessionState>((set) => ({
  signedIn: false,
  email: null,
  ready: false,
  setSignedIn: (signedIn, email = null) => {
    set({ signedIn, email });
  },
  hydrate: async () => {
    // Recover the session from the cookie. A 401 here is the ordinary
    // signed-out case, not an error worth surfacing.
    try {
      const me = await api().me();
      set({ signedIn: true, email: me.email, ready: true });
    } catch {
      set({ signedIn: false, email: null, ready: true });
    }
  },
  signOut: async () => {
    await api().logout();
    inMemoryTokens = null;
    set({ signedIn: false, email: null });
  },
}));
