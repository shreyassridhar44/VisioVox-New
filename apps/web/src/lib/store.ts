'use client';

/**
 * Client-side session and API access.
 *
 * Zustand selectors are always narrow (repository convention). Subscribing to
 * the whole store re-renders every consumer on any change, which is the
 * documented cause of the player dropping from 60 fps to 10 — the habit is
 * worth keeping everywhere, not only in the player.
 */

import { VisioVoxClient, type Tokens } from '@visiovox/ts-client';
import { create } from 'zustand';

const STORAGE_KEY = 'visiovox.tokens';
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

function loadTokens(): Tokens | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Tokens) : null;
  } catch {
    // Private mode, cleared storage, or a corrupt value: start signed out
    // rather than throwing during hydration.
    return null;
  }
}

function saveTokens(tokens: Tokens | null): void {
  if (typeof window === 'undefined') return;
  try {
    if (tokens) window.localStorage.setItem(STORAGE_KEY, JSON.stringify(tokens));
    else window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* storage unavailable; the session simply will not survive a reload */
  }
}

let clientSingleton: VisioVoxClient | null = null;

export function api(): VisioVoxClient {
  clientSingleton ??= new VisioVoxClient({
    baseUrl: API_BASE,
    tokens: loadTokens() ?? undefined,
    onTokens: (tokens) => {
      saveTokens(tokens);
      useSession.setState({ signedIn: tokens !== null });
    },
  });
  return clientSingleton;
}

interface SessionState {
  signedIn: boolean;
  email: string | null;
  ready: boolean;
  setSignedIn: (signedIn: boolean, email?: string | null) => void;
  hydrate: () => void;
  signOut: () => Promise<void>;
}

export const useSession = create<SessionState>((set) => ({
  signedIn: false,
  email: null,
  ready: false,
  setSignedIn: (signedIn, email = null) => {
    set({ signedIn, email });
  },
  hydrate: () => {
    const tokens = loadTokens();
    set({ signedIn: tokens !== null, ready: true });
  },
  signOut: async () => {
    await api().logout();
    set({ signedIn: false, email: null });
  },
}));
