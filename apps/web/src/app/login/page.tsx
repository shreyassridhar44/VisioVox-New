'use client';

import { useRouter } from 'next/navigation';
import { useState } from 'react';
import { api, useSession } from '@/lib/store';

export default function LoginPage() {
  const router = useRouter();
  const setSignedIn = useSession((s) => s.setSignedIn);
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: React.SyntheticEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (mode === 'register') await api().register(email, password);
      else await api().login(email, password);
      setSignedIn(true, email);
      router.push('/projects');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong');
    } finally {
      setBusy(false);
    }
  }

  return (
    <form
      className="panel stack"
      onSubmit={(e) => {
        void submit(e);
      }}
      style={{ maxWidth: '26rem' }}
    >
      <h1>{mode === 'login' ? 'Sign in' : 'Create an account'}</h1>
      <div>
        <label htmlFor="email">Email</label>
        <input
          id="email"
          type="email"
          value={email}
          required
          onChange={(e) => {
            setEmail(e.target.value);
          }}
        />
      </div>
      <div>
        <label htmlFor="password">Password</label>
        <input
          id="password"
          type="password"
          value={password}
          required
          minLength={mode === 'register' ? 12 : 1}
          onChange={(e) => {
            setPassword(e.target.value);
          }}
        />
        {mode === 'register' && <p className="muted">At least 12 characters.</p>}
      </div>
      {error !== null && <p className="error">{error}</p>}
      <button type="submit" disabled={busy}>
        {busy ? 'Working…' : mode === 'login' ? 'Sign in' : 'Create account'}
      </button>
      <p className="muted">
        <a
          href="#"
          onClick={(e) => {
            e.preventDefault();
            setMode(mode === 'login' ? 'register' : 'login');
            setError(null);
          }}
        >
          {mode === 'login' ? 'Need an account?' : 'Already have an account?'}
        </a>
      </p>
    </form>
  );
}
