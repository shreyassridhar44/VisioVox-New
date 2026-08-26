/**
 * Typed VisioVox API client.
 *
 * Request and response shapes come from `./generated/api`, which is produced
 * from `packages/contracts/openapi.json`. Never hand-edit the generated file —
 * CI regenerates and diffs it, so a spec change that would break this client
 * fails on the pull request that caused it rather than after deploy.
 *
 * Token refresh is handled here rather than in each caller. A single-flight
 * guard means a burst of concurrent 401s triggers one refresh, not one per
 * request — and since refresh tokens rotate, parallel refreshes would consume
 * each other's tokens and trip the server's reuse detection, logging the user
 * out for doing nothing wrong.
 */

import type { components } from './generated/api';

export type Schemas = components['schemas'];
export type TokenResponse = Schemas['TokenResponse'];
export type UserResponse = Schemas['UserResponse'];
export type ProjectResponse = Schemas['ProjectResponse'];
export type ProjectListResponse = Schemas['ProjectListResponse'];
export type JobResponse = Schemas['JobResponse'];
export type UploadInitResponse = Schemas['UploadInitResponse'];

export const CLIENT_VERSION = '0.1.0' as const;

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
  ) {
    super(`${String(status)}: ${detail}`);
    this.name = 'ApiError';
  }
}

export interface Tokens {
  accessToken: string;
  refreshToken: string;
}

export interface ClientOptions {
  baseUrl: string;
  tokens?: Tokens | undefined;
  /** Called whenever tokens change, so the caller can persist them. */
  onTokens?: ((tokens: Tokens | null) => void) | undefined;
  fetchImpl?: typeof fetch | undefined;
}

export class VisioVoxClient {
  private tokens: Tokens | null;
  private readonly baseUrl: string;
  private readonly onTokens: ((tokens: Tokens | null) => void) | undefined;
  private readonly fetchImpl: typeof fetch;
  /** In-flight refresh, shared by every request that hits a 401 at once. */
  private refreshing: Promise<boolean> | null = null;

  constructor(options: ClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/$/, '');
    this.tokens = options.tokens ?? null;
    this.onTokens = options.onTokens;
    this.fetchImpl = options.fetchImpl ?? globalThis.fetch.bind(globalThis);
  }

  private setTokens(tokens: Tokens | null): void {
    this.tokens = tokens;
    this.onTokens?.(tokens);
  }

  private async raw(path: string, init: RequestInit = {}): Promise<Response> {
    const headers = new Headers(init.headers);
    if (!headers.has('content-type') && init.body !== undefined) {
      headers.set('content-type', 'application/json');
    }
    if (this.tokens) {
      headers.set('authorization', `Bearer ${this.tokens.accessToken}`);
    }
    return this.fetchImpl(`${this.baseUrl}${path}`, { ...init, headers });
  }

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    let response = await this.raw(path, init);

    if (response.status === 401 && this.tokens) {
      const refreshed = await this.refreshOnce();
      if (refreshed) {
        response = await this.raw(path, init);
      }
    }

    if (!response.ok) {
      throw new ApiError(response.status, await this.readDetail(response));
    }
    return (await response.json()) as T;
  }

  /** For endpoints that answer 204 and have no body to parse. */
  private async requestNoContent(path: string, init: RequestInit = {}): Promise<void> {
    let response = await this.raw(path, init);
    if (response.status === 401 && this.tokens) {
      if (await this.refreshOnce()) {
        response = await this.raw(path, init);
      }
    }
    if (!response.ok) {
      throw new ApiError(response.status, await this.readDetail(response));
    }
  }

  private async readDetail(response: Response): Promise<string> {
    try {
      const body: unknown = await response.json();
      if (typeof body === 'object' && body !== null && 'detail' in body) {
        // `'detail' in body` already narrows this; no assertion needed.
        const { detail } = body;
        return typeof detail === 'string' ? detail : JSON.stringify(detail);
      }
      return JSON.stringify(body);
    } catch {
      return response.statusText;
    }
  }

  /** At most one refresh in flight; concurrent callers await the same promise. */
  private refreshOnce(): Promise<boolean> {
    this.refreshing ??= this.doRefresh().finally(() => {
      this.refreshing = null;
    });
    return this.refreshing;
  }

  private async doRefresh(): Promise<boolean> {
    if (!this.tokens) return false;
    const response = await this.fetchImpl(`${this.baseUrl}/v1/auth/refresh`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ refresh_token: this.tokens.refreshToken }),
    });
    if (!response.ok) {
      // Reuse detection revokes the family, so there is nothing to retry with.
      this.setTokens(null);
      return false;
    }
    const body = (await response.json()) as TokenResponse;
    this.setTokens({
      accessToken: body.access_token,
      refreshToken: body.refresh_token,
    });
    return true;
  }

  // ---- auth ----

  async register(email: string, password: string, displayName?: string): Promise<TokenResponse> {
    const body = await this.request<TokenResponse>('/v1/auth/register', {
      method: 'POST',
      body: JSON.stringify({
        email,
        password,
        ...(displayName !== undefined ? { display_name: displayName } : {}),
      }),
    });
    this.setTokens({ accessToken: body.access_token, refreshToken: body.refresh_token });
    return body;
  }

  async login(email: string, password: string): Promise<TokenResponse> {
    const body = await this.request<TokenResponse>('/v1/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
    this.setTokens({ accessToken: body.access_token, refreshToken: body.refresh_token });
    return body;
  }

  async logout(): Promise<void> {
    if (!this.tokens) return;
    await this.requestNoContent('/v1/auth/logout', {
      method: 'POST',
      body: JSON.stringify({ refresh_token: this.tokens.refreshToken }),
    });
    this.setTokens(null);
  }

  me(): Promise<UserResponse> {
    return this.request<UserResponse>('/v1/auth/me');
  }

  // ---- projects ----

  createProject(title: string): Promise<ProjectResponse> {
    return this.request<ProjectResponse>('/v1/projects', {
      method: 'POST',
      body: JSON.stringify({ title, rights_attested: true }),
    });
  }

  listProjects(limit = 20): Promise<ProjectListResponse> {
    return this.request<ProjectListResponse>(`/v1/projects?limit=${String(limit)}`);
  }

  getProject(id: string): Promise<ProjectResponse> {
    return this.request<ProjectResponse>(`/v1/projects/${id}`);
  }

  getJob(id: string): Promise<JobResponse> {
    return this.request<JobResponse>(`/v1/projects/${id}/job`);
  }

  getManifest(id: string): Promise<unknown> {
    return this.request<unknown>(`/v1/projects/${id}/manifest`);
  }

  // ---- upload ----

  uploadInit(
    projectId: string,
    file: { name: string; type: string; size: number },
    partCount = 1,
  ): Promise<UploadInitResponse> {
    return this.request<UploadInitResponse>(`/v1/projects/${projectId}/upload/init`, {
      method: 'POST',
      body: JSON.stringify({
        filename: file.name,
        content_type: file.type || 'application/octet-stream',
        size_bytes: file.size,
        part_count: partCount,
      }),
    });
  }

  uploadComplete(
    projectId: string,
    uploadId: string,
    key: string,
    parts: { part_number: number; etag: string }[],
  ): Promise<{ job_id: string; status: string }> {
    return this.request(`/v1/projects/${projectId}/upload/complete`, {
      method: 'POST',
      body: JSON.stringify({ upload_id: uploadId, key, parts }),
    });
  }

  /**
   * Progress stream URL. EventSource cannot set an Authorization header, so
   * the caller passes the access token as a query parameter or proxies the
   * stream server-side; the Next.js app does the latter.
   */
  eventsUrl(projectId: string): string {
    return `${this.baseUrl}/v1/projects/${projectId}/events`;
  }
}
