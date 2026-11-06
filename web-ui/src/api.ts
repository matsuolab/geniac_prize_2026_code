// The access token comes in the URL fragment the server printed (#token=...). It is kept
// in this origin's storage, not a cookie: cookies are sent to every port on the host, so
// another user's service on the same machine would receive them.
const TOKEN_KEY = 'capx-webui-token';
const TOKEN_SUBPROTOCOL_PREFIX = 'capx-token.';

// In the Vite dev server, a separate origin with its own lifetime, the token stays in
// memory only: once the dev server stops, another user can take its port and would read
// whatever this origin stored while the backend's token is still valid. There, reopen
// the token URL after a reload.
const PERSIST_TOKEN = !import.meta.env.DEV;

let memoryToken: string | null = null;

function clearFragment(): void {
  window.history.replaceState(null, '', window.location.pathname + window.location.search);
}

/** Take the token from #token=... into storage (memory only in the dev server). Returns
 *  whether there was one. In the built UI the fragment is removed only once the token is
 *  stored, so a reload still finds it. */
export function adoptTokenFromUrl(): boolean {
  const match = window.location.hash.match(/(?:^#|&)token=([^&]+)/);
  if (!match) return false;
  memoryToken = decodeURIComponent(match[1]);
  if (!PERSIST_TOKEN) {
    clearFragment();
    return true;
  }
  try {
    localStorage.setItem(TOKEN_KEY, memoryToken);
    clearFragment();
  } catch {
    // Storage disabled: keep the token in memory and in the address bar.
  }
  return true;
}

function accessToken(): string | null {
  if (memoryToken || !PERSIST_TOKEN) return memoryToken;
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

/** fetch() for the web UI's own /api routes, with the access token. */
export function apiFetch(input: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  const token = accessToken();
  if (token) headers.set('Authorization', `Bearer ${token}`);
  return fetch(input, { ...init, headers });
}

/** WebSocket subprotocols that carry the access token (browsers cannot set headers). */
export function tokenSubprotocols(): string[] {
  const token = accessToken();
  return token ? [`${TOKEN_SUBPROTOCOL_PREFIX}${token}`] : [];
}
