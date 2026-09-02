// Thin client for the FastAPI underwriting backend.
//
// Development: requests go to `/api/*` and Vite proxies them to the backend
// (see vite.config.js), so the browser stays same-origin.
// Production build: set VITE_API_BASE_URL to the API's absolute origin; the
// backend must then list the app's origin in AIU_CORS_ORIGINS.

const BASE = (import.meta.env.VITE_API_BASE_URL || "/api").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(status, detail) {
    super(ApiError.messageFrom(status, detail));
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }

  static messageFrom(status, detail) {
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      // FastAPI 422: [{ loc: [...], msg, type }, ...]
      return detail
        .map((d) => {
          const field = Array.isArray(d.loc) ? d.loc[d.loc.length - 1] : d.loc;
          return field ? `${field}: ${d.msg}` : d.msg;
        })
        .join("; ");
    }
    return `Request failed (${status})`;
  }
}

async function request(path, { method = "GET", body, signal } = {}) {
  let res;
  try {
    res = await fetch(`${BASE}${path}`, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal,
    });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new ApiError(
      0,
      `Cannot reach the API at ${BASE}. Is the backend running (uvicorn app.backend.main:app)?`,
    );
  }

  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) {
    throw new ApiError(res.status, data && data.detail != null ? data.detail : res.statusText);
  }
  return data;
}

export const api = {
  health: (opts) => request("/health", opts),
  predict: (application, opts) =>
    request("/predict", { method: "POST", body: application, ...opts }),
  explain: (application, opts) =>
    request("/explain", { method: "POST", body: application, ...opts }),
  adverseAction: (application, opts) =>
    request("/adverse-action", { method: "POST", body: application, ...opts }),
  review: (application, opts) =>
    request("/review", { method: "POST", body: application, ...opts }),
  listApplications: ({ decision, limit = 100 } = {}, opts) => {
    const qs = new URLSearchParams();
    if (decision) qs.set("decision", decision);
    if (limit) qs.set("limit", String(limit));
    const q = qs.toString();
    return request(`/applications${q ? `?${q}` : ""}`, opts);
  },
  getApplication: (id, opts) => request(`/applications/${id}`, opts),
};
