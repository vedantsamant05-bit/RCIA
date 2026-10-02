/**
 * RCIA Frontend Configuration
 *
 * Configures the backend API URL (API_BASE):
 * - If running locally on a frontend dev server (e.g. python -m http.server 5500),
 *   it directs requests to the FastAPI backend on port 8000 of the same host.
 * - If served directly from FastAPI (port 8000) or deployed remotely (Vercel),
 *   it uses window.location.origin.
 */
(function () {
  "use strict";

  function resolveApiBase() {
    if (typeof window !== "undefined" && window.__RCIA_API_BASE__) {
      return window.__RCIA_API_BASE__;
    }
    const origin = window.location.origin;
    const hostname = window.location.hostname;
    const port = window.location.port;

    // Running on Vercel or any remote cloud domain
    if (hostname && !["localhost", "127.0.0.1", "0.0.0.0"].includes(hostname)) {
      return origin;
    }

    // Direct FastAPI backend serving on port 8000
    if (port === "8000") {
      return origin;
    }

    // Local static server (e.g. port 5500, 3000) or file:// -> target FastAPI on port 8000
    const host = hostname || "127.0.0.1";
    return `http://${host}:8000`;
  }

  window.RCIA_CONFIG = {
    API_BASE: resolveApiBase(),
  };

  console.info(
    "[RCIA] API_BASE =",
    window.RCIA_CONFIG.API_BASE,
    "| Origin =",
    window.location.origin
  );
})();