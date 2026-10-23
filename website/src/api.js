const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();
  if (!response.ok) {
    const detail = payload?.detail || payload?.message || "The request could not be completed.";
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload;
}

export const getConfig = () => request("/api/config");
export const getHistory = () => request("/api/history?limit=50");
export const deleteRun = (runId) => request(`/api/history/${runId}`, { method: "DELETE" });

export const predictOne = (payload) =>
  request("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

export const predictCsv = (file) => {
  const body = new FormData();
  body.append("file", file);
  return request("/api/predict/csv", { method: "POST", body });
};

export const getGlobalShap = (outputName) =>
  request(`/api/shap/global/${encodeURIComponent(outputName)}`);

export const getLocalShap = (payload) =>
  request("/api/shap/local", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
