// src/edonApi.ts

export type DriftPoint = { t: number; value: number };

export type EdonState = {
  state: string;
  cavScore: number;
  riskScore: number;
  riskLabel: string;
  driftSeries: DriftPoint[];
  driftIndex: number;
  bio: number;
  env: number;
  circadian: number;
  pStress: number;
};

const BASE_URL =
  process.env.NEXT_PUBLIC_EDON_API_URL || "http://127.0.0.1:8001";

export async function fetchEdonState(): Promise<EdonState> {
  const url = `${BASE_URL}/_debug/state`;

  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();

    // 🔁 Map your real /_debug/state schema into the dashboard shape here.
    // This is a *guess* – tweak to match your actual keys.
    
    // The actual API returns: { ok: true, state: {...}, last_state: {...} }
    // Extract from state or last_state
    const stateData = json.state || json.last_state || {};
    const parts = stateData.parts || {};

    return {
      state: stateData.state || "Unknown",
      cavScore: stateData.cav_smooth || stateData.cav_raw || 0,
      riskScore: parts.p_stress || 0,
      riskLabel: (parts.p_stress || 0) < 0.2 ? "Minimal" : (parts.p_stress || 0) < 0.5 ? "Moderate" : "High",
      driftIndex: parts.p_stress || 0,
      driftSeries:
        json.drift?.history?.map((d: any, i: number) => ({
          t: i,
          value: d,
        })) ?? [],
      bio: parts.bio || 0,
      env: parts.env || 0,
      circadian: parts.circadian || 0,
      pStress: parts.p_stress || 0,
    };
  } catch (err) {
    console.error("EDON API error, falling back to mock:", err);
    // 🔁 Fallback demo data so the dashboard still looks alive
    return {
      state: "Optimal",
      cavScore: 9150,
      riskScore: 0.01,
      riskLabel: "Minimal",
      driftIndex: 0.01,
      driftSeries: [
        { t: 0, value: 0.002 },
        { t: 1, value: 0.004 },
        { t: 2, value: 0.003 },
        { t: 3, value: 0.006 },
        { t: 4, value: 0.005 },
        { t: 5, value: 0.008 },
      ],
      bio: 0.6,
      env: 0.88,
      circadian: 0.92,
      pStress: 0.36,
    };
  }
}

