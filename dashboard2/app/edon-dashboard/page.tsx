"use client";

import React, { useEffect, useState } from "react";
import { EdonState, fetchEdonState } from "../../src/edonApi";
import CavGauge from "../../src/components/CavGauge";
import DriftWave from "../../src/components/DriftWave";
import HolographicProgressBar from "../../src/components/HolographicProgressBar";

type DriftPoint = { t: number; value: number };
type DashboardState = EdonState & { latencyMs: number };

const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
const clamp = (x: number, min: number, max: number) =>
  Math.min(max, Math.max(min, x));

const initial: DashboardState = {
  state: "Optimal",
  cavScore: 9150,
  riskScore: 0.01,
  riskLabel: "Minimal",
  driftIndex: 0.01,
  driftSeries: [
    { t: 0, value: 0.01 },
    { t: 1, value: -0.02 },
    { t: 2, value: 0.03 },
    { t: 3, value: -0.01 },
    { t: 4, value: 0.02 },
    { t: 5, value: -0.03 },
    { t: 6, value: 0.01 },
    { t: 7, value: 0.04 },
    { t: 8, value: -0.02 },
    { t: 9, value: 0.01 },
  ],
  bio: 0.60,
  env: 0.88,
  circadian: 0.92,
  pStress: 0.36,
  latencyMs: 42,
};

export default function EdonDashboardPage() {
  const [data, setData] = useState<DashboardState>(initial);

  // Fetch real data
  useEffect(() => {
    const tick = async () => {
      try {
        const start = performance.now();
        const state = await fetchEdonState();
        const latency = performance.now() - start;

        setData((prev) => ({
          ...state,
          driftSeries: state.driftSeries.length > 0 ? state.driftSeries : prev.driftSeries,
          latencyMs: latency,
        }));
      } catch (err) {
        // Keep using existing data if fetch fails
      }
    };

    tick();
    const id = setInterval(tick, 2000);
    return () => clearInterval(id);
  }, []);

  // Subtle data updates
  useEffect(() => {
    const id = setInterval(() => {
      setData((prev) => {
        const nextDrift: DriftPoint[] = [
          ...prev.driftSeries.slice(1),
          {
            t: prev.driftSeries[prev.driftSeries.length - 1].t + 1,
            value: clamp(
              prev.driftSeries[prev.driftSeries.length - 1].value + (Math.random() - 0.5) * 0.01,
              -10,
              10
            ),
          },
        ];

        return {
          ...prev,
          cavScore: clamp(prev.cavScore + (Math.random() - 0.5) * 20, 8500, 9500),
          driftIndex: clamp(prev.driftIndex + (Math.random() - 0.5) * 0.001, 0, 0.1),
          driftSeries: nextDrift,
        };
      });
    }, 3000);

    return () => clearInterval(id);
  }, []);

  const factorBars = [
    { label: "Bio", value: data.bio, color: "cyan" as const },
    { label: "Env", value: data.env, color: "cyan" as const },
    { label: "Circadian", value: data.circadian, color: "cyan" as const },
    { label: "P(Stress)", value: data.pStress, color: "pink" as const },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-950">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-50 mb-2">
            EDON Live Inference
          </h1>
          <p className="text-slate-600 dark:text-slate-400">
            Real-time cognitive assessment and physiological monitoring
          </p>
        </div>

        {/* Key Metrics Row */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          {/* CAV Score Card */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6 border border-slate-200 dark:border-slate-700">
            <div className="text-sm font-medium text-slate-600 dark:text-slate-400 mb-2">
              CAV Score
            </div>
            <div className="text-4xl font-bold text-cyan-600 dark:text-cyan-400 mb-1">
              {Math.round(data.cavScore).toLocaleString()}
            </div>
            <div className="text-xs text-slate-500 dark:text-slate-500">
              Range: 0 - 10,000
            </div>
          </div>

          {/* State Card */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6 border border-slate-200 dark:border-slate-700">
            <div className="text-sm font-medium text-slate-600 dark:text-slate-400 mb-2">
              Current State
            </div>
            <div className="text-2xl font-semibold text-slate-900 dark:text-slate-50 mb-1">
              {data.state}
            </div>
            <div className="text-xs text-slate-500 dark:text-slate-500">
              Cognitive assessment status
            </div>
          </div>

          {/* Risk Score Card */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6 border border-slate-200 dark:border-slate-700">
            <div className="text-sm font-medium text-slate-600 dark:text-slate-400 mb-2">
              Risk Level
            </div>
            <div className="text-2xl font-semibold text-emerald-600 dark:text-emerald-400 mb-1">
              {data.riskLabel}
            </div>
            <div className="text-xs text-slate-500 dark:text-slate-500">
              Score: {data.riskScore.toFixed(3)}
            </div>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left: Progress Bars */}
          <div className="lg:col-span-1 bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6 border border-slate-200 dark:border-slate-700">
            <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-50 mb-4">
              Physiological Factors
            </h2>
            <div className="space-y-4">
              {factorBars.map((bar, idx) => (
                <HolographicProgressBar
                  key={idx}
                  label={bar.label}
                  value={bar.value}
                  color={bar.color}
                />
              ))}
            </div>
          </div>

          {/* Right: Charts */}
          <div className="lg:col-span-2 space-y-6">
            {/* CAV Gauge */}
            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6 border border-slate-200 dark:border-slate-700">
              <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-50 mb-4">
                CAV Score Visualization
              </h2>
              <div className="flex justify-center">
                <CavGauge value={data.cavScore} max={10000} size={240} />
              </div>
            </div>

            {/* Drift Chart */}
            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6 border border-slate-200 dark:border-slate-700">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-50">
                  Drift over Time
                </h2>
                <div className="text-sm text-slate-600 dark:text-slate-400">
                  Index: {data.driftIndex.toFixed(3)}
                </div>
              </div>
              <DriftWave data={data.driftSeries} width={600} height={200} />
            </div>
          </div>
        </div>

        {/* Footer Info */}
        <div className="mt-8 text-center text-sm text-slate-500 dark:text-slate-400">
          <div className="inline-flex items-center gap-4">
            <span>Latency: {data.latencyMs.toFixed(1)}ms</span>
            <span>•</span>
            <span>Last updated: {new Date().toLocaleTimeString()}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
