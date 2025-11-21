"use client";

import React, { useEffect, useState } from "react";
import dynamic from "next/dynamic";

// Dynamically import Recharts to avoid hydration errors
const LineChart = dynamic(
  () => import("recharts").then((mod) => mod.LineChart),
  { ssr: false }
);
const Line = dynamic(
  () => import("recharts").then((mod) => mod.Line),
  { ssr: false }
);
const XAxis = dynamic(
  () => import("recharts").then((mod) => mod.XAxis),
  { ssr: false }
);
const YAxis = dynamic(
  () => import("recharts").then((mod) => mod.YAxis),
  { ssr: false }
);
const Tooltip = dynamic(
  () => import("recharts").then((mod) => mod.Tooltip),
  { ssr: false }
);
const RadialBarChart = dynamic(
  () => import("recharts").then((mod) => mod.RadialBarChart),
  { ssr: false }
);
const RadialBar = dynamic(
  () => import("recharts").then((mod) => mod.RadialBar),
  { ssr: false }
);
const PolarAngleAxis = dynamic(
  () => import("recharts").then((mod) => mod.PolarAngleAxis),
  { ssr: false }
);

type DriftPoint = { t: number; value: number };

type EdonState = {
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

type DashboardState = EdonState & { latencyMs: number };

// Humanoid removed

const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
const clamp = (x: number, min: number, max: number) =>
  Math.min(max, Math.max(min, x));

const initial: DashboardState = {
  state: "restorative",
  cavScore: 9997,
  riskScore: 0.0,
  riskLabel: "Minimal",
  driftIndex: 0.0,
  driftSeries: [
    { t: 0, value: 0.003 },
    { t: 1, value: 0.004 },
    { t: 2, value: 0.002 },
    { t: 3, value: 0.006 },
    { t: 4, value: 0.005 },
    { t: 5, value: 0.007 },
  ],
  bio: 1.0,
  env: 1.0,
  circadian: 1.0,
  pStress: 0.0,
  latencyMs: 42,
};

export default function Home() {
  const [data, setData] = useState<DashboardState>(initial);
  const [shouldPulse, setShouldPulse] = useState(false);
  const [particles, setParticles] = useState<Array<{id: number, left: number, top: number, size: number, duration: number, delay: number}>>([]);
  
  // Generate particles on client side only
  useEffect(() => {
    setParticles(
      Array.from({ length: 20 }, (_, i) => ({
        id: i,
        left: Math.random() * 100,
        top: Math.random() * 100,
        size: Math.random() * 3 + 1,
        duration: 15 + Math.random() * 10,
        delay: Math.random() * 5,
      }))
    );
  }, []);

  // simple animation so it feels alive
  useEffect(() => {
    const id = setInterval(() => {
      setData((prev) => {
        const nextDrift: DriftPoint[] = [
          ...prev.driftSeries.slice(1),
          {
            t: prev.driftSeries[prev.driftSeries.length - 1].t + 1,
            value:
              prev.driftSeries[prev.driftSeries.length - 1].value +
              (Math.random() - 0.5) * 0.003,
          },
        ];

        const newCavScore = clamp(
          prev.cavScore + (Math.random() - 0.5) * 40,
          0,
          10000
        );
        if (newCavScore !== prev.cavScore) {
          setShouldPulse(true);
          setTimeout(() => setShouldPulse(false), 600);
        }
        return {
          ...prev,
          latencyMs: 35 + Math.random() * 15,
          cavScore: newCavScore,
          driftIndex: prev.driftIndex + (Math.random() - 0.5) * 0.0005,
          pStress: clamp01(prev.pStress + (Math.random() - 0.5) * 0.02),
          driftSeries: nextDrift,
        };
      });
    }, 1800);

    return () => clearInterval(id);
  }, []);

  const cavGaugeData = [
    {
      name: "CAV",
      value: clamp(data.cavScore, 0, 10000),
      fill: "rgba(56,189,248,1)", // cyan
    },
  ];

  const factorBars = [
    { label: "Bio", value: data.bio, color: "bg-cyan-400" },
    { label: "Env", value: data.env, color: "bg-cyan-300" },
    { label: "Circadian", value: data.circadian, color: "bg-cyan-500" },
    { label: "P(Stress)", value: data.pStress, color: "bg-pink-400" },
  ];

  return (
    <div className="relative min-h-screen bg-slate-950 text-slate-50 overflow-hidden">
      {/* background glow with animation */}
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(56,189,248,0.3),_transparent_55%),radial-gradient(circle_at_bottom,_rgba(236,72,153,0.25),_transparent_55%)] animate-pulse" style={{ animationDuration: '8s' }} />
      {/* background grid */}
      <div className="pointer-events-none absolute inset-0 opacity-40 [background-image:linear-gradient(#020617_1px,transparent_1px),linear-gradient(90deg,#020617_1px,transparent_1px)] [background-size:80px_80px]" />
      
      {/* Floating particles background */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        {particles.map((p) => (
          <div
            key={p.id}
            className="absolute rounded-full bg-cyan-400/20"
            style={{
              width: `${p.size}px`,
              height: `${p.size}px`,
              left: `${p.left}%`,
              top: `${p.top}%`,
              animation: `floatUp ${p.duration}s infinite`,
              animationDelay: `${p.delay}s`,
            }}
          />
        ))}
      </div>

      {/* latency label (top-right like the render) */}
      <div className="absolute top-6 right-10 text-xs text-slate-300 z-30">
        Latency: {data.latencyMs.toFixed(1)} ms
      </div>


      {/* main content: everything vertically centered over torso */}
      <div className="relative flex items-center justify-center min-h-screen px-8">
        <div className="flex items-center gap-10 max-w-6xl w-full">
          {/* left factor bars */}
          <div className="flex flex-col gap-4 w-52">
            {factorBars.map((f) => (
              <div key={f.label} className="space-y-1">
                <div className="flex justify-between text-xs text-slate-300">
                  <span>{f.label}</span>
                  <span>{Math.round(clamp01(f.value) * 100)}%</span>
                </div>
                <div className="h-2 w-full rounded-full bg-slate-900/70 overflow-hidden relative">
                  <div
                    className={`h-full ${f.color} transition-all duration-500 ease-out relative`}
                    style={{ 
                      width: `${clamp01(f.value) * 100}%`,
                      animation: 'barShimmer 3s ease-in-out infinite',
                    }}
                  >
                    {/* Shimmer effect */}
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer" />
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* central glass panel, narrower + closer to render */}
          <div className="relative flex-1 max-w-3xl rounded-[32px] border border-cyan-500/30 bg-slate-950/85 backdrop-blur-2xl shadow-[0_0_60px_rgba(56,189,248,0.5)] px-10 py-8 flex flex-col gap-6 transition-all duration-300 hover:shadow-[0_0_80px_rgba(56,189,248,0.6)] hover:border-cyan-500/40 hover:-translate-y-1 group">
            {/* Holographic grid glow behind card - more visible */}
            <div 
              className="absolute inset-0 rounded-[32px] pointer-events-none -z-10"
              style={{
                background: 'radial-gradient(circle at center, rgba(56, 189, 248, 0.15) 0%, rgba(56, 189, 248, 0.05) 50%, transparent 80%)',
                filter: 'blur(60px)',
                animation: 'glowPulse 4s ease-in-out infinite',
              }}
            />
            {/* Corner accents - more visible */}
            <div className="absolute top-0 left-0 w-10 h-10 border-t-2 border-l-2 border-cyan-400/70 rounded-tl-[32px] shadow-[0_0_10px_rgba(56,189,248,0.5)]" />
            <div className="absolute top-0 right-0 w-10 h-10 border-t-2 border-r-2 border-cyan-400/70 rounded-tr-[32px] shadow-[0_0_10px_rgba(56,189,248,0.5)]" />
            <div className="absolute bottom-0 left-0 w-10 h-10 border-b-2 border-l-2 border-cyan-400/70 rounded-bl-[32px] shadow-[0_0_10px_rgba(56,189,248,0.5)]" />
            <div className="absolute bottom-0 right-0 w-10 h-10 border-b-2 border-r-2 border-cyan-400/70 rounded-br-[32px] shadow-[0_0_10px_rgba(56,189,248,0.5)]" />
            
            {/* Vertical divider light bars - more visible */}
            <div className="absolute left-1/3 top-[15%] bottom-[15%] w-[1px] bg-gradient-to-b from-transparent via-cyan-400/40 to-transparent" />
            <div className="absolute right-1/3 top-[15%] bottom-[15%] w-[1px] bg-gradient-to-b from-transparent via-cyan-400/40 to-transparent" />
            {/* header row */}
            <div className="flex justify-between items-baseline mb-1">
              <h2 className="text-sm font-bold tracking-wider text-slate-100 uppercase">
                EDON Live Inference
              </h2>
            </div>

            {/* State / CAV / Risk row */}
            <div className="grid grid-cols-3 gap-6 text-xs">
              <div className="space-y-1">
                <div className="text-slate-400 text-xs uppercase tracking-wider">State</div>
                <div className="text-base font-bold text-slate-50 transition-colors duration-300">
                  {data.state}
                </div>
              </div>
              <div className="space-y-1">
                <div className="text-slate-400 text-xs uppercase tracking-wider">CAV Score</div>
                <div className="text-base font-bold text-cyan-300 transition-all duration-300">
                  {Math.round(data.cavScore).toLocaleString()}
                </div>
              </div>
              <div className="space-y-1 text-right">
                <div className="text-slate-400 text-xs uppercase tracking-wider">Risk Score</div>
                <div className="text-base font-bold text-emerald-300 transition-colors duration-300">
                  {data.riskScore.toFixed(2)} ({data.riskLabel})
                </div>
              </div>
            </div>

            {/* main content inside card: drift + gauge */}
            <div className="grid grid-cols-[1.1fr_0.9fr] gap-8 items-center">
              {/* Drift chart */}
              <div className="h-40">
                <div className="flex justify-between items-baseline mb-1">
                  <span className="text-xs text-slate-400">
                    Drift over Time
                  </span>
                  <span className="text-[10px] text-slate-500">
                    Drift Index: {data.driftIndex.toFixed(3)}
                  </span>
                </div>
                <LineChart
                  width={320}
                  height={130}
                  data={data.driftSeries}
                  margin={{ top: 10, left: -15, right: 10, bottom: 0 }}
                >
                  <XAxis dataKey="t" hide />
                  <YAxis hide domain={["auto", "auto"]} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#020617",
                      border: "1px solid #1e293b",
                      borderRadius: 8,
                    }}
                    labelStyle={{ fontSize: 10 }}
                    itemStyle={{ fontSize: 10 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#22d3ee"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </div>

              {/* CAV gauge */}
              <div className="relative flex flex-col items-center justify-center">
                {/* Rotating light arc - more visible */}
                <div 
                  className="absolute w-[240px] h-[240px] rounded-full"
                  style={{
                    background: 'conic-gradient(from 0deg, transparent 0%, rgba(56, 189, 248, 0.2) 40%, rgba(56, 189, 248, 0.3) 45%, rgba(56, 189, 248, 0.2) 50%, transparent 60%, transparent 100%)',
                    animation: 'rotateArc 20s linear infinite',
                    filter: 'blur(10px)',
                    opacity: 0.6,
                  }}
                />
                
                {/* Live signal waveform behind score - more visible */}
                <div className="absolute inset-0 flex items-center justify-center opacity-30">
                  <svg width="200" height="60" className="absolute">
                    <defs>
                      <linearGradient id="waveGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.8" />
                        <stop offset="50%" stopColor="#06b6d4" stopOpacity="0.6" />
                        <stop offset="100%" stopColor="#a855f7" stopOpacity="0.8" />
                      </linearGradient>
                    </defs>
                    <path
                      d="M 0 30 Q 25 20, 50 30 T 100 30 T 150 30 T 200 30"
                      stroke="url(#waveGradient)"
                      strokeWidth="3"
                      fill="none"
                      className="animate-wave"
                    />
                  </svg>
                </div>
                
                <RadialBarChart
                  width={220}
                  height={220}
                  cx="50%"
                  cy="50%"
                  innerRadius="70%"
                  outerRadius="100%"
                  barSize={12}
                  data={cavGaugeData}
                  startAngle={220}
                  endAngle={-40}
                >
                  <PolarAngleAxis
                    type="number"
                    domain={[0, 10000]}
                    dataKey="value"
                    tick={false}
                  />
                  <RadialBar
                    background
                    clockWise
                    dataKey="value"
                    cornerRadius={999}
                  />
                </RadialBarChart>
                <div className="absolute text-center">
                  <div 
                    className="text-2xl font-bold text-slate-50 transition-all duration-300 drop-shadow-[0_0_8px_rgba(56,189,248,0.5)]"
                    style={{
                      animation: shouldPulse ? 'scorePulse 0.6s ease-out' : 'none',
                    }}
                  >
                    {Math.round(data.cavScore).toLocaleString()}
                  </div>
                  <div className="text-[10px] tracking-wide text-slate-400 mt-1">
                    CAV Score (0 – 10,000)
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* little right spacer so composition matches the image */}
          <div className="w-4" />
        </div>
      </div>
    </div>
  );
}
