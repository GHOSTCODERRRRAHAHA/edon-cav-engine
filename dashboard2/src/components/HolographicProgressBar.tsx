// Holographic progress bar with neon glow and shimmer
"use client";

import React from "react";

interface HolographicProgressBarProps {
  label: string;
  value: number; // 0-1
  color?: "cyan" | "pink";
}

export default function HolographicProgressBar({
  label,
  value,
  color = "cyan",
}: HolographicProgressBarProps) {
  const clampedValue = Math.min(Math.max(value, 0), 1);
  const percentage = Math.round(clampedValue * 100);

  const gradientColors =
    color === "cyan"
      ? "from-cyan-400 via-teal-400 to-cyan-500"
      : "from-pink-400 via-rose-400 to-pink-500";

  return (
    <div className="space-y-2">
      <div className="flex justify-between text-sm text-slate-700 dark:text-slate-300">
        <span className="font-medium">{label}</span>
        <span className="font-semibold">{percentage}%</span>
      </div>
      <div className="h-3 w-full rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden relative">
        {/* Filled portion with gradient */}
        <div
          className={`h-full bg-gradient-to-r ${gradientColors} transition-all duration-500 ease-out rounded-full`}
          style={{
            width: `${clampedValue * 100}%`,
          }}
        />
      </div>
    </div>
  );
}

