// Animated drift waveform
"use client";

import React, { useEffect, useRef, useState } from "react";

interface DriftWaveProps {
  data: Array<{ t: number; value: number }>;
  width?: number;
  height?: number;
}

export default function DriftWave({ data, width = 320, height = 130 }: DriftWaveProps) {
  const [pathData, setPathData] = useState("");
  const [offset, setOffset] = useState(0);
  const animationRef = useRef<number>();

  // Generate smooth path from data
  useEffect(() => {
    if (data.length === 0) return;

    const padding = 20;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;
    const xStep = chartWidth / Math.max(data.length - 1, 1);
    
    // Normalize values to -10 to 10 range, then map to chart height
    const minValue = -10;
    const maxValue = 10;
    const valueRange = maxValue - minValue;
    
    const points = data.map((point, i) => {
      const x = padding + i * xStep;
      // Map value from [-10, 10] to [height - padding, padding]
      const normalized = (point.value - minValue) / valueRange;
      const y = padding + (1 - normalized) * chartHeight;
      return { x, y };
    });

    // Create smooth path using quadratic curves
    let path = `M ${points[0].x} ${points[0].y}`;
    for (let i = 1; i < points.length; i++) {
      const prev = points[i - 1];
      const curr = points[i];
      const controlX = (prev.x + curr.x) / 2;
      path += ` Q ${controlX} ${prev.y}, ${curr.x} ${curr.y}`;
    }

    setPathData(path);
  }, [data, width, height]);

  // Animate horizontal drift
  useEffect(() => {
    const animate = () => {
      setOffset((prev) => (prev + 0.5) % 100);
      animationRef.current = requestAnimationFrame(animate);
    };
    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  return (
    <div className="relative" style={{ width, height }}>
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        className="overflow-visible"
      >
        <defs>
          <linearGradient id="driftGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.3" />
            <stop offset="100%" stopColor="#22d3ee" stopOpacity="0.05" />
          </linearGradient>
          <linearGradient id="driftStroke" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.8" />
            <stop offset="50%" stopColor="#06b6d4" stopOpacity="0.6" />
            <stop offset="100%" stopColor="#a855f7" stopOpacity="0.8" />
          </linearGradient>
        </defs>
        
        {/* Grid lines */}
        <g opacity="0.15">
          <line
            x1={20}
            y1={height / 2}
            x2={width - 20}
            y2={height / 2}
            stroke="currentColor"
            strokeWidth="1"
            className="text-slate-400"
          />
        </g>
        
        {/* Filled area under curve */}
        {pathData && (
          <path
            d={`${pathData} L ${width - 20} ${height - 20} L 20 ${height - 20} Z`}
            fill="url(#driftGradient)"
            style={{
              transform: `translateX(${offset * 0.1}px)`,
              transition: "transform 0.1s linear",
            }}
          />
        )}
        
        {/* Main waveform */}
        {pathData && (
          <path
            d={pathData}
            fill="none"
            stroke="url(#driftStroke)"
            strokeWidth="2"
            strokeLinecap="round"
            style={{
              transform: `translateX(${offset * 0.1}px)`,
              transition: "transform 0.1s linear",
            }}
          />
        )}
      </svg>
    </div>
  );
}

