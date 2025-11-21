// SVG radial gauge for CAV score
"use client";

import React, { useEffect, useRef, useState } from "react";

interface CavGaugeProps {
  value: number; // 0-10000
  max?: number; // default 10000
  size?: number; // default 220
}

export default function CavGauge({ value, max = 10000, size = 220 }: CavGaugeProps) {
  const [animatedValue, setAnimatedValue] = useState(value);
  const [displayValue, setDisplayValue] = useState(value);
  const animationRef = useRef<number>();

  // Smooth animation when value changes
  useEffect(() => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }

    const startValue = animatedValue;
    const endValue = value;
    const duration = 800; // ms
    const startTime = performance.now();

    const animate = (currentTime: number) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      // Ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = startValue + (endValue - startValue) * eased;
      
      setAnimatedValue(current);
      setDisplayValue(Math.round(current));

      if (progress < 1) {
        animationRef.current = requestAnimationFrame(animate);
      } else {
        setAnimatedValue(endValue);
        setDisplayValue(Math.round(endValue));
      }
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [value]);

  const normalized = Math.min(Math.max(animatedValue / max, 0), 1);
  const radius = size / 2 - 20;
  const center = size / 2;
  const circumference = 2 * Math.PI * radius;
  
  // Arc from 220deg to -40deg (240deg = 2/3 of full circle)
  // We want to draw 240 degrees, which is 2/3 of 360
  const arcLength = circumference * (240 / 360); // 2/3 of full circle
  const offset = arcLength * (1 - normalized);

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        className="transform -rotate-90"
      >
        <defs>
          <linearGradient id="cavGaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#22d3ee" />
            <stop offset="85%" stopColor="#22d3ee" />
            <stop offset="100%" stopColor="#ec4899" />
          </linearGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        
        {/* Background track - 240deg arc starting at 220deg */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke="rgba(15, 23, 42, 0.5)"
          strokeWidth="12"
          strokeLinecap="round"
          style={{
            strokeDasharray: `${arcLength} ${circumference}`,
            strokeDashoffset: circumference * (220 / 360), // Start at 220deg
          }}
        />
        
        {/* Active arc with gradient */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke="url(#cavGaugeGradient)"
          strokeWidth="12"
          strokeLinecap="round"
          filter="url(#glow)"
          style={{
            strokeDasharray: `${arcLength} ${circumference}`,
            strokeDashoffset: circumference * (220 / 360) + offset,
            transition: "stroke-dashoffset 0.3s ease-out",
          }}
        />
      </svg>
      
      {/* Center text */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div className="text-center">
          <div className="text-2xl font-bold text-slate-900 dark:text-slate-50">
            {displayValue.toLocaleString()}
          </div>
          <div className="text-[10px] tracking-wide text-slate-500 dark:text-slate-400 mt-1">
            CAV Score (0 – 10,000)
          </div>
        </div>
      </div>
    </div>
  );
}

