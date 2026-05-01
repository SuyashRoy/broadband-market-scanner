interface ProgressBarProps {
  value: number; /* 0-1 */
  color?: string;
  className?: string;
}

export function ProgressBar({ value, color = "bg-accent", className = "" }: ProgressBarProps) {
  const percentage = Math.min(100, Math.max(0, value * 100));

  return (
    <div className={`w-full h-1 bg-muted rounded-full overflow-hidden ${className}`}>
      <div
        className={`h-full ${color} transition-smooth`}
        style={{ width: `${percentage}%` }}
      />
    </div>
  );
}
