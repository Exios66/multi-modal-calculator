import math
import csv

# Total duration in seconds (20 minutes)
total_seconds = 1200
# Baseline phase duration (5 minutes = 300 seconds)
baseline_seconds = 300

with open("session.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Write header
    writer.writerow([
        "Time(s)",
        "Alpha Power (μV²)",
        "Beta Power (μV²)",
        "Theta Power (μV²)",
        "Theta/Beta Ratio",
        "Theta/Alpha Ratio"
    ])
    
    # Loop over each 1s epoch
    for t in range(total_seconds):
        if t < baseline_seconds:
            # Baseline phase: relatively constant values with slight oscillatory variations
            alpha = 50 + 2 * math.sin(2 * math.pi * t / 30)
            beta  = 15 + 0.5 * math.sin(2 * math.pi * t / 45 + 1)
            theta = 10 + 1 * math.sin(2 * math.pi * t / 60 + 2)
        else:
            # Cognitive load phase: linear trend plus similar oscillations.
            # p goes from 0 to 1 over the 15 minutes.
            p = (t - baseline_seconds) / (total_seconds - baseline_seconds)
            alpha = (50 - 20 * p) + 2 * math.sin(2 * math.pi * t / 30)
            beta  = (15 + 15 * p) + 0.5 * math.sin(2 * math.pi * t / 45 + 1)
            theta = (10 + 15 * p) + 1 * math.sin(2 * math.pi * t / 60 + 2)
        
        theta_beta_ratio = theta / beta
        theta_alpha_ratio = theta / alpha
        
        writer.writerow([
            t,
            f"{alpha:.2f}",
            f"{beta:.2f}",
            f"{theta:.2f}",
            f"{theta_beta_ratio:.2f}",
            f"{theta_alpha_ratio:.2f}"
        ])