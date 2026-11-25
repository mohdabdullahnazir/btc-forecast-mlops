"""Plot real Kubernetes observations collected by scripts/sample_hpa.ps1."""

import argparse
import csv
from datetime import datetime

import matplotlib.pyplot as plt


def chart(source, destination):
    with open(source, encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["cpu_pct"]]
    if not rows:
        raise ValueError("No CPU observations recorded. Check Metrics Server.")
    times = [datetime.fromisoformat(row["timestamp"]) for row in rows]
    minutes = [(time - times[0]).total_seconds() / 60 for time in times]
    figure, (cpu, pods) = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    cpu.plot(minutes, [int(row["cpu_pct"]) for row in rows], color="#126e70")
    cpu.axhline(60, color="#a45720", linestyle="--", label="60% HPA target")
    cpu.set_ylabel("CPU (% of request)")
    cpu.legend()
    for key, label in (("replicas", "Running"), ("desired", "Desired")):
        pods.step(minutes, [int(row[key]) for row in rows], where="post", label=label)
    pods.legend()
    pods.set_ylabel("Replicas")
    pods.set_xlabel("Minutes from first CPU observation")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    chart(args.source, args.destination)
