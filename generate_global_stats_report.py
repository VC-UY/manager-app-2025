import csv
import glob
import json
import matplotlib.pyplot as plt
from pathlib import Path

path = Path('distributed_learning/results')
path.mkdir(parents=True, exist_ok=True)
with open(path / 'global_stats.json') as f:
    data = json.load(f)

# Load per-volunteer full files if present
vol_files = sorted(glob.glob(str(path / 'volunteer_*.json')))
vol_details = {}
for vf in vol_files:
    try:
        with open(vf) as f:
            j = json.load(f)
            vol_details[j.get('volunteer_ip', vf)] = j
    except Exception:
        pass

vols = data.get('volunteer_summaries', {})
ids = list(vols.keys())
accuracies = [vols[v]['final_test_acc'] for v in ids]
best_acc = [vols[v]['best_test_acc'] for v in ids]
bytes_sent = [vols[v]['total_bytes_sent'] for v in ids]
bytes_received = [vols[v]['total_bytes_received'] for v in ids]
durations = [vols[v]['total_train_duration_s'] for v in ids]
compression = [vols[v]['avg_compression_ratio'] for v in ids]

# Accuracy plot
plt.figure(figsize=(8,5))
bar_width = 0.35
x = list(range(len(ids)))
plt.bar([i - bar_width/2 for i in x], best_acc, width=bar_width, label='Best test accuracy')
plt.bar([i + bar_width/2 for i in x], accuracies, width=bar_width, label='Final test accuracy')
plt.xticks(x, ids, rotation=30)
plt.ylim(0, 1.0)
plt.ylabel('Accuracy')
plt.title('Volunteer test accuracies')
plt.legend()
plt.tight_layout()
plt.savefig(path / 'global_stats_accuracy.png')
plt.close()

# Traffic plot
plt.figure(figsize=(8,5))
plt.bar([i - bar_width/2 for i in x], [b / 1e6 for b in bytes_sent], width=bar_width, label='Bytes sent (MB)')
plt.bar([i + bar_width/2 for i in x], [b / 1e6 for b in bytes_received], width=bar_width, label='Bytes received (MB)')
plt.xticks(x, ids, rotation=30)
plt.ylabel('Traffic (MB)')
plt.title('Volunteer network traffic')
plt.legend()
plt.tight_layout()
plt.savefig(path / 'global_stats_traffic.png')
plt.close()

# Duration + compression
fig, ax1 = plt.subplots(figsize=(8,5))
ax2 = ax1.twinx()
ax1.bar(x, [d / 3600 for d in durations], width=bar_width, color='C0', label='Training duration (h)')
ax2.plot(x, compression, color='C1', marker='o', label='Compression ratio')
ax1.set_xticks(x)
ax1.set_xticklabels(ids, rotation=30)
ax1.set_ylabel('Training duration (hours)')
ax2.set_ylabel('Avg compression ratio')
ax1.set_title('Volunteer training duration and compression ratio')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig(path / 'global_stats_duration_compression.png')
plt.close()

report = []
report.append('# Rapport des résultats globaux')
report.append('\n## Résumé global\n')
report.append(f"- Durée totale d’exécution : {data['runtime_s']:.1f} s")
report.append(f"- Nombre de volontaires actifs : {data['n_active_volunteers']}")
report.append(f"- Échanges de modèles : {data['total_model_exchanges']}")
report.append(f"- Total des octets routés : {data['total_bytes_routed']:,}")
report.append(f"- Débit global : {data['throughput_KB_per_s']:.3f} KB/s\n")

report.append('## Tableau des volontaires\n')
report.append('| Volontaire | Rounds | Meilleure acc. test | Acc. test finale | Bytes envoyés | Bytes reçus | Durée train (h) | Comp. ratio |')
report.append('|---|---|---|---|---|---|---|---|')
for v in ids:
    vol = vols[v]
    report.append(f"| {v} | {vol.get('total_rounds',0)} | {vol.get('best_test_acc',0):.4f} | {vol.get('final_test_acc',0):.4f} | {vol.get('total_bytes_sent',0):,} | {vol.get('total_bytes_received',0):,} | {vol.get('total_train_duration_s',0)/3600:.2f} | {vol.get('avg_compression_ratio',0):.2f} |")

# Per-volunteer round traces (if volunteer files exist)
report.append('\n## Détails par volontaire et par round\n')
for ip, vf in vol_details.items():
    report.append(f"### Volontaire {ip}")
    report.append(f"- Total rounds: {vf.get('total_rounds',0)}")
    report.append(f"- Total bytes sent: {vf.get('total_bytes_sent',0):,}")
    report.append(f"- Total bytes received: {vf.get('total_bytes_received',0):,}\n")
    report.append('| Round | Duration (s) | Test acc | Best acc so far | Best acc ts | #sent | #recv |')
    report.append('|---|---:|---:|---:|---:|---:|---:|')
    for r in vf.get('rounds', []):
        rd = r.get('round_num')
        dur = r.get('round_duration_s') or (r.get('round_end_ts',0)-r.get('round_start_ts',0))
        tacc = r.get('test_acc',0)
        best = r.get('best_test_acc_so_far',0)
        best_ts = r.get('best_test_acc_ts',0)
        n_sent = len(r.get('sent_details',[]))
        n_recv = len(r.get('recv_details',[]))
        report.append(f"| {rd} | {dur:.1f} | {tacc:.4f} | {best:.4f} | {best_ts:.1f} | {n_sent} | {n_recv} |")
    report.append('\n')

# Exchanges detailed from manager
report.append('## Historique des échanges (manager)\n')
for ex in data.get('exchanges', []):
    s = ex.get('sender')
    r = ex.get('receiver')
    b = ex.get('bytes',0)
    q = ex.get('queued_ts')
    d = ex.get('delivered_ts')
    tt = ex.get('transfer_time_s')
    report.append(f"- {s} → {r} : {b/1024:.1f} KB queued={q} delivered={d} transfer_time_s={tt}")

# Write report
with open(path / 'global_stats_report.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

# Export CSV global summary
csv_summary_path = path / 'global_stats_volunteers.csv'
with open(csv_summary_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'volunteer_ip', 'total_rounds', 'best_test_acc', 'final_test_acc',
        'total_bytes_sent', 'total_bytes_received', 'total_train_duration_s',
        'avg_compression_ratio'
    ])
    for v in ids:
        vol = vols[v]
        writer.writerow([
            v,
            vol.get('total_rounds', 0),
            vol.get('best_test_acc', 0),
            vol.get('final_test_acc', 0),
            vol.get('total_bytes_sent', 0),
            vol.get('total_bytes_received', 0),
            vol.get('total_train_duration_s', 0),
            vol.get('avg_compression_ratio', 0),
        ])

# Export CSV round-level details if available
csv_rounds_path = path / 'global_stats_rounds.csv'
with open(csv_rounds_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'volunteer_ip', 'round_num', 'round_duration_s', 'test_acc',
        'best_test_acc_so_far', 'best_test_acc_ts', 'n_sent', 'n_recv'
    ])
    for ip, vf in vol_details.items():
        for r in vf.get('rounds', []):
            dur = r.get('round_duration_s') or (r.get('round_end_ts', 0) - r.get('round_start_ts', 0))
            writer.writerow([
                ip,
                r.get('round_num', ''),
                dur,
                r.get('test_acc', ''),
                r.get('best_test_acc_so_far', ''),
                r.get('best_test_acc_ts', ''),
                len(r.get('sent_details', [])),
                len(r.get('recv_details', [])),
            ])

# Export CSV exchanges
csv_exchanges_path = path / 'global_stats_exchanges.csv'
with open(csv_exchanges_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'sender', 'receiver', 'bytes', 'queued_ts', 'delivered_ts',
        'send_ts_start', 'send_duration_s', 'transfer_time_s'
    ])
    for ex in data.get('exchanges', []):
        writer.writerow([
            ex.get('sender', ''),
            ex.get('receiver', ''),
            ex.get('bytes', ''),
            ex.get('queued_ts', ''),
            ex.get('delivered_ts', ''),
            ex.get('send_ts_start', ''),
            ex.get('send_duration_s', ''),
            ex.get('transfer_time_s', ''),
        ])

print('Generated report, charts, and CSV files in', path)
