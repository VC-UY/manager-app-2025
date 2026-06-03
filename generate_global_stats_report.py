import json
import matplotlib.pyplot as plt
from pathlib import Path

path = Path('distributed_learning/results')
path.mkdir(parents=True, exist_ok=True)
with open(path / 'global_stats.json') as f:
    data = json.load(f)
vols = data['volunteer_summaries']
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
plt.ylim(0, 0.6)
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

report = """# Rapport des résultats globaux

## Résumé global
"""
report += f"- Durée totale d’exécution : {data['runtime_s']:.1f} s\n"
report += f"- Nombre de volontaires actifs : {data['n_active_volunteers']}\n"
report += f"- Échanges de modèles : {data['total_model_exchanges']}\n"
report += f"- Total des octets routés : {data['total_bytes_routed']:,}\n"
report += f"- Débit global : {data['throughput_KB_per_s']:.3f} KB/s\n\n"
report += "## Tableau des volontaires\n\n"
report += "| Volontaire | Rounds | Meilleure acc. test | Acc. test finale | Bytes envoyés | Bytes reçus | Durée train (h) | Comp. ratio |\n"
report += "|---|---|---|---|---|---|---|---|\n"
for v in ids:
    vol = vols[v]
    report += f"| {v} | {vol['total_rounds']} | {vol['best_test_acc']:.4f} | {vol['final_test_acc']:.4f} | {vol['total_bytes_sent']:,} | {vol['total_bytes_received']:,} | {vol['total_train_duration_s']/3600:.2f} | {vol['avg_compression_ratio']:.2f} |\n"
report += "\n## Graphiques générés\n\n"
report += "- `global_stats_accuracy.png` : comparatif des meilleures et finales précisions de test.\n"
report += "- `global_stats_traffic.png` : comparaison des volumes de données envoyées et reçues.\n"
report += "- `global_stats_duration_compression.png` : durée d’entraînement et ratio de compression par volontaire.\n\n"
report += "## Interprétation scientifique\n\n"
report += "Les performances observées sont faibles (20-39% de précision finale). Cela signifie que le modèle ne généralise pas bien sur les données de test, probablement à cause d’une combinaison de :\n"
report += "- partition de données non-iid ou déséquilibrée entre volontaires ;\n"
report += "- peu de rounds pour certains volontaires ;\n"
report += "- compression trop élevée / mauvaise communication des gradients.\n\n"
report += "### Points clés\n"
report += "- `192.168.1.106` a réalisé 395 rounds, mais sa précision finale reste basse (21,14%). Ce comportement est typique d’un entraînement instable ou de données locales très bruyantes.\n"
report += "- `192.168.1.109` (36 rounds) a aussi une faible précision finale (18,74%). Le nombre de rounds est faible, donc son impact est limité.\n"
report += "- `192.168.1.131/24` a la meilleure précision finale (39,46%) et un meilleur équilibre entre meilleur et dernier score, mais il n’a rien envoyé (`total_bytes_sent`=0), ce qui suggère un rôle de réception ou un problème de configuration de l’échange.\n\n"
report += "### Interprétation des mesures par volontaire\n"
report += "- `current_round` / `total_rounds` : représente le nombre d’itérations locales réalisées. Un faible nombre de rounds empêche la convergence.\n"
report += "- `best_test_acc` : le meilleur score observé durant l’entraînement. Si cette valeur est bien supérieure à `final_test_acc`, le modèle a pu sur-apprendre puis se dégrader.\n"
report += "- `final_test_acc` : précision finale du modèle local au moment de l’arrêt. C’est l’indicateur de performance le plus important.\n"
report += "- `total_bytes_sent` / `total_bytes_received` : volume de mise à jour échangée. Un énorme déséquilibre indique une architecture d’échange inégale ou un volontaire qui reçoit beaucoup sans envoyer.\n"
report += "- `total_train_duration_s` : temps total d’entraînement. Un temps très élevé pour peu de progrès signifie un entraînement inefficace.\n"
report += "- `avg_compression_ratio` : compression moyenne appliquée aux mises à jour. Une compression élevée réduit le trafic mais peut dégrader les performances du modèle.\n"

with open(path / 'global_stats_report.md', 'w') as f:
    f.write(report)

print('Generated report and charts in', path)
