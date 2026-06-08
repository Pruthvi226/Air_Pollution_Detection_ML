# Limitations and Future Scope

## Limitations

- The bundled smoke model is intentionally small so the dashboard and API can run after clone or deployment.
- Final portfolio claims should use the stronger quarter-hourly training run when available.
- AQI-style categories are demo risk labels, not certified regulatory AQI calculations or medical advice.
- Predictions depend on historical sensor quality, missing values, station coverage, and the distribution of the training period.
- Raw DCR archives and full processed datasets are not committed because they are large and should be regenerated locally or in Colab.

## Future Scope

- Add scheduled ingestion from live monitoring feeds.
- Add SHAP summary plots when deployment size allows optional SHAP installation.
- Add model registry versioning for final hourly and quarter-hourly artifacts.
- Add real-time anomaly alert delivery through email, Slack, or an operations dashboard.
- Add a PyTorch sequence model for comparison with the tabular ensemble workflow.
