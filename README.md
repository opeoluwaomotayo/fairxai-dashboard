# FairXAI Responsible AI Dashboard

An interactive **Streamlit** dashboard combining predictive-performance evaluation, group-fairness auditing, and SHAP-based local explainability on reproducible synthetic socio-economic data.

## Dashboard sections

- **Overview:** Accuracy, precision, recall and F1 comparison.
- **Fairness Audit:** Demographic parity difference, equal opportunity difference and equalized-odds gap.
- **XAI Explanation:** Interactive input controls and ranked SHAP contributions.
- **Methodology:** Responsible interpretation and limitations.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit will display a local URL in the terminal, normally `http://localhost:8501`.

## Run tests

```bash
python -m pytest -v
```

The included GitHub Actions workflow uses the same command.

## Deploy from GitHub

After publishing the repository, you can connect it to a compatible Streamlit hosting environment and use `app.py` as the application entry point. Keep the repository files together because the app generates its reproducible teaching data/model when needed.

## Important research warning

The dataset is synthetic and intentionally embeds a group-related historical effect to demonstrate auditing. The dashboard is **not** suitable for decisions about real people, and its numerical outcomes are not evidence of fairness or performance in any operational setting.

Removing the sensitive group variable is presented only as an auditable comparison, not a fairness guarantee. Proxy variables, measurement choices, legal obligations, stakeholder harms and incompatible fairness definitions still require domain-specific analysis.

## Suggested research extensions

- Use an ethically and legally appropriate real dataset with documented provenance.
- Add confidence intervals and subgroup sample sizes.
- Compare multiple mitigation strategies rather than attribute removal alone.
- Add global SHAP explanations and explanation-stability analysis.
- Add model cards and dataset documentation.
- Add downloadable audit reports after validation.

## Author

**Dr. Opeoluwa Omotayo Ajilore**  
Lecturer, Computer Science

## License

MIT License. See `LICENSE`.
