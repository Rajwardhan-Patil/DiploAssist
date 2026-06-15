# TODO - Support updated huge_colleges.csv (new schema)

- [ ] Update `train_model.py` to match new CSV columns (`college_name, branch, category_code, percentile, ...`).
- [ ] Update `app.py /predict` input mapping: user fields (`caste, branch, gender, quota`) must map to new dataset’s `category_code` (or change UI contract).
- [x] Train model now supports both old and new CSV schemas (see `train_model.py`).
- [ ] Update `app.py /predict` to accept dataset `caste` codes if needed (unknown labels currently throw 400).

- [ ] Retrain: run `python train_model.py`.
- [ ] Smoke test: start server `python app.py` and call `/predict` with existing UI values; confirm top results include colleges from the new CSV.

