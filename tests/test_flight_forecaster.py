from model_pipeline import generate_synthetic_dataset, TravelDealForecaster


def test_training_pipeline_runs():
    df = generate_synthetic_dataset(300)
    model = TravelDealForecaster()
    metrics = model.train_with_time_split(df)

    assert metrics["test_samples"] > 0
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["mae"] >= 0.0

    prediction = model.predict(df.head(1))
    assert set(["trend", "confidence", "predicted_future_price"]).issubset(prediction)
    assert 0.0 <= prediction["confidence"] <= 100.0
    assert prediction["predicted_future_price"] >= 0.0
