"""The public ReFT helper owns Trainer configuration and checkpoint saving."""

from types import SimpleNamespace

from easysteer.reft import train


def test_training_helper_passes_explicit_experiment_settings(monkeypatch, tmp_path):
    events = []
    model = SimpleNamespace(config=SimpleNamespace(hidden_size=8))
    tokenizer = object()
    wrapper = SimpleNamespace(
        set_device=lambda device: events.append(("device", device)),
        print_trainable_parameters=lambda: None,
        save=lambda **kwargs: events.append(("save", kwargs)),
    )
    monkeypatch.setattr(
        train, "load_model_and_tokenizer", lambda *args: (model, tokenizer)
    )
    monkeypatch.setattr(train, "_build_intervention", lambda *args: object())
    monkeypatch.setattr(train.pyreft, "ReftConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(train.pyreft, "get_reft_model", lambda *args: wrapper)

    def make_data(tok, base, prompts, responses):
        assert tok is tokenizer and base is model
        assert prompts == ["Prompt: hello"] and responses == ["world"]
        return {"train_dataset": [1], "data_collator": object()}

    class Trainer:
        def __init__(self, **kwargs):
            events.append(("trainer", kwargs))

        def add_callback(self, callback):
            events.append(("callback", callback))

        def train(self):
            events.append(("train", None))

    monkeypatch.setattr(
        train.pyreft, "make_last_position_supervised_data_module", make_data
    )
    monkeypatch.setattr(train.pyreft, "ReftTrainerForCausalLM", Trainer)
    monkeypatch.setattr(
        train.transformers, "TrainingArguments", lambda **kwargs: kwargs
    )
    callback = object()
    result = train.train_reft(
        "model",
        [["hello", "world"]],
        intervention="loreft",
        layer=8,
        low_rank_dimension=4,
        device="cpu",
        prompt_template="Prompt: %s",
        callbacks=[callback],
        save_dir=str(tmp_path),
        num_train_epochs=200.0,
        per_device_train_batch_size=10,
        learning_rate=4e-3,
    )
    assert result == (wrapper, tokenizer)
    kwargs = next(value for key, value in events if key == "trainer")
    assert kwargs["processing_class"] is tokenizer and "tokenizer" not in kwargs
    assert kwargs["args"]["num_train_epochs"] == 200.0
    assert kwargs["args"]["per_device_train_batch_size"] == 10
    assert kwargs["args"]["learning_rate"] == 4e-3
    assert events.index(("callback", callback)) < events.index(("train", None))
    assert events[-3:] == [
        ("device", "cpu"),
        ("save", {"save_directory": str(tmp_path), "save_to_hf_hub": False}),
        ("device", "cpu"),
    ]
