from data_stream import electricity_stream
from ilstm_model import build_ilstm_model
from ilstm_trainer import run_incremental_training


def main():
    CSV_PATH = "data/electricity.csv"

    SEQ_LENGTH = 48
    INITIAL_BATCH = 960
    STREAM_BATCH = 336
    INIT_EPOCHS = 300
    STREAM_EPOCHS = 60
    BATCH_SIZE = 1
    N_FEATURES = 6

    stream = electricity_stream(CSV_PATH)

    model = build_ilstm_model(
        batch_size=BATCH_SIZE,
        seq_length=SEQ_LENGTH,
        n_features=N_FEATURES,
    )

    metrics = run_incremental_training(
        model=model,
        data_stream=stream,
        sequence_length=SEQ_LENGTH,
        initial_batch_size=INITIAL_BATCH,
        stream_batch_size=STREAM_BATCH,
        initial_epochs=INIT_EPOCHS,
        stream_epochs=STREAM_EPOCHS,
    )

    print("\nFINAL PREQUENTIAL PERFORMANCE (Electricity)")
    print(f"Accuracy : {metrics['accuracy'][0]*100:.2f} ± {metrics['accuracy'][1]*100:.2f}")
    print(f"Precision: {metrics['precision'][0]*100:.2f} ± {metrics['precision'][1]*100:.2f}")
    print(f"Recall   : {metrics['recall'][0]*100:.2f} ± {metrics['recall'][1]*100:.2f}")
    print(f"Loss     : {metrics['loss'][0]:.4f} ± {metrics['loss'][1]:.4f}")


if __name__ == "__main__":
    main()
