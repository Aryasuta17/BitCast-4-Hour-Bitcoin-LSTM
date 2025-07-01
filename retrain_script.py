from model_utils import fetch_new_4h_data, retrain_model

if __name__ == "__main__":
    print("==== Mulai Fetch Data ====")
    fetch_new_4h_data()
    print("==== Fetch Selesai, Mulai Retrain ====")
    retrain_model()
    print("==== Retrain SELESAI! ====")
