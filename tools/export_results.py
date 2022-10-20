from ctypes.wintypes import PCHAR
import pickle


def main():
    with open("results.pkl", "rb") as f:
        results = pickle.load(f)

    new_results = []
    for result in results:
        boxes_3d = result["boxes_3d"]
        scores_3d = result["scores_3d"]
        labels_3d = result["labels_3d"]
        new_results.append({
            "boxes_3d": boxes_3d.tensor.numpy(),
            "scores_3d": scores_3d.numpy(),
            "labels_3d": labels_3d.numpy(),
        })

    with open("results_numpy.pkl", "wb") as f:
        pickle.dump(new_results, f)


if __name__ == "__main__":
    main()
