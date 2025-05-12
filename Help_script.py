import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# === 1. Učitaj klase iz CSV ===
class_dict_path = "data/class_dict.csv"
class_dict = pd.read_csv(class_dict_path)

green_classes = class_dict[class_dict["name"].isin(["agriculture_land", "rangeland", "forest_land"])]
GREEN_CLASSES = [(r, g, b) for r, g, b in zip(green_classes["r"], green_classes["g"], green_classes["b"])]
print("RGB vrijednosti zelenih površina:", GREEN_CLASSES)

# === 2. Učitavanje i segmentacija ground-truth maske ===
def load_and_segment_mask(mask_path):
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    binary_mask = np.zeros(mask.shape[:2], dtype=np.uint8)

    for green in GREEN_CLASSES:
        green_mask = np.all(mask == green, axis=-1)
        binary_mask[green_mask] = 1

    return binary_mask

# === 3. Evaluacija segmentacije ===
def evaluate_segmentation(predicted_mask, ground_truth_mask):
    predicted_mask = (predicted_mask > 0).astype(np.uint8)
    ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8)

    intersection = np.logical_and(predicted_mask, ground_truth_mask).sum()
    union = np.logical_or(predicted_mask, ground_truth_mask).sum()
    iou = intersection / union if union > 0 else 0

    print(f"IoU (Intersection over Union) score: {iou:.4f}")

    return iou

# === 4. HSV + LAB segmentacija ===
def improved_hsv_lab_segmentation(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_hsv_green = np.array([30, 40, 40])
    upper_hsv_green = np.array([90, 255, 255])
    mask_hsv_green = cv2.inRange(hsv, lower_hsv_green, upper_hsv_green)

    lower_hsv_dry = np.array([10, 30, 60])
    upper_hsv_dry = np.array([30, 255, 255])
    mask_hsv_dry = cv2.inRange(hsv, lower_hsv_dry, upper_hsv_dry)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lower_lab = np.array([20, 120, 20])
    upper_lab = np.array([255, 160, 140])
    mask_lab = cv2.inRange(lab, lower_lab, upper_lab)

    combined_mask = cv2.bitwise_or(mask_hsv_green, mask_hsv_dry)
    combined_mask = cv2.bitwise_or(combined_mask, mask_lab)

    return combined_mask

# === 5. Učitavanje parova slika i maski ===
def load_image_and_mask_paths(data_dir, sample_size=25):
    all_files = sorted(os.listdir(data_dir))

    image_dict = {f.split('_')[0]: os.path.join(data_dir, f)
                  for f in all_files if f.endswith("_sat.jpg")}

    mask_dict = {f.split('_')[0]: os.path.join(data_dir, f)
                 for f in all_files if f.endswith("_mask.png")}

    common_ids = list(set(image_dict.keys()) & set(mask_dict.keys()))
    print(f"Pronađeno {len(common_ids)} sparenih parova slika i maski.")

    if sample_size and sample_size < len(common_ids):
        selected_ids = []
        while len(selected_ids) < sample_size:
            candidate = random.choice(common_ids)
            if candidate not in selected_ids:
                selected_ids.append(candidate)
    else:
        selected_ids = common_ids

    image_list = [image_dict[i] for i in selected_ids]
    mask_list = [mask_dict[i] for i in selected_ids]

    return image_list, mask_list

# === 6. Izračun zelene površine ===
def calculate_green_area(mask, pixel_area_m2=0.25):
    green_pixels = np.sum(mask == 255)
    total_pixels = mask.size
    green_area_m2 = green_pixels * pixel_area_m2
    green_percent = 100 * green_pixels / total_pixels
    return green_area_m2, green_percent

# === 7. Batch evaluacija ===
def batch_evaluate(images, masks, csv_path="rezultati.csv"):
    total_iou = 0
    results = []
    skipped = []

    for i, (image_path, mask_path) in enumerate(zip(images, masks)):
        print(f"\nObrađujem sliku {i+1}/{len(images)}: {os.path.basename(image_path)}")

        segmented_mask = improved_hsv_lab_segmentation(image_path)
        ground_truth_mask = load_and_segment_mask(mask_path)
        iou_score = evaluate_segmentation(segmented_mask, ground_truth_mask)

        if iou_score == 0:
            print("-> IoU = 0, preskačem ovu sliku.")
            skipped.append(os.path.basename(image_path))
            continue

        total_iou += iou_score

        green_area, green_percent = calculate_green_area(segmented_mask)

        results.append({
            "image_name": os.path.basename(image_path),
            "iou": round(iou_score, 4),
            "green_area_m2": round(green_area, 2),
            "green_percent": round(green_percent, 2),
        })

        print(f"IoU: {iou_score:.4f} | Zelenilo: {green_area:.2f} m² ({green_percent:.2f}%)")

    avg_iou = total_iou / len(results) if results else 0
    print("\nProsječni IoU za dataset: {:.4f}".format(avg_iou))

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"\nRezultati spremljeni u: {csv_path}")

    if skipped:
        print(f"\nPreskočene slike (IoU = 0): {len(skipped)}")
        for name in skipped:
            print("-", name)

    return avg_iou, results

# === 8. Prikaz usporedbe ===
def show_comparison(image_path, predicted_mask, ground_truth_mask, title):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 4))
    plt.suptitle(title, fontsize=16)

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original")

    plt.subplot(1, 3, 2)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title("Tvoja segmentacija")

    plt.subplot(1, 3, 3)
    plt.imshow(ground_truth_mask)
    plt.title("Ground Truth maska")

    plt.tight_layout()
    plt.show()

# === 9. Glavni dio ===
if __name__ == "__main__":
    random.seed(42)

    data_dir = "data/train"
    image_list, mask_list = load_image_and_mask_paths(data_dir, sample_size=25)

    average_iou, results = batch_evaluate(image_list, mask_list, csv_path="iou_zelenilo.csv")

    if results:
        image_names = [res["image_name"] for res in results]
        ious = [res["iou"] for res in results]

        plt.figure(figsize=(12, 5))
        plt.bar(image_names, ious)
        plt.xticks(rotation=90)
        plt.ylabel("IoU")
        plt.title("IoU po slici (nasumičnih 25 bez IoU=0)")
        plt.tight_layout()
        plt.show()

        best = max(results, key=lambda x: x["iou"])
        worst = min(results, key=lambda x: x["iou"])

        print(f"\nNajbolja slika: {best['image_name']} s IoU = {best['iou']:.4f}")
        print(f"Najgora slika: {worst['image_name']} s IoU = {worst['iou']:.4f}")

        best_image = os.path.join(data_dir, best['image_name'])
        best_mask = best_image.replace('_sat.jpg', '_mask.png')

        worst_image = os.path.join(data_dir, worst['image_name'])
        worst_mask = worst_image.replace('_sat.jpg', '_mask.png')

        seg_best = improved_hsv_lab_segmentation(best_image)
        gt_best = cv2.imread(best_mask)

        seg_worst = improved_hsv_lab_segmentation(worst_image)
        gt_worst = cv2.imread(worst_mask)

        show_comparison(best_image, seg_best, gt_best, "Najbolja slika")
        show_comparison(worst_image, seg_worst, gt_worst, "Najgora slika")
    else:
        print("\nNema valjanih slika s IoU > 0 za prikaz.")