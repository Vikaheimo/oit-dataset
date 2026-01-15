import os


def count_files_in_subfolders(base_folder="data"):
    total_files = 0
    folder_counts = {}

    for root, dirs, files in os.walk(base_folder):
        if root == base_folder:
            continue

        file_count = len(files)
        folder_counts[root] = file_count
        total_files += file_count

    return folder_counts, total_files


def warn_imbalances(folder_counts):
    folders = list(folder_counts.keys())
    warnings = []

    for i, folder_a in enumerate(folders):
        for j, folder_b in enumerate(folders):
            if i == j:
                continue
            count_a = folder_counts[folder_a]
            count_b = folder_counts[folder_b]
            if count_a > 2 * count_b:
                warnings.append(
                    f"⚠️ Warning: '{folder_a}' has more than twice as many files "
                    f"({count_a}) as '{folder_b}' ({count_b})."
                )

    return warnings


if __name__ == "__main__":
    data_folder = "data"
    folder_counts, total_files = count_files_in_subfolders(data_folder)

    print("File counts per subfolder:")
    for folder, count in folder_counts.items():
        percentage_of_files = count / total_files * 100
        print(f"{folder}: {count} ({percentage_of_files:.2f}%)")

    print(f"\nTotal number of files: {total_files}\n")

    warnings = warn_imbalances(folder_counts)
    if warnings:
        print("Imbalance warnings:")
        for w in warnings:
            print(w)
    else:
        print("✅ All subfolders are within a reasonable balance.")
