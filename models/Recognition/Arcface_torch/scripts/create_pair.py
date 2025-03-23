import os
import random
import argparse

# Hàm thiết lập và phân tích các tham số dòng lệnh
def parse_arguments():
    parser = argparse.ArgumentParser(description="Tạo file pairs.txt tương tự LFW từ dataset tùy chỉnh.")
    parser.add_argument("--dataset_path", type=str, default="custom_dataset",
                        help="Đường dẫn đến thư mục chứa dataset (mặc định: custom_dataset)")
    parser.add_argument("--num_folds", type=int, default=10,
                        help="Số lượng folds (mặc định: 10)")
    parser.add_argument("--pairs_per_fold", type=int, default=300,
                        help="Số cặp matched/mismatched mỗi fold (mặc định: 300)")
    parser.add_argument("--output_file", type=str, default="pairs.txt",
                        help="Tên file đầu ra (mặc định: pairs.txt)")
    return parser.parse_args()

# Lấy danh sách tất cả người và ảnh của họ
def load_dataset(dataset_path):
    people = {}
    for person in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person)
        if os.path.isdir(person_dir):
            images = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if len(images) >= 2:  # Chỉ thêm người có ít nhất 2 ảnh
                # Loại bỏ phần mở rộng khỏi tên ảnh
                images_no_ext = [os.path.splitext(img)[0] for img in images]
                people[person] = images_no_ext
    return people

# Hàm tạo cặp giống nhau (matched pairs) - 3 phần
def generate_matched_pairs(people_dict, num_pairs):
    matched_pairs = []
    person_list = list(people_dict.keys())
    while len(matched_pairs) < num_pairs:
        person = random.choice(person_list)
        if len(people_dict[person]) >= 2:
            img1, img2 = random.sample(people_dict[person], 2)  # Lấy tên ảnh không có phần mở rộng
            matched_pairs.append((person, img1, img2))  # 3 phần: tên người, tên ảnh 1, tên ảnh 2
    return matched_pairs

# Hàm tạo cặp khác nhau (mismatched pairs) - 4 phần
def generate_mismatched_pairs(people_dict, num_pairs):
    mismatched_pairs = []
    person_list = list(people_dict.keys())
    while len(mismatched_pairs) < num_pairs:
        person1, person2 = random.sample(person_list, 2)
        img1 = random.choice(people_dict[person1])
        img2 = random.choice(people_dict[person2])
        mismatched_pairs.append((person1, img1, person2, img2))  # 4 phần: tên người 1, tên ảnh 1, tên người 2, tên ảnh 2
    return mismatched_pairs

def main():
    # Phân tích tham số dòng lệnh
    args = parse_arguments()

    # Gán giá trị từ tham số
    dataset_path = args.dataset_path
    num_folds = args.num_folds
    pairs_per_fold = args.pairs_per_fold
    output_file = args.output_file

    # Tải dữ liệu từ dataset
    people = load_dataset(dataset_path)

    # Kiểm tra dữ liệu
    if len(people) < 2:
        raise ValueError("Cần ít nhất 2 người trong dataset để tạo cặp mismatched.")
    if len(people) < 100:
        print(f"Cảnh báo: Chỉ tìm thấy {len(people)} người, ít hơn 100 như yêu cầu.")

    # Tạo file pairs.txt
    with open(output_file, "w") as f:
        # Ghi dòng đầu tiên
        f.write(f"{num_folds}\t{pairs_per_fold}\n")
        
        # Tạo dữ liệu cho từng fold
        for fold in range(num_folds):
            # Tạo cặp giống nhau - 3 phần
            matched = generate_matched_pairs(people, pairs_per_fold)
            for person, img1, img2 in matched:
                f.write(f"{person}\t{img1}\t{img2}\n")
            
            # Tạo cặp khác nhau - 4 phần
            mismatched = generate_mismatched_pairs(people, pairs_per_fold)
            for p1, img1, p2, img2 in mismatched:
                f.write(f"{p1}\t{img1}\t{p2}\t{img2}\n")

    print(f"Đã tạo file {output_file} thành công!")

if __name__ == "__main__":
    # main()

    for folder in os.listdir("dataset/VN-celeb"):
        if not os.path.isdir(os.path.join("dataset/VN-celeb", folder)):
            continue
        for image_name in os.listdir(os.path.join("dataset/VN-celeb", folder)):
            new_name = image_name.replace(".png ", ".jpeg")


        