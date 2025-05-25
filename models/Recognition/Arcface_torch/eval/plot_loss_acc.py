import re
import matplotlib.pyplot as plt
import os

# Đường dẫn đến các file
log_cmdl = "models/Recognition/Arcface_torch/weights/casia_webface_cmd_r50_lite_fp16/training.log"       # file log CDML
log_arc = "models/Recognition/Arcface_torch/weights/casia_webface_arcface_r50_lite_fp16/training.log"     # file log ArcFace

# Đường dẫn lưu hình ảnh
save_dir = "models/Recognition/Arcface_torch/eval/r50_lite_loss_acc_plot"
os.makedirs(save_dir, exist_ok=True)

# Hàm trích xuất step và loss từ file log
def extract_loss_and_step(filepath):
    pattern = re.compile(r"Loss (\d+\.\d+).*?Global Step: (\d+)")
    losses = []
    steps = []
    with open(filepath, "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                loss = float(match.group(1))
                step = int(match.group(2))  
                losses.append(loss)
                steps.append(step)
    return steps, losses

# Hàm trích xuất tất cả accuracy lfw qua các iteration từ file log
def extract_lfw_accuracy_from_log(filepath):
    pattern = re.compile(r"\[lfw\]\[(\d+)\]Accuracy-Flip: ([0-9.]+)\+\-[0-9.]+")
    iterations = []
    accuracies = []
    with open(filepath, "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                iteration = int(match.group(1))
                accuracy = float(match.group(2))
                iterations.append(iteration)
                accuracies.append(accuracy)
    return iterations, accuracies
def extract_cfp_fp_accuracy_from_log(filepath):
    # Nhận mọi số thực, số mũ, số 1.0, số 0, ...
    pattern = re.compile(r"\[cfp_fp\]\[(\d+)\]Accuracy-Flip: ([\d\.eE\+\-]+)\+\-([\d\.eE\+\-]+)")
    iterations = []
    accuracies = []
    stds = []
    with open(filepath, "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                iteration = int(match.group(1))
                accuracy = float(match.group(2))
                std = float(match.group(3))
                iterations.append(iteration)
                accuracies.append(accuracy)
                stds.append(std)
    return iterations, accuracies, stds
def extract_agedb_30_accuracy_from_log(filepath):
    pattern = re.compile(r"\[agedb_30\]\[(\d+)\]Accuracy-Flip: ([\d\.eE\+\-]+)\+\-[\d\.eE\+\-]+")
    iterations = []
    accuracies = []
    with open(filepath, "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                iteration = int(match.group(1))
                accuracy = float(match.group(2))
                iterations.append(iteration)
                accuracies.append(accuracy)
    return iterations, accuracies

# Trích xuất dữ liệu từ các file
steps_cmdl, losses_cmdl = extract_loss_and_step(log_cmdl)
steps_arc, losses_arc = extract_loss_and_step(log_arc)

# Trích xuất accuracy lfw
iters_cmdl, accs_cmdl = extract_lfw_accuracy_from_log(log_cmdl)
iters_arc, accs_arc = extract_lfw_accuracy_from_log(log_arc)

# Trích xuất accuracy cfp_fp
iters_cmdl_cfp, accs_cmdl_cfp, stds_cmdl_cfp = extract_cfp_fp_accuracy_from_log(log_cmdl)
iters_arc_cfp, accs_arc_cfp, stds_arc_cfp = extract_cfp_fp_accuracy_from_log(log_arc)

# Trích xuất accuracy agedb_30
iters_cmdl_agedb, accs_cmdl_agedb = extract_agedb_30_accuracy_from_log(log_cmdl)
iters_arc_agedb, accs_arc_agedb = extract_agedb_30_accuracy_from_log(log_arc)

# Vẽ loss (giữ nguyên)
plt.figure(figsize=(12, 6))
plt.plot(steps_cmdl, losses_cmdl, label="CDML Loss", color="blue", linewidth=1)
plt.plot(steps_arc, losses_arc, label="ArcFace Loss", color="green", linewidth=1)
plt.xlabel("Step", fontsize=16)
plt.ylabel("Loss", fontsize=16)
# plt.title("So sánh Loss giữa CDML và ArcFace", fontsize=18) 
plt.legend(fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "loss_compare.png"))
plt.close()

# Vẽ accuracy lfw qua các iteration
plt.figure(figsize=(12, 6))
plt.plot(iters_cmdl, accs_cmdl, marker='o', label="CDML lfw Accuracy", color="blue", linewidth=2)
plt.plot(iters_arc, accs_arc, marker='o', label="ArcFace lfw Accuracy", color="green", linewidth=2)
plt.xlabel("Iteration", fontsize=16)
plt.ylabel("Accuracy (lfw)", fontsize=16)
# plt.title("Accuracy trên tập lfw qua các iteration", fontsize=18)
plt.legend(fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "accuracy_lfw.png"))
plt.close()

# Vẽ accuracy cfp_fp qua các iteration
plt.figure(figsize=(12, 6))
plt.plot(iters_cmdl_cfp, accs_cmdl_cfp, marker='o', label="CDML cfp_fp Accuracy", color="blue", linewidth=2)
plt.plot(iters_arc_cfp, accs_arc_cfp, marker='o', label="ArcFace cfp_fp Accuracy", color="green", linewidth=2)
plt.xlabel("Iteration", fontsize=16)
plt.ylabel("Accuracy (cfp_fp)", fontsize=16)
# plt.title("Accuracy trên tập cfp_fp qua các iteration", fontsize=18)
plt.legend(fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "accuracy_cfp_fp.png"))
plt.close()

# Vẽ accuracy agedb_30 qua các iteration
plt.figure(figsize=(12, 6))
plt.plot(iters_cmdl_agedb, accs_cmdl_agedb, marker='o', label="CDML agedb_30 Accuracy", color="blue", linewidth=2)
plt.plot(iters_arc_agedb, accs_arc_agedb, marker='o', label="ArcFace agedb_30 Accuracy", color="green", linewidth=2)
plt.xlabel("Iteration", fontsize=16)
plt.ylabel("Accuracy (agedb_30)", fontsize=16)
# plt.title("Accuracy trên tập agedb_30 qua các iteration", fontsize=18)
plt.legend(fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "accuracy_agedb_30.png"))
plt.close()


