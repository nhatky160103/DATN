document.addEventListener('DOMContentLoaded', () => {
  const openBtn = document.getElementById('open-modal-btn');
  const modal = document.getElementById('camera-modal');
  const closeBtn = document.getElementById('close-modal');
  const video = document.getElementById('video');
  const capture = document.getElementById('capture');
  const toast = document.getElementById('toast');
  const openUploadBtn = document.getElementById('open-upload-btn');
  const uploadModal = document.getElementById('upload-modal');
  const closeUploadBtn = document.getElementById('close-upload-modal');
  const realFileInput = document.getElementById('upload-photo-real');
  const hiddenFileInput = document.getElementById('upload-photo-hidden');
  const loadingOverlay = document.getElementById('loading-overlay'); // ✅ Thêm dòng này
  let stream;
  let capturedBlobs = [];  // Mảng để lưu nhiều ảnh
  let selectedFiles = [];  // Mảng để lưu các ảnh đã chọn từ máy tính

  // Open camera modal
  openBtn.addEventListener('click', async () => {
    modal.style.display = 'flex';
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;
    } catch (err) {
      alert('Can not open camera: ' + err.message);
      closeModal();
    }
  });

  // Capture photo from camera
  capture.addEventListener('click', () => {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);

    canvas.toBlob(blob => {
      if (blob) {
        capturedBlobs.push(blob);  // Thêm ảnh vào mảng
        showToast(`Captured ${capturedBlobs.length} photos!`);
      }
    }, 'image/jpeg');
  });

  // Close camera modal
  closeBtn.addEventListener('click', closeModal);
  function closeModal() {
    modal.style.display = 'none';
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
  }

  // Upload photo modal
  openUploadBtn.addEventListener('click', () => {
    uploadModal.style.display = 'flex';
  });

  closeUploadBtn.addEventListener('click', () => {
    uploadModal.style.display = 'none';
  });

  // Handle file selection for upload
  realFileInput.addEventListener('change', () => {
    selectedFiles = Array.from(realFileInput.files);
    hiddenFileInput.files = realFileInput.files;  // Assign the selected files to hidden input
    uploadModal.style.display = 'none';
    showToast(`${selectedFiles.length} files selected.`);
  });

  // Handle form submission
  const form = document.getElementById('employeeForm');
  form.addEventListener('submit', function (e) {
    e.preventDefault();

    loadingOverlay.style.display = 'flex';  // ✅ Hiện overlay khi submit

    const formData = new FormData(form);

    // Add photos captured by camera
    capturedBlobs.forEach((blob, index) => {
      formData.append('photos', blob, `photo_${index}.jpg`);
    });

    // Add photos uploaded from computer
    selectedFiles.forEach((file, index) => {
      formData.append('photos', file, file.name);
    });

    fetch('/add_member', {
      method: 'POST',
      body: formDatasave_config
    })
    .then(res => res.json())
    .then(result => {
      loadingOverlay.style.display = 'none';  // ✅ Ẩn overlay khi có phản hồi

      if (result.status === 'success') {
        showToast('Employee added successfully!');
        form.reset();
        capturedBlobs = [];
        selectedFiles = [];  // Reset uploaded files
      } else {
        showToast('Failed to add employee');
      }
    })
    .catch(err => {
      loadingOverlay.style.display = 'none';  // ✅ Ẩn overlay khi lỗi
      console.error('Form submission error:', err);
      showToast('Connection error.');
    });
  });

  // Show toast message
  function showToast(message) {
    toast.textContent = message;
    toast.classList.add('show');
    setTimeout(() => {
      toast.classList.remove('show');
    }, 2000);
  }
});



function collectConfigData() {
  const configData = {
      infer_video: {
          min_face_area: parseFloat(document.getElementById('min_face_area').value),
          bbox_threshold: parseFloat(document.getElementById('bbox_threshold').value),
          required_images: parseInt(document.getElementById('required_images').value),
          validation_threshold: parseFloat(document.getElementById('validation_threshold').value),
          is_anti_spoof: document.getElementById('is_anti_spoof').checked,
          distance_mode: document.getElementById('distance_mode').value,
          anti_spoof_threshold: parseFloat(document.getElementById('anti_spoof_threshold').value),
      },
      identity_person: {
          l2_threshold: parseFloat(document.getElementById('l2_threshold').value),
          cosine_threshold: parseFloat(document.getElementById('cosine_threshold').value),
          distance_mode: document.getElementById('identity_distance_mode').value,
      },
      collect_data: {
          min_face_area: parseInt(document.getElementById('collect_min_face_area').value),
          bbox_threshold: parseFloat(document.getElementById('collect_bbox_threshold').value),
          required_images: parseInt(document.getElementById('collect_required_images').value),
          is_anti_spoof: document.getElementById('collect_is_anti_spoof').checked,
          anti_spoof_threshold: parseFloat(document.getElementById('collect_anti_spoof_threshold').value),
      }
  };
  console.log(configData);  // Kiểm tra dữ liệu gửi lên
  return configData;
}

function saveConfig() {
  const configData = collectConfigData();

  fetch('/save_config', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json',
      },
      body: JSON.stringify(configData)
  })
  .then(response => response.json())
  .then(data => {
      alert('Configuration saved successfully');
  })
  .catch((error) => {
      console.error('Error:', error);
  });
}



function createEmbedding() {
  const statusElement = document.getElementById("embeddingStatus");
  statusElement.innerText = "Creating embeddings...";

  statusElement.classList.add("loading");

  fetch("/create-embedding", {
      method: "POST", 
      headers: {
          'Content-Type': 'application/json',
      },
      body: JSON.stringify({}) 
  })
  .then(response => {
      if (!response.ok) {
          throw new Error('Failed to create embeddings');
      }
      return response.json();
  })
  .then(data => {
      if (data.success) {
          statusElement.innerText = "✅ Embeddings created successfully!";
      } else {
          statusElement.innerText = "❌ Failed to create embeddings.";
      }
      statusElement.classList.remove("loading");
  })
  .catch(error => {
      console.error("Error:", error);
      statusElement.innerText = "❌ Error occurred during embedding creation.";
      // Xử lý lỗi và loại bỏ class loading nếu có lỗi
      statusElement.classList.remove("loading");
  });
}


function exportTimekeeping() {
  const btn = document.querySelector("#exportTimekeeping button");
  const status = document.getElementById("exportStatus");
  const loading = document.getElementById("exportAnimation");
  const selectedDate = document.getElementById("exportDate").value;

  if (!selectedDate) {
      status.innerText = "❗ Please select a date to export!";
      status.className = "error";
      return;
  }

  // Processing state
  btn.disabled = true;
  btn.innerText = "⏳ Exporting data...";
  status.innerText = "";
  loading.classList.remove("hidden");

  fetch("/export-timekeeping", {
      method: "POST",
      headers: {
          "Content-Type": "application/json"
      },
      body: JSON.stringify({ date: selectedDate })
  })
  .then(res => res.json())
  .then(data => {
      status.innerText = data.message || "✅ Export successful!";
      status.className = "success";
  })
  .catch(err => {
      console.error(err);
      status.innerText = "❌ Failed to export data!";
      status.className = "error";
  })
  .finally(() => {
      btn.disabled = false;
      btn.innerText = "🚀 Start Export";
      loading.classList.add("hidden");
  });
}
