document.addEventListener('DOMContentLoaded', () => {
  const openBtn  = document.getElementById('open-modal-btn');
  const modal    = document.getElementById('camera-modal');
  const closeBtn = document.getElementById('close-modal');
  const video    = document.getElementById('video');
  const capture  = document.getElementById('capture');
  const toast    = document.getElementById('toast');
  let stream;
  let capturedBlobs = [];  // Mảng để lưu nhiều ảnh

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

  capture.addEventListener('click', () => {
    const canvas = document.createElement('canvas');
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);

    canvas.toBlob(blob => {
      if (blob) {
        capturedBlobs.push(blob);  // Thêm ảnh vào mảng
        showToast(`Đã chụp ${capturedBlobs.length} ảnh!`);
      }
    }, 'image/jpeg');
  });

  closeBtn.addEventListener('click', closeModal);

  function closeModal() {
    modal.style.display = 'none';
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
  }

  function showToast(message) {
    toast.textContent = message;
    toast.classList.add('show');
    setTimeout(() => {
      toast.classList.remove('show');
    }, 2000);
  }

  const form = document.getElementById('employeeForm');
  form.addEventListener('submit', function(e) {
    e.preventDefault();

    const formData = new FormData(form);

    // Gửi tất cả ảnh đã chụp
    capturedBlobs.forEach((blob, index) => {
      formData.append('photos', blob, `photo_${index}.jpg`);
    });

    fetch('/add_member', {
      method: 'POST',
      body: formData
    })
    .then(res => res.json())
    .then(result => {
      if (result.status === 'success') {
        showToast('Add employee successfully!');
        form.reset();
        capturedBlobs = [];  // Reset mảng ảnh sau khi gửi
      } else {
        showToast('Can not add employee');
      }
    })
    .catch(err => {
      console.error('Lỗi gửi form:', err);
      showToast('Lỗi kết nối đến server.');
    });
  });
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
