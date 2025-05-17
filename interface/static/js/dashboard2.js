var notifications = [];

function showToast(message, status = 'success') {
  const toastContainer = document.getElementById('toastContainer');
  const toast = document.getElementById('toast');
 
  toastContainer.classList.add('show');
  toastContainer.style.display = 'block';

  toast.textContent = message;
  notifications.push(message);
  toast.classList.add(status);
  setTimeout(() => {
    toast.classList.remove(status);
  }, 2000);

  // Gọi lại renderNotifications khi đã sẵn sàng
  if (typeof renderNotifications === 'function') {
    renderNotifications();
  }
}

var notificationBtn, notificationDropdown, notificationList, notificationBadge;

function renderNotifications() {
  notificationList.innerHTML = '';

  if (notifications.length === 0) {
    notificationList.innerHTML = '<li>No notifications</li>';
    notificationBadge.style.display = 'none';
    return;
  }

  notificationBadge.style.display = 'inline-block';
  notificationBadge.textContent = notifications.length;

  notifications.forEach((msg, idx) => {
    const li = document.createElement('li');
    li.textContent = msg;

    li.addEventListener('click', () => {
      alert(`${msg}`);
      notifications.splice(idx, 1);
      renderNotifications();
    });

    notificationList.appendChild(li);
  });
}

document.addEventListener('DOMContentLoaded', () => {
  notificationBtn = document.getElementById('notificationBtn');
  notificationDropdown = document.getElementById('notificationDropdown');
  notificationList = document.getElementById('notificationList');
  notificationBadge = document.getElementById('notificationBadge');

  notificationBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    notificationDropdown.classList.toggle('hidden');
  });

  document.addEventListener('click', () => {
    notificationDropdown.classList.add('hidden');
  });

  renderNotifications();
});



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
        showToast(`Captured ${capturedBlobs.length} photos!`, 'success');
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
    showToast(`${selectedFiles.length} files selected.`, 'success');
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
      body: formData
    })
    .then(res => res.json())
    .then(result => {
      loadingOverlay.style.display = 'none';  // ✅ Ẩn overlay khi có phản hồi

      if (result.status === 'success') {
        showToast('Employee added successfully!', 'success');
        form.reset();
        capturedBlobs = [];
        selectedFiles = [];  // Reset uploaded files
      } else {
        showToast('Failed to add employee', 'error');
      }
    })
    .catch(err => {
      loadingOverlay.style.display = 'none';  // ✅ Ẩn overlay khi lỗi
      console.error('Form submission error:', err);
      showToast('Connection error.', 'error');
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
          anti_spoof_threshold: parseFloat(document.getElementById('anti_spoof_threshold').value),
      },
      identity_person: {
          l2_threshold: parseFloat(document.getElementById('l2_threshold').value),
          cosine_threshold: parseFloat(document.getElementById('cosine_threshold').value),
          distance_mode: document.getElementById('identity_distance_mode').value,
      },
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
      showToast('Configuration saved successfully', 'success');
  })
  .catch((error) => {
      showToast('Please try again!', 'error')
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





let isTableVisible = false;
let latestData = null;
let popupWindow = null;

async function toggleTimekeeping() {
    const btnToggle = document.getElementById("toggleDetailBtn");
    const selectedDate = document.getElementById("exportDate").value;
    const overlay = document.getElementById("overlay");

    // Kiểm tra nếu chưa chọn ngày
    if (!selectedDate) {
        showToast("Please select a date!", "error");
        return;
    }

    if (!isTableVisible) {
        btnToggle.disabled = true;
        try {
            const response = await fetch("/export-timekeeping", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ date: selectedDate })
            });

            const result = await response.json();
            if (result.success) {
                latestData = result.data;
                openPopupTable(latestData);
                showToast("✅ Data loaded successfully!", "success");
                overlay.classList.add("show");
                isTableVisible = true;
            } else {
                throw new Error(result.message || "❌ Error loading data!");
            }
        } catch (error) {
            console.error(error);
            showToast("Failed to load data!", "error");
        } finally {
            btnToggle.disabled = false;
        }
    } else {
        closePopup();
        overlay.classList.remove("show");
        isTableVisible = false;
    }
}

function openPopupTable(data) {
    popupWindow = document.createElement('div');
    popupWindow.className = 'popup-window';
    popupWindow.innerHTML = `
        <div class="popup-header">
            <h3>🕒 Timekeeping Data</h3>
            <button class="close-btn" onclick="closePopup()">Close</button>
        </div>
        <div class="table-container">
            <table id="timekeepingTable">
                <thead><tr id="tableHeader"></tr></thead>
                <tbody id="tableBody"></tbody>
            </table>
        </div>
    `;

    document.body.appendChild(popupWindow);
    renderTable(data);
}


function closePopup() {
    if (popupWindow) {
        popupWindow.remove();
        popupWindow = null;
    }
    document.getElementById("overlay").classList.remove("show");
    isTableVisible = false;
}

function renderTable(data) {
    const header = document.getElementById('tableHeader');
    const body = document.getElementById('tableBody');

    header.innerHTML = "";
    body.innerHTML = "";

    if (!data || data.length === 0) return;

    const keys = Object.keys(data[0]);
    const sortedKeys = ['Employee ID', ...keys.filter(k => k !== 'Employee ID')];

    sortedKeys.forEach(key => {
        const th = document.createElement('th');
        th.innerText = key;
        header.appendChild(th);
    });

    data.forEach(item => {
        const tr = document.createElement('tr');
        sortedKeys.forEach(key => {
            const td = document.createElement('td');
            td.innerText = item[key] !== undefined ? item[key] : "";
            tr.appendChild(td);
        });
        body.appendChild(tr);
    });
}

async function downloadExcel() {
  const selectedDate = document.getElementById("exportDate").value;
  if (!selectedDate) {
      showToast("Please select a date!", "error");
      return;
  }

  try {
      // Gửi yêu cầu để xuất Excel
      const response = await fetch("/download-excel", { method: "GET" });
      const result = await response.blob();  // Nhận dữ liệu dưới dạng Blob

      if (result) {
          // Dùng FileSaver.js để lưu tệp Excel
          saveAs(result, "timekeeping.xlsx");
          showToast("✅ Data exported successfully!", "success");
      } else {
          throw new Error(result.message || "Failed to export data.");
      }
  } catch (error) {
      console.error(error);
      showToast("Error exporting data!", "error");
  }
}


function loadPersonList() {
  fetch('/get_person_ids') // Gửi yêu cầu tới server để lấy danh sách person_id
      .then(response => response.json())
      .then(personIds => {
          const selectElement = document.getElementById('personListSelect');
          selectElement.innerHTML = ''; // Xóa tất cả các option hiện có
          personIds.forEach(personId => {
              const option = document.createElement('option');
              option.value = personId;
              option.textContent = personId;
              selectElement.appendChild(option);
          });
      })
      .catch(error => console.error('Error loading person IDs:', error));
}


function deleteEmployee() {
  const personId = document.getElementById('personListSelect').value;
  const loadingOverlay = document.getElementById('loading-overlay'); 
  const toast = document.querySelector('#toast2')
  if (!personId) {
      showToast('Please select employee', 'error');
      return;
  }
  
  loadingOverlay.style.display = 'flex'; 

  
  // Gửi request DELETE đến API
  fetch(`/delete_person/${personId}`, {
      method: 'DELETE',
  })
  .then(response => response.json())
  .then(data => {
      loadingOverlay.style.display = 'none';
      if (data.success) {
          
          showToast('Delete succesfully!', 'success');

          const select = document.getElementById('personListSelect');
          const option = select.querySelector(`option[value="${personId}"]`);
          if (option) option.remove();
      } else {
          showToast(data.error || 'Failed!', 'error');
      }

  })
  .catch(error => {
      loadingOverlay.style.display = 'none';
      showToast('Error: ' + error.message, 'error');
    
  });

}


window.onload = loadPersonList;

document.addEventListener('DOMContentLoaded', function() {
  // Popup xác nhận xóa bucket
  const showConfirmBtn = document.getElementById('showConfirmDeleteBtn');
  const modal = document.getElementById('deleteBucketModal');
  const closeModalBtn = document.getElementById('closeDeleteBucketModal');
  const deleteBucketBtn = document.getElementById('deleteBucketBtn');
  const confirmInput = document.getElementById('confirmBucketName');

  if (showConfirmBtn && modal && closeModalBtn && deleteBucketBtn && confirmInput) {
    showConfirmBtn.addEventListener('click', function() {
      modal.style.display = 'flex';
      confirmInput.value = '';
      confirmInput.focus();
    });

    closeModalBtn.addEventListener('click', function() {
      modal.style.display = 'none';
    });

    window.onclick = function(event) {
      if (event.target === modal) {
        modal.style.display = 'none';
      }
    };

    deleteBucketBtn.addEventListener('click', function() {
      const input = confirmInput.value.trim();
      const currentBucket = document.querySelector('.profile span').textContent.trim();
      if (input !== currentBucket) {
        showToast('Bucket name does not match. Please type the exact bucket name to confirm.', 'error');
        return;
      }
      if (!confirm(`Are you sure you want to delete bucket "${currentBucket}"? This action cannot be undone!`)) {
        return;
      }

      const loadingOverlay = document.getElementById('loading-overlay');
      if (loadingOverlay) loadingOverlay.style.display = 'flex';
      
      fetch('/delete_bucket', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({bucket_name: currentBucket})
      })
      .then(res => res.json())
      .then(data => {
        if (loadingOverlay) loadingOverlay.style.display = 'none';
        if (data.success) {
          showToast(data.message, 'success');
          modal.style.display = 'none';
          setTimeout(() => {
            window.location.href = '/';
          }, 1500);
        } else {
          showToast(data.message || 'Failed to delete bucket!', 'error');
        }
      })
      .catch(err => {
        if (loadingOverlay) loadingOverlay.style.display = 'none';
        showToast('Error deleting bucket!', 'error');
      });
    });
  }
});