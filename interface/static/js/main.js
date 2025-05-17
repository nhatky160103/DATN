let resultCheckingInterval;

function showToast(message, isError=false) {
    const toast = document.getElementById('toast');
    toast.innerText = message;
    toast.style.backgroundColor = isError ? '#f44336' : '#4CAF50';
    toast.classList.remove('hidden');
    setTimeout(() => {
        toast.classList.add('hidden');
    }, 3000);
}

function toggleCamera() {
    const cameraModal = document.getElementById("cameraModal");
    const cameraFeed = document.getElementById("cameraFeed");
    const spinner = document.getElementById("loadingSpinner");

    cameraModal.classList.remove("hidden");
    setTimeout(() => cameraModal.classList.add("show"), 10); // Trigger animation

    // Hiển thị loading
    spinner.style.display = "block";
    cameraFeed.style.display = "none";

    // Gán stream mới (tránh cache)
    cameraFeed.src = "/video_feed?" + new Date().getTime();

    // Khi ảnh bắt đầu hiển thị, tắt loading + bắt đầu kiểm tra kết quả sau delay
    cameraFeed.onload = () => {
        spinner.style.display = "none";
        cameraFeed.style.display = "block";

        setTimeout(() => {
            startCameraMonitoring();
        }, 500); 
    };
}

function closeCamera() {
    const cameraModal = document.getElementById("cameraModal");
    const cameraFeed = document.getElementById("cameraFeed");

    cameraFeed.src = ""; // Ngắt kết nối stream

    clearInterval(resultCheckingInterval);

    cameraModal.classList.remove("show");
    setTimeout(() => {
        cameraModal.classList.add("hidden");
    }, 400); // Khớp với animation
}

function startCameraMonitoring() {
    resultCheckingInterval = setInterval(() => {
        fetch('/get_results')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'no_results') {
                    console.log("⏳ Chưa có kết quả nhận diện...");
                    return;
                }

                clearInterval(resultCheckingInterval);
                closeCamera();

                getResults(data);
            })
            .catch(error => {
                console.error("Polling error:", error);
            });
    }, 50); // Kiểm tra mỗi 0.1 giây
}

function getResults(dataFromPolling = null) {
    const nameSpan = document.getElementById("name");
    const timeSpan = document.getElementById("time");

    const renderResult = (data) => {
        if (data.status === 'no_results') {
            nameSpan.textContent = "UNKNOWN";
            timeSpan.textContent = "0000:00:00 00:00:00";
        } else if (data.status === 'error') {
            nameSpan.textContent = "--";
            timeSpan.textContent = `Error: ${data.message}`;
        } else {
            nameSpan.textContent = data.employee_id;
            timeSpan.textContent = new Date(data.time * 1000).toLocaleString();
            console.log("✅ Nhận diện:", data.employee_id, data.time);
        }

        // Reset sau 2 giây
        setTimeout(() => {
            nameSpan.textContent = "--";
            timeSpan.textContent = "--";
        }, 3000);
    };

    if (dataFromPolling) {
        renderResult(dataFromPolling);
    } else {
        fetch('/get_results')
            .then(response => response.json())
            .then(renderResult)
            .catch(error => {
                nameSpan.textContent = "--";
                timeSpan.textContent = `Error: ${error}`;
            });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const bucketSelect = document.getElementById('bucketSelect');
    
    // Lắng nghe sự thay đổi trong dropdown
    bucketSelect.addEventListener('change', function() {
        const selectedBucket = this.value;
  
        // Lưu bucket vào localStorage
        localStorage.setItem('selectedBucket', selectedBucket);
  
        // Gửi bucket lên server để xử lý (ví dụ qua AJAX)
        fetch('/set_selected_bucket', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ bucket: selectedBucket })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Bucket selected:', data);
        })
        .catch(error => console.error('Error:', error));
    });
  
    // Lấy bucket đã chọn từ localStorage (nếu có)
    const savedBucket = localStorage.getItem('selectedBucket');
    if (savedBucket) {
        bucketSelect.value = savedBucket;
  
        // Gửi lại bucket từ localStorage lên server nếu có
        fetch('/set_selected_bucket', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ bucket: savedBucket })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Bucket đã được cập nhật từ localStorage:', data);
        })
        .catch(error => console.error('Error:', error));
    }
  });



function openNewBucketModal() {
    document.getElementById('newBucketModal').classList.remove('hidden');
}

function closeNewBucketModal() {
    document.getElementById('newBucketModal').classList.add('hidden');
}



function createNewBucket() {
    const bucketName = document.getElementById('bucketNameInput').value.trim();
    const logoInput = document.getElementById('bucketLogoInput');
    const logoFile = logoInput ? logoInput.files[0] : null;
    const loadingOverlay = document.getElementById('loading-overlay'); 
    if (!bucketName) {
        showToast("❌ Please enter a bucket name!", true);
        return;
    }

    loadingOverlay.style.display = 'flex';

    let fetchOptions;
    if (logoFile) {
        // Gửi form-data nếu có file logo
        const formData = new FormData();
        formData.append('bucket_name', bucketName);
        formData.append('logo', logoFile);
        fetchOptions = {
            method: 'POST',
            body: formData
        };
    } else {
        // Gửi JSON như cũ nếu không có logo
        fetchOptions = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ bucket_name: bucketName })
        };
    }

    fetch('/create_bucket', fetchOptions)
    .then(response => response.json())
    .then(data => {
        loadingOverlay.style.display = 'none';
        if (data.success) {
            showToast(`✅ Created bucket '${bucketName}' successfully!`);
            closeNewBucketModal();
            setTimeout(() => window.location.reload(), 1500);
        } else {
            showToast(`⚠️ ${data.message}`, true);
        }
    })
    .catch(error => {
        loadingOverlay.style.display = 'none';
        showToast("❌ Error creating bucket!", true);
        console.error(error);
    });
}