let resultCheckingInterval;
let cameraForceClosed = false;

function showToast(message, isError=false) {
    const toast = document.getElementById('toast');
    toast.innerText = message;
    toast.style.backgroundColor = isError ? '#f44336' : '#4CAF50';
    toast.classList.remove('hidden');
    setTimeout(() => {
        toast.classList.add('hidden');
    }, 2000);
}

function toggleCamera() {
    cameraForceClosed = false; // Reset trạng thái khi mở lại camera
    const cameraModal = document.getElementById("cameraModal");
    const cameraFeed = document.getElementById("cameraFeed");
    const spinner = document.getElementById("loadingSpinner");

    if (!cameraModal || !cameraFeed || !spinner) {
        console.error("❌ Không tìm thấy các element cần thiết cho camera");
        showToast("❌ Lỗi khởi tạo camera!", true);
        return;
    }

    if (cameraModal.classList.contains("show")) {
        console.log("📷 Camera đã đang mở, đóng camera...");
        closeCamera();
        return;
    }

    cameraModal.classList.remove("show");
    cameraModal.classList.remove("hidden");
    cameraModal.offsetHeight;
    cameraModal.classList.remove("hidden");
    requestAnimationFrame(() => {
        cameraModal.classList.add("show");
    });

    cameraFeed.style.display = "block";
    spinner.style.display = "none";
    console.log("🎥 Đang mở camera...");
    cameraFeed.src = "/video_feed?" + new Date().getTime();
    cameraFeed.onload = () => {
        console.log("✅ Camera đã load thành công");
        setTimeout(() => {
            startCameraMonitoring();
        }, 500); 
    };
    cameraFeed.onerror = () => {
        console.error("❌ Lỗi khi load camera");
        showToast("❌ Không thể kết nối camera!", true);
        closeCamera();
    };
}

function closeCamera() {
    const cameraModal = document.getElementById("cameraModal");
    const cameraFeed = document.getElementById("cameraFeed");
    const spinner = document.getElementById("loadingSpinner");

    if (!cameraModal || !cameraFeed) {
        console.error("❌ Không tìm thấy các element cần thiết để đóng camera");
        return;
    }

    console.log("🔒 Đang đóng camera...");
    clearInterval(resultCheckingInterval);
    cameraFeed.src = "";
    cameraFeed.onload = null;
    cameraFeed.onerror = null;
    cameraFeed.style.display = "none";
    if (spinner) {
        spinner.style.display = "flex";
    }
    cameraModal.classList.remove("show");
    setTimeout(() => {
        cameraModal.classList.add("hidden");
    }, 400);
}

function forceCloseCamera() {
    cameraForceClosed = true;
    const cameraModal = document.getElementById("cameraModal");
    const cameraFeed = document.getElementById("cameraFeed");
    
    // Stop the stream immediately to release the backend camera
    if (cameraFeed) {
        cameraFeed.src = "";
    }
    
    // Stop any ongoing polling
    clearInterval(resultCheckingInterval);

    // Animate closing the modal, then reload the page
    if (cameraModal && cameraModal.classList.contains('show')) {
        cameraModal.classList.remove("show");
        // Reload after the animation finishes
        setTimeout(() => {
            window.location.reload();
        }, 400); // Match animation duration
    } else {
        // If modal is not shown or doesn't exist, just reload
        window.location.reload();
    }
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
                getResults(data);
            })
            .catch(error => {
                console.error("Polling error:", error);
            });
    }, 50); 
}

function getResults(dataFromPolling = null) {
    const nameSpan = document.getElementById("name");
    const timeSpan = document.getElementById("time");
    
    // Lấy các element của overlay
    const recognitionOverlay = document.getElementById("recognitionOverlay");
    const recognitionCard = recognitionOverlay.querySelector('.recognition-card');
    const recognitionName = document.getElementById("recognitionName");
    const recognitionTime = document.getElementById("recognitionTime");
    const recognitionStatus = document.getElementById("recognitionStatus");
    const recognitionIcon = document.getElementById("recognitionIcon");

    const renderResult = (data) => {
        // Bỏ qua nếu không có kết quả
        if (data.status === 'no_results') {
            return;
        }

        const formattedTime = new Date(data.time * 1000).toLocaleString();

        if (data.employee_id === 'UNKNOWN') {
            // Cập nhật trang chính
            nameSpan.textContent = "UNKNOWN";
            timeSpan.textContent = formattedTime;
            
            // Cập nhật overlay cho trường hợp lỗi
            recognitionName.textContent = "UNKNOWN";
            recognitionTime.textContent = formattedTime;
            recognitionStatus.textContent = "Failed to checkin!";
            recognitionIcon.textContent = "❌";
            recognitionCard.classList.add('error');
        } else {
            // Cập nhật trang chính
            nameSpan.textContent = data.employee_id;
            timeSpan.textContent = formattedTime;
            
            // Cập nhật overlay cho trường hợp thành công
            recognitionName.textContent = data.employee_id;
            recognitionTime.textContent = formattedTime;
            recognitionStatus.textContent = "Successfully!";
            recognitionIcon.textContent = "✅";
            recognitionCard.classList.remove('error');
        }
        
        // Hiển thị overlay với animation
        recognitionOverlay.classList.add("show");
        
        // Tự động ẩn overlay sau 3 giây, sau đó tắt camera và mở lại camera để nhận diện người tiếp theo
        setTimeout(() => {
            recognitionOverlay.classList.remove("show");
            closeCamera();
            setTimeout(() => {
                if (!cameraForceClosed) {
                    waitForCameraReadyAndOpen();
                }
            }, 500); // delay ngắn, sau đó polling tới khi backend sẵn sàng
        }, 3000);
    };

    if (dataFromPolling) {
        renderResult(dataFromPolling);
        // Không cần setTimeout ở đây nữa, vì đã được xử lý bên trong renderResult
    } else {
        fetch('/get_results')
            .then(response => response.json())
            .then(renderResult)
            .catch(error => {
                console.error("Lỗi get_results:", error);
            });
    }
}

function waitForCameraReadyAndOpen() {
    function check() {
        fetch('/camera_status')
            .then(res => res.json())
            .then(data => {
                if (data.status === 'ready') {
                    toggleCamera();
                } else {
                    setTimeout(check, 300); // thử lại sau 300ms
                }
            });
    }
    check();
}

document.addEventListener('DOMContentLoaded', () => {
    const bucketSelect = document.getElementById('bucketSelect');
    
    // Đảm bảo nút camera hoạt động
    const cameraButton = document.querySelector('button[onclick="toggleCamera()"]');
    if (cameraButton) {
        // Thêm event listener backup
        cameraButton.addEventListener('click', (e) => {
            e.preventDefault();
            console.log("🔘 Nút camera được click");
            toggleCamera();
        });
    }
    
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
    
    // Debug: Kiểm tra các element quan trọng
    console.log("🔍 Kiểm tra các element quan trọng:");
    console.log("- Camera Modal:", document.getElementById("cameraModal") ? "✅" : "❌");
    console.log("- Camera Feed:", document.getElementById("cameraFeed") ? "✅" : "❌");
    console.log("- Loading Spinner:", document.getElementById("loadingSpinner") ? "✅" : "❌");
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
    if (!logoFile) {
        showToast("❌ Please select logo image!", true);
        return;
    }

    loadingOverlay.style.display = 'flex';

    let fetchOptions;
    // Chỉ còn nhánh có logoFile
    const formData = new FormData();
    formData.append('bucket_name', bucketName);
    formData.append('logo', logoFile);
    fetchOptions = {
        method: 'POST',
        body: formData
    };

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