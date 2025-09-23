let isCameraOpen = false;
let cameraState = {
    stream: null,
    captureInterval: null
};

let currentAudio = null;
let lastAudioTime = 0;
const audioDelay = 2000; // ms
let audioPlayedForRecognition = false; // flag audio greeting đã play

function showToast(message, isError = false) {
    const toast = document.getElementById('toast');
    if (!toast) return;
    toast.innerText = message;
    toast.style.backgroundColor = isError ? '#f44336' : '#4CAF50';
    toast.classList.remove('hidden');
    setTimeout(() => toast.classList.add('hidden'), 3000);
}

let greetingPlayed = false;
let greetingAlreadyTriggered = false;


function playAudioGuide(audioUrl, forcePlay = false) {
    if (!audioUrl) return;

    // Chặn greeting.mp3 chỉ phát 1 lần
    if (forcePlay) {
        if (greetingAlreadyTriggered) return; 
        greetingAlreadyTriggered = true;
    }

    const now = Date.now();
    if (!forcePlay && now - lastAudioTime < audioDelay) return;
    lastAudioTime = now;

    if (currentAudio && !forcePlay) {
        currentAudio.pause();
        currentAudio = null;
    }

    currentAudio = new Audio(audioUrl);
    currentAudio.play().catch(e => console.log('Audio play error:', e));

    currentAudio.onended = () => {
        if (forcePlay) audioPlayedForRecognition = true;
        currentAudio = null;
    };
}


function showRecognitionResult(employee_id, time) {
    const nameSpan = document.getElementById("name");
    const timeSpan = document.getElementById("time");
    if (nameSpan) nameSpan.textContent = employee_id ?? "--";
    if (timeSpan) timeSpan.textContent = time ? new Date(time * 1000).toLocaleString() : "--";

    // Reset sau 3 giây
    setTimeout(() => {
        if (nameSpan) nameSpan.textContent = "--";
        if (timeSpan) timeSpan.textContent = "--";
    }, 3000);
}

// Gửi frame lên backend
async function sendFrameForRecognition(blob) {
    if (!blob) return null;

    const formData = new FormData();
    formData.append('image', blob, 'frame.jpg');

    try {
        const response = await fetch('/infer_camera_upload', { method: 'POST', body: formData });
        const result = await response.json();

        if (result?.audio_guide) {
            const forcePlay = result.enough_images === true; // ưu tiên audio greeting
            if (currentAudio?.src !== result.audio_guide || forcePlay) {
                playAudioGuide(result.audio_guide, forcePlay);
            }
        }

        if (result?.enough_images && isCameraOpen) {
            showRecognitionResult(result.employee_id, result.time);

            // Đánh dấu audio greeting sẽ được play, disable interval capture ngay
            if (cameraState.captureInterval) {
                clearInterval(cameraState.captureInterval);
                cameraState.captureInterval = null;
            }

            // Chờ audio greeting play xong mới đóng camera
            const checkAudioEnd = setInterval(() => {
                if (audioPlayedForRecognition) {
                    clearInterval(checkAudioEnd);
                    closeCamera();
                    audioPlayedForRecognition = false; // reset flag
                }
            }, 100);
        }

        return result;
    } catch (err) {
        console.error('sendFrameForRecognition error', err);
        showToast('❌ Recognition error!', true);
        return null;
    }
}

// Mở / đóng camera
function toggleCamera() {
    const cameraModal = document.getElementById("cameraModal");
    const spinner = document.getElementById("loadingSpinner");
    const video = document.getElementById("cameraVideo");

    if (!cameraModal || !video) {
        showToast('❌ Camera elements not found', true);
        return;
    }

    if (isCameraOpen) {
        closeCamera();
        return;
    }

    cameraModal.classList.remove("hidden");
    setTimeout(() => cameraModal.classList.add("show"), 10);
    spinner.style.display = "block";
    video.style.display = "block";

    cameraState.captureInterval = null;
    cameraState.stream = null;

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(mediaStream => {
            cameraState.stream = mediaStream;
            video.srcObject = mediaStream;
            video.play().catch(() => {});
            spinner.style.display = "none";
            isCameraOpen = true;

            const CAPTURE_INTERVAL_MS = 300;
            cameraState.captureInterval = setInterval(() => {
                if (!video.videoWidth || !video.videoHeight) return;

                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');
                if (!ctx) return;
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                canvas.toBlob(blob => {
                    if (blob) sendFrameForRecognition(blob);
                }, 'image/jpeg');
            }, CAPTURE_INTERVAL_MS);
        })
        .catch(err => {
            console.error('getUserMedia error', err);
            showToast('❌ Cannot open camera: ' + (err.message || err), true);
            closeCamera();
        });
}

function closeCamera() {
    const cameraModal = document.getElementById("cameraModal");
    const video = document.getElementById("cameraVideo");
    const spinner = document.getElementById("loadingSpinner");

    if (cameraState.captureInterval) {
        clearInterval(cameraState.captureInterval);
        cameraState.captureInterval = null;
    }

    if (cameraState.stream) {
        try {
            cameraState.stream.getTracks().forEach(track => track.stop());
        } catch (e) {
            console.warn('Error stopping stream tracks', e);
        }
        cameraState.stream = null;
    }

    if (video) {
        try { video.pause(); } catch(e) {}
        try { video.srcObject = null; } catch(e) {}
        video.style.display = "none";
    }

    if (spinner) spinner.style.display = "none";

    if (cameraModal) {
        cameraModal.classList.remove("show");
        setTimeout(() => cameraModal.classList.add("hidden"), 400);
    }

    isCameraOpen = false;

    // Chỉ pause audio bình thường, không dừng audio greeting ưu tiên
    if (currentAudio && !audioPlayedForRecognition) {
        currentAudio.pause();
        currentAudio = null;
    }
    greetingPlayed = false;
    greetingAlreadyTriggered = false;
}

// --- Bucket dropdown và modal tạo bucket ---
document.addEventListener('DOMContentLoaded', () => {
    const bucketSelect = document.getElementById('bucketSelect');

    if (bucketSelect) {
        // Lắng nghe sự thay đổi trong dropdown
        bucketSelect.addEventListener('change', function () {
            const selectedBucket = this.value;
            localStorage.setItem('selectedBucket', selectedBucket);

            fetch('/set_selected_bucket', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ bucket: selectedBucket })
            })
                .then(response => response.json())
                .then(data => console.log('Bucket selected:', data))
                .catch(error => console.error('Error setting bucket:', error));
        });

        // Lấy bucket đã chọn từ localStorage (nếu có)
        const savedBucket = localStorage.getItem('selectedBucket');
        if (savedBucket) {
            bucketSelect.value = savedBucket;
            fetch('/set_selected_bucket', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ bucket: savedBucket })
            })
                .then(response => response.json())
                .then(data => console.log('Bucket updated from localStorage:', data))
                .catch(error => console.error('Error:', error));
        }
    }
});

function openNewBucketModal() {
    const m = document.getElementById('newBucketModal');
    if (m) m.classList.remove('hidden');
}

function closeNewBucketModal() {
    const m = document.getElementById('newBucketModal');
    if (m) m.classList.add('hidden');
}

function createNewBucket() {
    const bucketName = document.getElementById('bucketNameInput')?.value.trim() ?? '';
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

    if (loadingOverlay) loadingOverlay.style.display = 'flex';

    const formData = new FormData();
    formData.append('bucket_name', bucketName);
    formData.append('logo', logoFile);

    fetch('/create_bucket', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (loadingOverlay) loadingOverlay.style.display = 'none';
            if (data && data.success) {
                showToast(`✅ Created bucket '${bucketName}' successfully!`);
                closeNewBucketModal();
                setTimeout(() => window.location.reload(), 1500);
            } else {
                showToast(`⚠️ ${data?.message ?? 'Create bucket failed'}`, true);
            }
        })
        .catch(error => {
            if (loadingOverlay) loadingOverlay.style.display = 'none';
            showToast("❌ Error creating bucket!", true);
            console.error('createNewBucket error', error);
        });
}
