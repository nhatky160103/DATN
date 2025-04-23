function toggleCamera() {
    const cameraModal = document.getElementById("cameraModal");
    const cameraFeed = document.getElementById("cameraFeed");
    const spinner = document.getElementById("loadingSpinner");

    cameraModal.classList.remove("hidden");
    setTimeout(() => cameraModal.classList.add("show"), 10); // Trigger animation

    // Hiển thị loading trong vài giây đầu
    spinner.style.display = "block";
    cameraFeed.style.display = "none";

    // Khi ảnh bắt đầu hiển thị, tắt loading
    cameraFeed.onload = () => {
        spinner.style.display = "none";
        cameraFeed.style.display = "block";
    };

    // Đảm bảo luôn reload stream mới (tránh cache)
    cameraFeed.src = "/video_feed?" + new Date().getTime();
}
function closeCamera() {
    const cameraModal = document.getElementById("cameraModal");
    const cameraFeed = document.getElementById("cameraFeed");

    cameraFeed.src = "";  // Ngắt kết nối stream

    cameraModal.classList.remove("show");
    setTimeout(() => {
        cameraModal.classList.add("hidden");
    }, 400); // Khớp với animation
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