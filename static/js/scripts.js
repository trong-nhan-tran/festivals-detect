function validateFileCount(input) {
  const fileCount = input.files.length;
  if (fileCount < 1 || fileCount > 10) {
    alert("Vui lòng chọn từ 1 đến 10 ảnh.");
    input.value = ""; // Reset the input
  }
}

function updateFileCount(input) {
  const fileCount = input.files.length;
  document.getElementById("file-count").textContent =
    "Chọn ảnh (" + fileCount + ")";
  sessionStorage.setItem("fileCount", fileCount); // Store file count
}

// Restore file count on page load
document.addEventListener("DOMContentLoaded", function () {
  const storedFileCount = sessionStorage.getItem("fileCount");
  if (storedFileCount) {
    document.getElementById("file-count").textContent =
      "Chọn hình ảnh (" + storedFileCount + ")";
  }
});
function showSpinner() {
  document.getElementById("loading-spinner").style.display = "block"; // Show the spinner
}