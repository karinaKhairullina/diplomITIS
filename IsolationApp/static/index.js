function updateFileName() {
    const input = document.getElementById('file');
    const fileNameSpan = document.getElementById('file-name');
    if (input.files.length > 0) {
        const fileName = input.files[0].name;
        fileNameSpan.textContent = fileName;
        localStorage.setItem('fileName', fileName);
    } else {
        fileNameSpan.textContent = "Файл не выбран";
    }
}

document.addEventListener("DOMContentLoaded", function () {
    const storedFileName = localStorage.getItem('fileName');
    if (storedFileName) {
        document.getElementById('file-name').textContent = storedFileName;
    }
});

function resetPage() {
    localStorage.removeItem('fileName');
    window.location.href = "/";

    const fileNameSpan = document.getElementById('file-name');
    fileNameSpan.textContent = "Файл не выбран";

    const fileInput = document.getElementById('file');
    fileInput.value = "";

    const resultsSection = document.querySelector('.results');
    if (resultsSection) {
        resultsSection.innerHTML = '';
    }

    const missingFeaturesSection = document.getElementById('missing-features');
    if (missingFeaturesSection) {
        missingFeaturesSection.innerHTML = '';
    }
}


function openModal() {
    document.getElementById('modal').style.display = 'block';
}

function closeModal() {
    document.getElementById('modal').style.display = 'none';
}
