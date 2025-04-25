document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("id_file");
    const fileNameDisplay = document.querySelector(".file-name-display");
    const analyzeButton = document.querySelector(".analyze-button");
    const resetButton = document.querySelector(".reset-button");

    // При выборе файла
    fileInput.addEventListener("change", function () {
        if (fileInput.files.length > 0) {
            fileNameDisplay.textContent = fileInput.files[0].name;
            analyzeButton.disabled = false;
        } else {
            fileNameDisplay.textContent = "Файл не выбран";
            analyzeButton.disabled = true;
        }
    });

    // Сброс формы
    resetButton.addEventListener("click", function () {
    fileInput.value = "";
    fileNameDisplay.textContent = "Файл не выбран";
    analyzeButton.disabled = true;

    // Очистить имя файла в сессии через запрос к серверу
    fetch('/reset-file/', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            // Можете обновить UI если нужно, но в основном это просто очистит сессию
        });

    const resultsSection = document.querySelector(".results-section");
    if (resultsSection) resultsSection.innerHTML = "";
});


    // Модальное окно
    const infoButton = document.querySelector(".info-button");
    const modal = document.getElementById("infoModal");
    const closeButton = document.querySelector(".close-button");

    infoButton.addEventListener("click", function (e) {
        e.preventDefault();
        modal.style.display = "flex";
    });

    closeButton.addEventListener("click", function () {
        modal.style.display = "none";
    });

    window.addEventListener("click", function (e) {
        if (e.target === modal) modal.style.display = "none";
    });
});
