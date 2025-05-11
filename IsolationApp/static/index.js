document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("id_file");
    const fileNameDisplay = document.querySelector(".file-name-display");
    const analyzeButton = document.querySelector(".analyze-button");
    const resetButton = document.querySelector(".reset-button");
    const resultsSection = document.querySelector(".results-section");
    const anomalyToggle = document.getElementById("anomalyToggle");

    const statusMessage = document.createElement("p");
    statusMessage.className = "status-message";
    analyzeButton.parentNode.appendChild(statusMessage);

    let anomaliesHTML = {};
    let allRowsHTML = {};
    let anomalyCount = 0;

    fileInput.addEventListener("change", function () {
        fileNameDisplay.textContent = fileInput.files.length > 0
            ? fileInput.files[0].name
            : "Файл не выбран";
        analyzeButton.disabled = fileInput.files.length === 0;
        setStatus("");
    });

    analyzeButton.addEventListener("click", function (e) {
        e.preventDefault();
        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append("file", file);

        setStatus("Статус. Обработка файла...");

        fetch("/", {
            method: "POST",
            body: formData,
            headers: {
                "X-CSRFToken": getCSRFToken()
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                setStatus("Статус.Ошибка: " + data.error);
            } else {
                anomaliesHTML = data.anomalies;
                allRowsHTML = data.all_rows;
                anomalyCount = data.n_anomalies;
                displayResults(anomaliesHTML, "Аномальные данные", anomalyCount);
                setStatus("Статус.Файл успешно обработан");
            }
        })
        .catch(() => {
            setStatus("Статус.Ошибка отправки файла.");
        });
    });

    anomalyToggle.addEventListener("change", function () {
        const isChecked = anomalyToggle.checked;
        if (Object.keys(anomaliesHTML).length === 0) return;
        if (isChecked) {
            displayResults(allRowsHTML, "Все строки");
        } else {
            displayResults(anomaliesHTML, "Аномальные данные", anomalyCount);
        }
    });

    resetButton.addEventListener("click", function () {
        fileInput.value = "";
        fileNameDisplay.textContent = "Файл не выбран";
        analyzeButton.disabled = true;

        fetch("/reset-file/", {
            method: "POST",
            headers: { "X-CSRFToken": getCSRFToken() }
        });

        resultsSection.innerHTML = "";
        setStatus("");
    });

    function displayResults(groupedTables, title, count = null) {
        resultsSection.innerHTML = `<h2>${title}:</h2>`;

        for (const player in groupedTables) {
            const tableHTML = groupedTables[player];
            const playerSection = document.createElement("div");
            playerSection.classList.add("player-section");
            playerSection.innerHTML = `
                <h3>Игрок: ${player}</h3>
                <div class="table-wrapper">${tableHTML}</div>
            `;
            resultsSection.appendChild(playerSection);
        }

        if (count !== null) {
            const summary = document.createElement("p");
            summary.innerHTML = `<strong>Количество аномальных строк:</strong> ${count}`;
            resultsSection.appendChild(summary);
        }
    }

    function setStatus(message) {
        statusMessage.textContent = message;
    }

    function getCSRFToken() {
        return document.querySelector('[name=csrfmiddlewaretoken]').value;
    }

    const infoButton = document.querySelector(".info-button");
    const infoModal = document.getElementById("infoModal");
    const closeButton = document.querySelector(".close-button");

    infoButton.addEventListener("click", function () {
        infoModal.style.display = "block";
    });

    closeButton.addEventListener("click", function () {
        infoModal.style.display = "none";
    });

    // Закрыть модальное окно, если пользователь кликает вне области окна
    window.addEventListener("click", function (e) {
        if (e.target === infoModal) {
            infoModal.style.display = "none";
        }
    });
});
