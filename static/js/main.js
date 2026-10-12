/**
 * Plant Leaf Disease Detection - Frontend Interactive Logic
 * Handles drag-and-drop, image preview, AJAX inference, and dynamic results display.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const dropContent = document.getElementById("drop-content");
    const previewContainer = document.getElementById("preview-container");
    const imagePreview = document.getElementById("image-preview");
    const btnRemoveFile = document.getElementById("btn-remove-file");
    const previewFilename = document.getElementById("preview-filename");

    const btnAnalyze = document.getElementById("btn-analyze");
    const btnText = btnAnalyze.querySelector(".btn-text");
    const loadingSpinner = document.getElementById("loading-spinner");

    const emptyState = document.getElementById("empty-state");
    const resultsContent = document.getElementById("results-content");

    const conditionBadge = document.getElementById("condition-badge");
    const predictedName = document.getElementById("predicted-name");
    const cropCategory = document.getElementById("crop-category");
    const confidenceValue = document.getElementById("confidence-value");
    const confidenceFill = document.getElementById("confidence-fill");
    const diseaseDescription = document.getElementById("disease-description");
    const preventionList = document.getElementById("prevention-list");
    const candidatesList = document.getElementById("candidates-list");

    const sampleChips = document.querySelectorAll(".sample-chip");

    let currentFile = null;
    let selectedSample = null;

    // Allowed extensions
    const ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"];
    const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

    // Click on dropzone opens file dialog
    dropZone.addEventListener("click", (e) => {
        if (e.target !== btnRemoveFile && !btnRemoveFile.contains(e.target)) {
            fileInput.click();
        }
    });

    // File Input change
    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    // Drag & Drop Events
    ["dragenter", "dragover"].forEach((eventName) => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("dragover");
        });
    });

    dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        if (dt.files && dt.files[0]) {
            handleSelectedFile(dt.files[0]);
        }
    });

    // Remove file button
    btnRemoveFile.addEventListener("click", (e) => {
        e.stopPropagation();
        resetUploadState();
    });

    // Sample chip buttons
    sampleChips.forEach((chip) => {
        chip.addEventListener("click", (e) => {
            e.stopPropagation();
            const sampleName = chip.getAttribute("data-sample");
            handleSampleSelection(sampleName, chip.innerText.trim());
        });
    });

    function handleSelectedFile(file) {
        const ext = file.name.split(".").pop().toLowerCase();
        if (!ALLOWED_EXTENSIONS.includes(ext)) {
            alert(`Invalid file type: .${ext}. Please select a JPG, JPEG, or PNG image.`);
            return;
        }

        if (file.size > MAX_FILE_SIZE) {
            alert("File is too large. Maximum allowed size is 10 MB.");
            return;
        }

        currentFile = file;
        selectedSample = null;

        // Show live preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewFilename.innerText = file.name;
            dropContent.style.display = "none";
            previewContainer.style.display = "flex";
            btnAnalyze.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function handleSampleSelection(sampleFilename, label) {
        currentFile = null;
        selectedSample = sampleFilename;

        // Set sample preview URL
        imagePreview.src = `/static/samples/${sampleFilename}`;
        previewFilename.innerText = `${label} (${sampleFilename})`;
        dropContent.style.display = "none";
        previewContainer.style.display = "flex";
        btnAnalyze.disabled = false;
    }

    function resetUploadState() {
        currentFile = null;
        selectedSample = null;
        fileInput.value = "";
        imagePreview.src = "";
        previewFilename.innerText = "";
        previewContainer.style.display = "none";
        dropContent.style.display = "flex";
        btnAnalyze.disabled = true;
    }

    // Submit Analysis to /predict
    btnAnalyze.addEventListener("click", async () => {
        if (!currentFile && !selectedSample) {
            return;
        }

        setLoading(true);

        const formData = new FormData();
        if (currentFile) {
            formData.append("file", currentFile);
        } else if (selectedSample) {
            formData.append("sample_image", selectedSample);
        }

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "X-Requested-With": "XMLHttpRequest",
                    "Accept": "application/json"
                },
                body: formData
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(data.error || "Prediction request failed.");
            }

            displayResults(data);
        } catch (error) {
            alert(`Analysis Error: ${error.message}`);
        } finally {
            setLoading(false);
        }
    });

    function setLoading(isLoading) {
        if (isLoading) {
            btnAnalyze.disabled = true;
            btnText.style.display = "none";
            loadingSpinner.style.display = "inline-block";
        } else {
            btnAnalyze.disabled = false;
            btnText.style.display = "inline";
            loadingSpinner.style.display = "none";
        }
    }

    function displayResults(data) {
        emptyState.style.display = "none";
        resultsContent.style.display = "block";

        // Condition Badge
        const isHealthy = data.condition.toLowerCase() === "healthy";
        conditionBadge.className = `condition-badge ${isHealthy ? "healthy" : "diseased"}`;
        conditionBadge.innerText = data.condition.toUpperCase();

        // Disease & Crop
        predictedName.innerText = data.display_name || data.raw_class;
        cropCategory.innerText = `Crop: ${data.crop} • Condition: ${data.condition}`;

        // Confidence
        confidenceValue.innerText = `${data.confidence}%`;
        confidenceFill.style.width = "0%";
        setTimeout(() => {
            confidenceFill.style.width = `${Math.min(data.confidence, 100)}%`;
        }, 50);

        // Clinical Description
        diseaseDescription.innerText = data.description || "No description provided.";

        // Prevention List
        preventionList.innerHTML = "";
        if (data.prevention && data.prevention.length > 0) {
            data.prevention.forEach((tip) => {
                const li = document.createElement("li");
                li.innerText = tip;
                preventionList.appendChild(li);
            });
        } else {
            const li = document.createElement("li");
            li.innerText = "Follow standard crop management guidelines and monitor weekly.";
            preventionList.appendChild(li);
        }

        // Top Alternative Candidates
        candidatesList.innerHTML = "";
        if (data.top_predictions && data.top_predictions.length > 0) {
            data.top_predictions.forEach((cand) => {
                const item = document.createElement("div");
                item.className = "candidate-item";
                item.innerHTML = `
                    <span class="candidate-name">${cand.display_name}</span>
                    <span class="candidate-prob">${cand.confidence}%</span>
                `;
                candidatesList.appendChild(item);
            });
        }
    }
});
