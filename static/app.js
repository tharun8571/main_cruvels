document.addEventListener("DOMContentLoaded", () => {
    const statusText = document.getElementById("statusText");
    const statusDot = document.querySelector(".status-dot");
    const modelBadge = document.getElementById("modelBadge");
    const embedBadge = document.getElementById("embedBadge");
    const documentList = document.getElementById("documentList");
    const refreshDocsBtn = document.getElementById("refreshDocsBtn");
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    
    const chatForm = document.getElementById("chatForm");
    const userQuery = document.getElementById("userQuery");
    const chatMessages = document.getElementById("chatMessages");
    const chatViewport = document.getElementById("chatViewport");
    const welcomeScreen = document.getElementById("welcomeScreen");
    const clearChatBtn = document.getElementById("clearChatBtn");
    const clearKbBtn = document.getElementById("clearKbBtn");

    // Initialize API Status Check
    checkHealth();
    fetchDocuments();

    // Event Listeners for Sample Prompts
    document.querySelectorAll(".prompt-card").forEach(card => {
        card.addEventListener("click", () => {
            const promptText = card.getAttribute("data-prompt");
            userQuery.value = promptText;
            submitQuery(promptText);
        });
    });

    // Clear Chat
    clearChatBtn.addEventListener("click", () => {
        chatMessages.innerHTML = "";
        welcomeScreen.style.display = "flex";
    });

    // Auto-resize textarea
    userQuery.addEventListener("input", () => {
        userQuery.style.height = "auto";
        userQuery.style.height = userQuery.scrollHeight + "px";
    });

    // Submit Query via Form
    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const text = userQuery.value.trim();
        if (text) {
            submitQuery(text);
        }
    });

    // Enter Key Handler (Shift+Enter for newline)
    userQuery.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event("submit"));
        }
    });

    // Upload Handlers
    dropZone.addEventListener("click", () => fileInput.click());
    
    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            uploadFile(fileInput.files[0]);
        }
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "#6366f1";
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.style.borderColor = "rgba(99, 102, 241, 0.4)";
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "rgba(99, 102, 241, 0.4)";
        if (e.dataTransfer.files.length > 0) {
            uploadFile(e.dataTransfer.files[0]);
        }
    });

    refreshDocsBtn.addEventListener("click", fetchDocuments);

    clearKbBtn.addEventListener("click", () => {
        if (confirm("⚠️ This will permanently delete all ingested documents and the vector database. Continue?")) {
            clearKnowledgeBase();
        }
    });

    // API Calls
    async function checkHealth() {
        try {
            const res = await fetch("/api/health");
            if (res.ok) {
                const data = await res.json();
                statusDot.classList.remove("connecting");
                statusText.textContent = "API Ready";
                modelBadge.innerHTML = `<i class="fa-solid fa-brain"></i> ${data.llm_provider} (${data.llm_model})`;
                embedBadge.innerHTML = `<i class="fa-solid fa-vector-square"></i> ${data.embedding_model}`;
            }
        } catch (err) {
            statusDot.classList.add("connecting");
            statusText.textContent = "Server Offline";
        }
    }

    async function fetchDocuments() {
        try {
            const res = await fetch("/api/documents");
            if (res.ok) {
                const data = await res.json();
                renderDocuments(data.documents || []);
            }
        } catch (err) {
            console.error("Error fetching documents:", err);
        }
    }

    function renderDocuments(docs) {
        if (docs.length === 0) {
            documentList.innerHTML = '<li class="empty-doc">No documents ingested</li>';
            return;
        }
        documentList.innerHTML = docs.map(d => `
            <li class="doc-item">
                <i class="fa-solid fa-file-pdf"></i>
                <span class="doc-name" title="${d.filename}">${d.filename}</span>
            </li>
        `).join("");
    }

    async function clearKnowledgeBase() {
        clearKbBtn.disabled = true;
        statusText.textContent = "Clearing knowledge base...";
        try {
            const res = await fetch("/api/knowledge-base", { method: "DELETE" });
            if (res.ok) {
                const data = await res.json();
                statusText.textContent = "API Ready";
                fetchDocuments();
                if (data.status === "cleared") {
                    appendSystemNotice("✅ Knowledge base cleared. All documents and vectors removed.");
                } else {
                    appendSystemNotice("⚠️ Partially cleared. Some files may remain.");
                }
            } else {
                statusText.textContent = "API Ready";
                appendSystemNotice("❌ Failed to clear knowledge base.");
            }
        } catch (err) {
            statusText.textContent = "API Ready";
            appendSystemNotice("❌ Network error while clearing knowledge base.");
        } finally {
            clearKbBtn.disabled = false;
        }
    }

    async function uploadFile(file) {
        const formData = new FormData();
        formData.append("file", file);

        statusText.textContent = "Ingesting " + file.name + "...";
        try {
            const res = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });
            if (res.ok) {
                const data = await res.json();
                statusText.textContent = "API Ready";
                fetchDocuments();
                alert(`Uploaded and ingested ${data.filename} (${data.chunks_ingested} chunks)`);
            } else {
                const err = await res.json();
                alert("Upload failed: " + (err.detail || "Error uploading file"));
            }
        } catch (err) {
            alert("Error connecting to server during upload");
        }
    }

    async function submitQuery(question) {
        welcomeScreen.style.display = "none";
        userQuery.value = "";
        userQuery.style.height = "auto";

        // Append User Message
        appendMessage("user", question);

        // Append Loading Indicator
        const loadingId = appendLoading();

        try {
            const res = await fetch("/api/ask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question: question })
            });

            removeLoading(loadingId);

            if (res.ok) {
                const data = await res.json();
                appendMessage("assistant", data.answer, data.sources, data.is_fallback);
            } else {
                const err = await res.json();
                appendMessage("assistant", "⚠️ **Error processing request**: " + (err.detail || "Server error"));
            }
        } catch (err) {
            removeLoading(loadingId);
            appendMessage("assistant", "⚠️ **Network Error**: Unable to reach backend server.");
        }
    }

    function appendMessage(role, text, sources = [], isFallback = false) {
        const row = document.createElement("div");
        row.className = `message-row ${role}`;

        const avatar = document.createElement("div");
        avatar.className = "avatar";
        avatar.innerHTML = role === "user" ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';

        const bubble = document.createElement("div");
        bubble.className = "message-bubble";

        // Parse markdown text using marked library
        if (typeof marked !== "undefined") {
            bubble.innerHTML = marked.parse(text);
        } else {
            bubble.textContent = text;
        }

        // Render sources/citations if present
        if (role === "assistant" && sources && sources.length > 0) {
            const sourcesBox = document.createElement("div");
            sourcesBox.className = "sources-container";
            
            const title = document.createElement("div");
            title.className = "sources-title";
            title.innerHTML = '<i class="fa-solid fa-bookmark"></i> Cited Sources';
            sourcesBox.appendChild(title);

            const pills = document.createElement("div");
            pills.className = "sources-pills";

            const uniqueSources = [...new Set(sources.map(s => `${s.file_name || s.doc_id || 'doc'} (p.${s.page_number || 1})`))];
            uniqueSources.forEach(src => {
                const pill = document.createElement("span");
                pill.className = "source-pill";
                pill.textContent = src;
                pills.appendChild(pill);
            });

            sourcesBox.appendChild(pills);
            bubble.appendChild(sourcesBox);
        }

        // Fallback Badge
        if (role === "assistant" && isFallback) {
            const fallbackBadge = document.createElement("div");
            fallbackBadge.className = "fallback-badge";
            fallbackBadge.innerHTML = '<i class="fa-solid fa-shield-cat"></i> Fallback Triggered (No matching evidence found)';
            bubble.appendChild(fallbackBadge);
        }

        row.appendChild(avatar);
        row.appendChild(bubble);
        chatMessages.appendChild(row);

        // Scroll to bottom
        chatViewport.scrollTop = chatViewport.scrollHeight;
    }

    function appendLoading() {
        const id = "loading-" + Date.now();
        const row = document.createElement("div");
        row.className = "message-row assistant";
        row.id = id;

        const avatar = document.createElement("div");
        avatar.className = "avatar";
        avatar.innerHTML = '<i class="fa-solid fa-robot"></i>';

        const bubble = document.createElement("div");
        bubble.className = "message-bubble";
        bubble.innerHTML = `
            <div class="typing-dots">
                <span></span><span></span><span></span>
            </div>
        `;

        row.appendChild(avatar);
        row.appendChild(bubble);
        chatMessages.appendChild(row);
        chatViewport.scrollTop = chatViewport.scrollHeight;
        return id;
    }

    function removeLoading(id) {
        const el = document.getElementById(id);
        if (el) {
            el.remove();
        }
    }

    function appendSystemNotice(text) {
        // Hide welcome screen if visible
        welcomeScreen.style.display = "none";

        const notice = document.createElement("div");
        notice.className = "system-notice";
        notice.textContent = text;
        chatMessages.appendChild(notice);
        chatViewport.scrollTop = chatViewport.scrollHeight;

        // Auto-fade after 5 seconds
        setTimeout(() => {
            notice.style.opacity = "0";
            setTimeout(() => notice.remove(), 400);
        }, 5000);
    }
});
