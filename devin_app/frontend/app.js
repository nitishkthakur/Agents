const API_BASE_URL = 'http://localhost:8000';

let currentConversationId = null;
let currentModel = null;
let uploadedFileData = null;
let isStreaming = false;

marked.setOptions({
    highlight: function(code, lang) {
        if (lang && hljs.getLanguage(lang)) {
            return hljs.highlight(code, { language: lang }).value;
        }
        return hljs.highlightAuto(code).value;
    },
    breaks: true,
    gfm: true
});

async function loadModels() {
    try {
        const response = await fetch(`${API_BASE_URL}/models`);
        const data = await response.json();
        
        const select = document.getElementById('model-select');
        select.innerHTML = '';
        
        data.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = model.name;
            select.appendChild(option);
        });
        
        select.value = data.default_model;
        currentModel = data.default_model;
        updateCurrentModelDisplay();
        
        select.addEventListener('change', (e) => {
            currentModel = e.target.value;
            updateCurrentModelDisplay();
        });
    } catch (error) {
        console.error('Failed to load models:', error);
        document.getElementById('current-model').textContent = 'Failed to load models';
    }
}

function updateCurrentModelDisplay() {
    const select = document.getElementById('model-select');
    const selectedOption = select.options[select.selectedIndex];
    document.getElementById('current-model').textContent = `Model: ${selectedOption ? selectedOption.textContent : 'Unknown'}`;
}

function startNewChat() {
    currentConversationId = null;
    uploadedFileData = null;
    
    const messagesContainer = document.getElementById('messages-container');
    messagesContainer.innerHTML = `
        <div class="welcome-message">
            <h2>Welcome to Devin Agent</h2>
            <p>Start a conversation by typing a message below. You can:</p>
            <ul>
                <li>Ask questions and get AI-powered responses</li>
                <li>Upload PDF files for analysis</li>
                <li>Enable web search for current information</li>
                <li>Save and manage artifacts</li>
            </ul>
        </div>
    `;
    
    removeUploadedFile();
}

function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
}

async function sendMessage() {
    const input = document.getElementById('message-input');
    const message = input.value.trim();
    
    if (!message && !uploadedFileData) return;
    if (isStreaming) return;
    
    const webSearchEnabled = document.getElementById('web-search-toggle').checked;
    
    let fullMessage = message;
    if (uploadedFileData) {
        fullMessage = `[Uploaded PDF: ${uploadedFileData.filename}]\n\nText content:\n${uploadedFileData.text_content}\n\n---\n\nUser message: ${message}`;
    }
    
    input.value = '';
    input.style.height = 'auto';
    
    removeWelcomeMessage();
    addMessage('user', message || `[Uploaded PDF: ${uploadedFileData.filename}]`);
    
    if (uploadedFileData && uploadedFileData.images && uploadedFileData.images.length > 0) {
        addPdfImages(uploadedFileData.images);
    }
    
    const assistantMessageId = addMessage('assistant', '', true);
    
    isStreaming = true;
    updateSendButton();
    
    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: fullMessage,
                model_id: currentModel,
                conversation_id: currentConversationId,
                web_search_enabled: webSearchEnabled
            })
        });
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullContent = '';
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        
                        if (data.type === 'content') {
                            fullContent += data.content;
                            updateMessageContent(assistantMessageId, fullContent);
                        } else if (data.type === 'tool_start') {
                            showToolIndicator(assistantMessageId, data.tool, true);
                        } else if (data.type === 'tool_end') {
                            showToolIndicator(assistantMessageId, data.tool, false);
                        } else if (data.type === 'done') {
                            currentConversationId = data.conversation_id;
                        } else if (data.type === 'error') {
                            updateMessageContent(assistantMessageId, `Error: ${data.error}`);
                        }
                    } catch (e) {
                        console.error('Failed to parse SSE data:', e);
                    }
                }
            }
        }
        
        removeLoadingIndicator(assistantMessageId);
        
    } catch (error) {
        console.error('Failed to send message:', error);
        updateMessageContent(assistantMessageId, `Error: ${error.message}`);
    } finally {
        isStreaming = false;
        updateSendButton();
        uploadedFileData = null;
        removeUploadedFile();
    }
}

function removeWelcomeMessage() {
    const welcome = document.querySelector('.welcome-message');
    if (welcome) {
        welcome.remove();
    }
}

function addMessage(role, content, isLoading = false) {
    const messagesContainer = document.getElementById('messages-container');
    const messageId = 'msg-' + Date.now();
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    messageDiv.id = messageId;
    
    const headerDiv = document.createElement('div');
    headerDiv.className = 'message-header';
    headerDiv.textContent = role === 'user' ? 'You' : 'Assistant';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (isLoading) {
        contentDiv.innerHTML = `
            <div class="loading-indicator">
                <span>Thinking</span>
                <div class="loading-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
    } else {
        contentDiv.innerHTML = renderMarkdown(content);
    }
    
    messageDiv.appendChild(headerDiv);
    messageDiv.appendChild(contentDiv);
    messagesContainer.appendChild(messageDiv);
    
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    return messageId;
}

function updateMessageContent(messageId, content) {
    const messageDiv = document.getElementById(messageId);
    if (!messageDiv) return;
    
    const contentDiv = messageDiv.querySelector('.message-content');
    if (!contentDiv) return;
    
    const loadingIndicator = contentDiv.querySelector('.loading-indicator');
    if (loadingIndicator) {
        loadingIndicator.remove();
    }
    
    contentDiv.innerHTML = renderMarkdown(content);
    
    const messagesContainer = document.getElementById('messages-container');
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function removeLoadingIndicator(messageId) {
    const messageDiv = document.getElementById(messageId);
    if (!messageDiv) return;
    
    const loadingIndicator = messageDiv.querySelector('.loading-indicator');
    if (loadingIndicator) {
        loadingIndicator.remove();
    }
}

function showToolIndicator(messageId, toolName, isActive) {
    const messageDiv = document.getElementById(messageId);
    if (!messageDiv) return;
    
    const contentDiv = messageDiv.querySelector('.message-content');
    if (!contentDiv) return;
    
    let toolIndicator = contentDiv.querySelector(`.tool-indicator[data-tool="${toolName}"]`);
    
    if (isActive) {
        if (!toolIndicator) {
            toolIndicator = document.createElement('div');
            toolIndicator.className = 'tool-indicator active';
            toolIndicator.setAttribute('data-tool', toolName);
            toolIndicator.textContent = `Using: ${toolName}...`;
            
            const loadingIndicator = contentDiv.querySelector('.loading-indicator');
            if (loadingIndicator) {
                contentDiv.insertBefore(toolIndicator, loadingIndicator);
            } else {
                contentDiv.insertBefore(toolIndicator, contentDiv.firstChild);
            }
        }
    } else {
        if (toolIndicator) {
            toolIndicator.className = 'tool-indicator complete';
            toolIndicator.textContent = `Used: ${toolName}`;
        }
    }
}

function renderMarkdown(content) {
    if (!content) return '';
    
    let html = marked.parse(content);
    
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = html;
    
    renderMathInElement(tempDiv, {
        delimiters: [
            {left: '$$', right: '$$', display: true},
            {left: '$', right: '$', display: false},
            {left: '\\[', right: '\\]', display: true},
            {left: '\\(', right: '\\)', display: false}
        ],
        throwOnError: false
    });
    
    return tempDiv.innerHTML;
}

function addPdfImages(images) {
    const messagesContainer = document.getElementById('messages-container');
    const lastUserMessage = messagesContainer.querySelector('.message.user:last-of-type');
    
    if (lastUserMessage && images.length > 0) {
        const imagesDiv = document.createElement('div');
        imagesDiv.className = 'pdf-images';
        
        images.slice(0, 5).forEach(img => {
            const imgEl = document.createElement('img');
            imgEl.className = 'pdf-image-thumb';
            imgEl.src = `data:image/png;base64,${img.data}`;
            imgEl.alt = `Page ${img.page}`;
            imgEl.title = `Page ${img.page}`;
            imagesDiv.appendChild(imgEl);
        });
        
        if (images.length > 5) {
            const moreSpan = document.createElement('span');
            moreSpan.style.color = 'var(--text-muted)';
            moreSpan.style.fontSize = '0.75rem';
            moreSpan.style.alignSelf = 'center';
            moreSpan.textContent = `+${images.length - 5} more pages`;
            imagesDiv.appendChild(moreSpan);
        }
        
        lastUserMessage.appendChild(imagesDiv);
    }
}

async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        alert('Only PDF files are supported');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        document.getElementById('file-name').textContent = 'Uploading...';
        document.getElementById('uploaded-file').style.display = 'inline-flex';
        
        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Upload failed');
        }
        
        uploadedFileData = await response.json();
        document.getElementById('file-name').textContent = uploadedFileData.filename;
        
    } catch (error) {
        console.error('Failed to upload file:', error);
        alert('Failed to upload file: ' + error.message);
        removeUploadedFile();
    }
}

function removeUploadedFile() {
    uploadedFileData = null;
    document.getElementById('uploaded-file').style.display = 'none';
    document.getElementById('file-name').textContent = '';
    document.getElementById('file-input').value = '';
}

function updateSendButton() {
    const sendBtn = document.getElementById('send-btn');
    sendBtn.disabled = isStreaming;
}

async function downloadConversation() {
    if (!currentConversationId) {
        alert('No conversation to download');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/download-pdf`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                conversation_id: currentConversationId
            })
        });
        
        if (!response.ok) {
            throw new Error('Download failed');
        }
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `conversation_${currentConversationId.slice(0, 8)}.pdf`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
    } catch (error) {
        console.error('Failed to download conversation:', error);
        alert('Failed to download conversation: ' + error.message);
    }
}

async function viewArtifacts() {
    const modal = document.getElementById('artifacts-modal');
    const artifactsList = document.getElementById('artifacts-list');
    
    modal.classList.add('active');
    artifactsList.innerHTML = 'Loading artifacts...';
    
    try {
        const response = await fetch(`${API_BASE_URL}/artifacts`);
        const data = await response.json();
        
        if (data.artifacts.length === 0) {
            artifactsList.innerHTML = '<p style="color: var(--text-muted);">No artifacts found.</p>';
            return;
        }
        
        artifactsList.innerHTML = '';
        data.artifacts.forEach(artifact => {
            const item = document.createElement('div');
            item.className = 'artifact-item';
            item.onclick = () => viewArtifact(artifact.name);
            
            item.innerHTML = `
                <span class="artifact-name">${artifact.name}</span>
                <span class="artifact-meta">${formatFileSize(artifact.size)}</span>
            `;
            
            artifactsList.appendChild(item);
        });
        
    } catch (error) {
        console.error('Failed to load artifacts:', error);
        artifactsList.innerHTML = '<p style="color: var(--error);">Failed to load artifacts.</p>';
    }
}

async function viewArtifact(filename) {
    closeArtifactsModal();
    
    const modal = document.getElementById('artifact-view-modal');
    const title = document.getElementById('artifact-view-title');
    const content = document.getElementById('artifact-content');
    
    modal.classList.add('active');
    title.textContent = filename;
    content.textContent = 'Loading...';
    
    try {
        const response = await fetch(`${API_BASE_URL}/artifacts/${encodeURIComponent(filename)}`);
        const data = await response.json();
        content.textContent = data.content;
        
    } catch (error) {
        console.error('Failed to load artifact:', error);
        content.textContent = 'Failed to load artifact.';
    }
}

function closeArtifactsModal() {
    document.getElementById('artifacts-modal').classList.remove('active');
}

function closeArtifactViewModal() {
    document.getElementById('artifact-view-modal').classList.remove('active');
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

document.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal')) {
        e.target.classList.remove('active');
    }
});

document.addEventListener('DOMContentLoaded', () => {
    loadModels();
});
