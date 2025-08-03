document.addEventListener('DOMContentLoaded', () => {
    // --- GENERAL UTILITY FUNCTIONS (from embedded JS) ---
    function updateDateTime() {
        const now = new Date();
        document.getElementById('currentDate').textContent = now.toLocaleDateString('en-CA');
        document.getElementById('currentTime').textContent = now.toLocaleTimeString('en-GB');
    }

    function showMessage(message, type = 'info') {
        const msgBox = document.getElementById('messageBox');
        if (msgBox) {
            msgBox.textContent = message;
            msgBox.className = 'show';
            if (type === 'error') {
                msgBox.style.backgroundColor = '#dc2626';
            } else if (type === 'success') {
                msgBox.style.backgroundColor = '#16a34a';
            } else {
                msgBox.style.backgroundColor = '#2563eb';
            }
            setTimeout(() => {
                msgBox.className = '';
            }, 3000);
        }
    }

    // Make showMessage available globally
    window.showMessage = showMessage;

    // --- RESIZABLE INPUT FUNCTIONALITY (from embedded JS) ---
    let isResizing = false;
    let currentResizeContainer = null;
    let startX = 0;
    let startWidth = 0;

    function makeInputResizable(input) {
        const container = document.createElement('div');
        container.className = 'resizable-input-container';
        
        const handle = document.createElement('div');
        handle.className = 'resize-handle';
        
        input.parentNode.insertBefore(container, input);
        container.appendChild(input);
        container.appendChild(handle);
        
        input.classList.add('resizable-input');
        
        handle.addEventListener('mousedown', (e) => {
            e.preventDefault();
            isResizing = true;
            currentResizeContainer = container;
            startX = e.clientX;
            startWidth = parseInt(window.getComputedStyle(container).width, 10);
            handle.classList.add('dragging');
            document.body.style.cursor = 'ew-resize';
            document.body.style.userSelect = 'none';
        });
    }

    document.addEventListener('mousemove', (e) => {
        if (!isResizing || !currentResizeContainer) return;
        
        const newWidth = startWidth + (e.clientX - startX);
        if (newWidth >= 60) { // Minimum width
            currentResizeContainer.style.width = newWidth + 'px';
        }
    });

    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            
            if (currentResizeContainer) {
                const handle = currentResizeContainer.querySelector('.resize-handle');
                if (handle) handle.classList.remove('dragging');
            }
            currentResizeContainer = null;
        }
    });

    // --- INITIALIZATION (from embedded JS) ---
    updateDateTime();
    setInterval(updateDateTime, 1000);
    
    // Make component table inputs resizable
    document.querySelectorAll('.resizable-component-input').forEach(input => {
        makeInputResizable(input);
    });
    
    console.log('Application initialized successfully!');
    showMessage('Application loaded successfully!', 'success');

});

/**
 * Form Data Persistence - Saves and loads form data using localStorage.
 */

// Use a descriptive key for your localStorage data.
const storageKey = 'componentFormData';

/**
 * Loads saved data from localStorage and populates the form fields.
 */
function loadSavedData() {
    // Find the form element (it doesn't have an ID, so find by tag)
    const form = document.querySelector('form[action="/run_lp"]');
    if (!form) {
        console.error("Form not found.");
        return;
    }

    try {
        // Retrieve the saved data as a JSON string.
        const savedData = localStorage.getItem(storageKey);

        if (savedData) {
            // Parse the JSON string back into a JavaScript object.
            const data = JSON.parse(savedData);

            // Loop through the saved data and populate the corresponding form inputs by name
            for (const inputName in data) {
                const input = form.querySelector(`input[name="${inputName}"]`);
                if (input) {
                    input.value = data[inputName];
                    console.log(`Restored ${inputName}: ${data[inputName]}`);
                }
            }
            
            showMessage('Previous data restored successfully!', 'success');
        }
    } catch (e) {
        // Log an error if localStorage parsing fails.
        console.error("Error loading data from localStorage:", e);
        // Optionally clear the corrupted data.
        localStorage.removeItem(storageKey);
    }
}
/**
 * Saves the current form data to localStorage.
 */
function saveData() {
    const form = document.querySelector('form[action="/run_lp"]');
    if (!form) {
        return;
    }

    // Create a new FormData object from the form.
    const formData = new FormData(form);
    const dataObject = {};

    // Convert the FormData object to a regular JavaScript object.
    formData.forEach((value, key) => {
        dataObject[key] = value;
    });

    try {
        // Save the JavaScript object to localStorage as a JSON string.
        localStorage.setItem(storageKey, JSON.stringify(dataObject));
        console.log('Form data saved to localStorage');
    } catch (e) {
        console.error("Error saving data to localStorage:", e);
    }
}

// Add an 'input' event listener to the form to save data in real-time as the user types.
document.addEventListener('DOMContentLoaded', () => {
    const form = document.querySelector('form[action="/run_lp"]');
    if (form) {
        form.addEventListener('input', saveData);
        
        // Also, listen for the form submission
        form.addEventListener('submit', (event) => {
            // Don't prevent default - let the form submit to Flask
            console.log("Form submitted. Values have been saved.");
            saveData(); // Save one final time before submitting
        });
    }
});

// Call this function when the page loads to restore the saved data.
document.addEventListener('DOMContentLoaded', loadSavedData);

  
        // --- RESIZABLE INPUT FUNCTIONALITY ---
        let isResizing = false;
        let currentResizeContainer = null;
        let startX = 0;
        let startWidth = 0;

        function makeInputResizable(input) {
            const container = document.createElement('div');
            container.className = 'resizable-input-container';
            
            const handle = document.createElement('div');
            handle.className = 'resize-handle';
            
            input.parentNode.insertBefore(container, input);
            container.appendChild(input);
            container.appendChild(handle);
            
            input.classList.add('resizable-input');
            
            handle.addEventListener('mousedown', (e) => {
                e.preventDefault();
                isResizing = true;
                currentResizeContainer = container;
                startX = e.clientX;
                startWidth = parseInt(window.getComputedStyle(container).width, 10);
                handle.classList.add('dragging');
                document.body.style.cursor = 'ew-resize';
                document.body.style.userSelect = 'none';
            });
        }

        document.addEventListener('mousemove', (e) => {
            if (!isResizing || !currentResizeContainer) return;
            
            const newWidth = startWidth + (e.clientX - startX);
            if (newWidth >= 60) { // Minimum width
                currentResizeContainer.style.width = newWidth + 'px';
            }
        });

        document.addEventListener('mouseup', () => {
            if (isResizing) {
                isResizing = false;
                document.body.style.cursor = '';
                document.body.style.userSelect = '';
                
                if (currentResizeContainer) {
                    const handle = currentResizeContainer.querySelector('.resize-handle');
                    if (handle) handle.classList.remove('dragging');
                }
                currentResizeContainer = null;
            }
        });

        // --- GENERAL UTILITY FUNCTIONS ---
        function updateDateTime() {
            const now = new Date();
            document.getElementById('currentDate').textContent = now.toLocaleDateString('en-CA');
            document.getElementById('currentTime').textContent = now.toLocaleTimeString('en-GB');
        }

        function showMessage(message, type = 'info') {
            const msgBox = document.getElementById('messageBox');
            msgBox.textContent = message;
            msgBox.className = 'show';
            if (type === 'error') {
                msgBox.style.backgroundColor = '#dc2626';
            } else if (type === 'success') {
                msgBox.style.backgroundColor = '#16a34a';
            } else {
                msgBox.style.backgroundColor = '#2563eb';
            }
            setTimeout(() => {
                msgBox.className = '';
            }, 3000);
        }

        // --- INITIALIZATION ---
        window.addEventListener('DOMContentLoaded', (event) => {
            updateDateTime();
            setInterval(updateDateTime, 1000);
            
            // Make component table inputs resizable
            document.querySelectorAll('.resizable-component-input').forEach(input => {
                makeInputResizable(input);
            });
            
            console.log('Application initialized successfully!');
            showMessage('Application loaded successfully!', 'success');
        });
    
    
       document.addEventListener('DOMContentLoaded', function() {
    const playButton = document.getElementById('playMusicButton');
    const music = document.getElementById('africanMusic');

    // This is the correct path for your music folder.
    const musicTracks = [
        'static/music/African Journey.mp3',
        'static/music/Batacumbele.mp3',
        'static/music/Drums.Chant.mp3',
        'static/music/Flute.Drums.mp3'
    ];

    function playRandomMusic() {
        const randomIndex = Math.floor(Math.random() * musicTracks.length);
        const selectedTrack = musicTracks[randomIndex];
        music.src = selectedTrack;
        music.play();
        playButton.textContent = 'Pause African Music ⏸️';
    }

    playButton.addEventListener('click', () => {
        if (music.paused) {
            playRandomMusic();
        } else {
            music.pause();
            playButton.textContent = 'Play African Music 🎵';
        }
    });

    music.addEventListener('ended', () => {
        playRandomMusic();
    });
});