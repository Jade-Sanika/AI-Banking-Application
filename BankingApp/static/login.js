document.addEventListener("DOMContentLoaded", async () => {
    const video = document.getElementById("video");

    // Request camera access
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    } catch (error) {
        console.error("Error accessing camera:", error);
        alert("Please allow camera access for login.");
    }
});

document.getElementById("capture").addEventListener("click", async () => {
    const canvas = document.getElementById("canvas");
    const context = canvas.getContext("2d");
    const video = document.getElementById("video");

    // Ensure video is playing before capturing
    if (!video.srcObject) {
        alert("Camera not available. Please refresh and allow access.");
        return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL("image/jpeg").split(",")[1];
    const email = document.getElementById("email").value.trim();
    const password = document.getElementById("password").value.trim();

    if (!email || !password) {
        alert("Please enter email and password.");
        return;
    }

    fetch("/login", {
        method: "POST",
        body: JSON.stringify({ email, password, image: imageData }),
        headers: { "Content-Type": "application/json" }
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
        if (data.success) {
            window.location.href = data.redirect;  // Redirecting to dashboard
        }
    })
    .catch(error => console.error("Error:", error));
});
