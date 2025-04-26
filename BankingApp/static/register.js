const video = document.getElementById("video");
navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
    video.srcObject = stream;
});

document.getElementById("capture").addEventListener("click", async () => {
    const canvas = document.getElementById("canvas");
    const context = canvas.getContext("2d");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL("image/jpeg").split(",")[1];

    const email = document.getElementById("email").value;
    const password = document.getElementById("password").value;

    fetch("/register", {
        method: "POST",
        body: JSON.stringify({ email, password, image: imageData }),
        headers: { "Content-Type": "application/json" }
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
        if (data.message === "User Registered Successfully!") {
            setTimeout(() => {
                window.location.href = "/";  // Redirect to home after 3 seconds
            }, 3000);
        }
    })
    .catch(error => console.error("Error:", error));
});
