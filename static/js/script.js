function recognizeVoice() {
  var audioInput = document.getElementById("audioInput");
  if (!audioInput.files.length) {
    document.getElementById("result").innerText =
      "Please upload an audio file.";
    return;
  }

  var file = audioInput.files[0];
  var formData = new FormData();
  formData.append("audioFile", file);

  fetch("/recognize", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        document.getElementById("result").innerText = data.result;
      } else {
        document.getElementById("result").innerText = data.error;
      }
    })
    .catch((error) => {
      console.error("Error:", error);
      document.getElementById("result").innerText = "An error occurred.";
    });
}
