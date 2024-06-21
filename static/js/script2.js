document
  .getElementById("uploadForm")
  .addEventListener("submit", function (event) {
    event.preventDefault();
    var formData = new FormData(this);

    fetch("/upload-face", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        displayResult(data);
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  });

function displayResult(data) {
  var resultDiv = document.getElementById("result");
  resultDiv.innerHTML = "";

  if (data.success) {
    if (data.labels && data.labels.length > 0) {
      resultDiv.innerHTML = "Labels: " + data.labels.join(", ");
    } else {
      resultDiv.innerHTML = "No match found";
    }

    if (data.image_path) {
      var img = document.createElement("img");
      img.src = data.image_path;
      resultDiv.appendChild(img);
    }
  } else {
    resultDiv.innerHTML = "Error: " + data.error;
  }
}
