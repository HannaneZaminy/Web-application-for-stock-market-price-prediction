function showPopup() {
    var popup = document.getElementById("popup");
    popup.style.display = "block";
    setTimeout(function() {
        popup.style.display = "none";
    }, 5000); // 5000 میلی‌ثانیه معادل 5 ثانیه
}

function closePopup() {
    var popup = document.getElementById("popup");
    popup.style.display = "none";
}