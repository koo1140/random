document.getElementById('login').addEventListener('click', () => {
  chrome.runtime.sendMessage({action: "login"});
});
