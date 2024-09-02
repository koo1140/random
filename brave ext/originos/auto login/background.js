chrome.action.onClicked.addListener((tab) => {
  chrome.scripting.executeScript({
    target: {tabId: tab.id},
    function: performLogin
  });
});

function performLogin() {
  // You can use JavaScript to interact with the page
  // However, clicking at specific coordinates directly is not straightforward
  // Here’s an example of how you might interact with elements by selector instead
  document.querySelector('input[name="password"]').value = "your_password";
  document.querySelector('button[type="submit"]').click();
}
