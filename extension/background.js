'use strict';

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type !== 'fcOpenOptions') return undefined;

  const createOptions = {
    url: chrome.runtime.getURL('options.html'),
    active: true,
  };
  if (Number.isInteger(sender.tab?.windowId)) createOptions.windowId = sender.tab.windowId;
  if (Number.isInteger(sender.tab?.index)) createOptions.index = sender.tab.index + 1;

  chrome.tabs.create(createOptions)
    .then(tab => sendResponse({ ok: true, tabId: tab.id }))
    .catch(error => sendResponse({
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    }));
  return true;
});
