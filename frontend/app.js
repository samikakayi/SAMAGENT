(() => {
  "use strict";

  const API = "";
  const REQUEST_TIMEOUT_MS = 16_000;
  const HEALTH_POLL_MS = 15_000;
  const APPROVAL_POLL_MS = 8_000;

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  const dom = {
    app: $("#app"),
    scrim: $("#mobile-scrim"),
    sidebar: $("#sidebar"),
    context: $("#context-pane"),
    feed: $("#chat-feed"),
    chatScroll: $("#chat-scroll"),
    welcome: $("#welcome-card"),
    composer: $("#chat-composer"),
    input: $("#message-input"),
    send: $("#send-button"),
    fileInput: $("#file-input"),
    attachments: $("#attachment-strip"),
    sessionList: $("#session-list"),
    sessionsEmpty: $("#sessions-empty"),
    title: $("#conversation-title"),
    workspaceName: $("#workspace-name"),
    connectionPill: $("#connection-pill"),
    sidebarStatusDot: $("#sidebar-status-dot"),
    sidebarStatusLabel: $("#sidebar-status-label"),
    sidebarProviderLabel: $("#sidebar-provider-label"),
    sidebarModelLabel: $("#sidebar-model-label"),
    sidebarLatencyLabel: $("#sidebar-latency-label"),
    systemMeter: $("#system-meter-fill"),
    permissionModeLabel: $("#permission-mode-label"),
    permissionModeLabels: $$('[data-sam-role="permission-mode-label"]'),
    voiceButton: $("#voice-input-button"),
    voiceWave: $("#voice-wave"),
    voiceStatus: $("#voice-status-text"),
    planLists: $$('[data-sam-role="plan-list"]'),
    planProgressIndicators: $$('[data-sam-role="plan-progress-indicator"]'),
    planProgressLabels: $$('[data-sam-role="plan-progress-label"]'),
    planStatusTitles: $$('[data-sam-role="plan-status-title"]'),
    planStatusCopies: $$('[data-sam-role="plan-status-copy"]'),
    toolActivities: $$('[data-sam-role="tool-activity"]'),
    toolActivityItems: $$('[data-sam-role="tool-activity-items"]'),
    approvalLists: $$('[data-sam-role="approval-list"]'),
    approvalsEmptyStates: $$('[data-sam-role="approvals-empty"]'),
    approvalCount: $("#approval-count"),
    memorySearches: $$('[data-sam-role="memory-search"]'),
    auditFilters: $$('[data-sam-role="audit-filter"]'),
    sidebarApprovalBadge: $("#sidebar-approval-badge"),
    memoryLists: $$('[data-sam-role="memory-list"]'),
    memoryEmptyStates: $$('[data-sam-role="memory-empty"]'),
    memorySearches: $$('[data-sam-role="memory-search"]'),
    auditLists: $$('[data-sam-role="audit-list"]'),
    auditEmptyStates: $$('[data-sam-role="audit-empty"]'),
    auditFilters: $$('[data-sam-role="audit-filter"]'),
    settingsModal: $("#settings-modal"),
    settingsForm: $("#settings-form"),
    settingsStatus: $("#settings-save-status"),
    approvalModal: $("#approval-modal"),
    workspaceModal: $("#workspace-modal"),
    workspaceForm: $("#workspace-form"),
    toastStack: $("#toast-stack"),
  };

  const state = {
    health: null,
    conversations: [],
    conversationId: null,
    messages: [],
    models: [],
    modelsLoaded: false,
    settings: {
      agent_name: "SAM",
      provider: "auto",
      model: "",
      model_mode: "AUTO",
      language: "ckb-IQ",
      temperature: 0.2,
      permission_mode: "guarded",
      voice_input: true,
      voice_output: false,
      voice: "",
      speech_rate: 1,
      voice_mode: "PUSH_TO_TALK",
      voice_language: "ckb-IQ",
      voice_wake_word: "SAM",
      voice_vad_threshold: 0.035,
      voice_silence_ms: 800,
      daily_budget_usd: 2,
      monthly_budget_usd: 30,
      computer_control_enabled: false,
      screen_access_enabled: false,
      memory_enabled: true,
      audit_enabled: true,
      cloud_fallback: false,
      show_plans: true,
      notifications: false,
    },
    workspace: { root: "", entries: [] },
    approvals: [],
    memories: [],
    audit: [],
    plan: [],
    tools: [],
    currentApproval: null,
    pendingFiles: [],
    sending: false,
    abortController: null,
    recognition: null,
    listening: false,
    transcribing: false,
    voiceTurn: false,
    speakingReply: false,
    ttsAudio: null,
    ttsQueue: [],
    ttsBusy: false,
    ttsCursor: 0,
    ttsGeneration: 0,
    ttsPrefetch: null,
    capturePcm: false,
    voicePcm: [],
    voiceSampleRate: 16000,
    voiceProcessor: null,
    voiceMute: null,
    voiceSource: null,
    voiceDeadline: 0,
    mediaStream: null,
    audioContext: null,
    analyser: null,
    vadFrame: 0,
    voiceHeard: false,
    voiceManualStop: false,
    voiceTranscript: "",
    connectivity: "connecting",
    activePanel: "agent",
    memoryQuery: "",
    auditFilter: "all",
  };

  const t = (key, fallback) => (window.SAM_I18N ? window.SAM_I18N.t(key, fallback) : (fallback || key));

  function applyUiLanguage() {
    if (!window.SAM_I18N) return;
    window.SAM_I18N.apply(state.settings.language || "ckb-IQ");
    applyPermissionMode();
  }

  class ApiError extends Error {
    constructor(message, status = 0, payload = null) {
      super(message);
      this.name = "ApiError";
      this.status = status;
      this.payload = payload;
    }
  }

  async function apiFetch(path, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
    const ownController = new AbortController();
    const externalSignal = options.signal;
    const timeout = window.setTimeout(() => ownController.abort("timeout"), timeoutMs);
    const signal = externalSignal ? combineSignals(externalSignal, ownController.signal) : ownController.signal;
    const headers = new Headers(options.headers || {});
    if (!headers.has("Accept")) headers.set("Accept", "application/json");
    if (options.body && !(options.body instanceof FormData) && !headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }

    try {
      const response = await fetch(`${API}${path}`, {
        credentials: "same-origin",
        cache: "no-store",
        ...options,
        headers,
        signal,
      });
      if (!response.ok) {
        let payload = null;
        try {
          payload = await response.json();
        } catch {
          payload = await response.text().catch(() => "");
        }
        const detail = payload?.detail || payload?.error || payload?.message;
        throw new ApiError(detail || `Request failed (${response.status})`, response.status, payload);
      }
      return response;
    } catch (error) {
      if (error?.name === "AbortError") {
        if (externalSignal?.aborted) throw error;
        throw new ApiError("The local service took too long to respond.", 0);
      }
      if (error instanceof ApiError) throw error;
      throw new ApiError("Could not reach the local SAM service.", 0, error);
    } finally {
      window.clearTimeout(timeout);
    }
  }

  function combineSignals(...signals) {
    if (typeof AbortSignal.any === "function") return AbortSignal.any(signals);
    const controller = new AbortController();
    for (const signal of signals) {
      if (signal.aborted) controller.abort(signal.reason);
      else signal.addEventListener("abort", () => controller.abort(signal.reason), { once: true });
    }
    return controller.signal;
  }

  async function getJson(path, options, timeoutMs) {
    const response = await apiFetch(path, options, timeoutMs);
    if (response.status === 204) return null;
    return response.json();
  }

  function asArray(payload, ...keys) {
    if (Array.isArray(payload)) return payload;
    for (const key of keys) {
      if (Array.isArray(payload?.[key])) return payload[key];
    }
    return [];
  }

  function safeText(value, fallback = "") {
    if (value === null || value === undefined) return fallback;
    if (typeof value === "string") return value;
    if (typeof value === "number" || typeof value === "boolean") return String(value);
    try {
      return JSON.stringify(value);
    } catch {
      return fallback;
    }
  }

  function escapeHtml(value) {
    return safeText(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function sanitizeLink(url) {
    const text = safeText(url).trim();
    if (/^(https?:\/\/|mailto:|\/)/i.test(text)) return text;
    return "#";
  }

  function renderMarkdown(source) {
    let text = escapeHtml(source || "");
    const codeBlocks = [];
    text = text.replace(/```([\w.+-]*)\n?([\s\S]*?)```/g, (_, language, code) => {
      const token = `@@SAM_CODE_${codeBlocks.length}@@`;
      const lang = language ? ` data-language="${escapeHtml(language)}"` : "";
      codeBlocks.push(`<pre${lang}><code>${code.trim()}</code></pre>`);
      return token;
    });

    text = text
      .replace(/`([^`\n]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
      .replace(/__([^_\n]+)__/g, "<strong>$1</strong>")
      .replace(/\*([^*\n]+)\*/g, "<em>$1</em>")
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label, url) => {
        const href = escapeHtml(sanitizeLink(url));
        return `<a href="${href}" target="_blank" rel="noreferrer noopener">${label}</a>`;
      });

    const lines = text.split("\n");
    const output = [];
    let listType = null;

    const closeList = () => {
      if (listType) output.push(`</${listType}>`);
      listType = null;
    };

    for (const rawLine of lines) {
      const line = rawLine.trimEnd();
      if (/^@@SAM_CODE_\d+@@$/.test(line.trim())) {
        closeList();
        output.push(line.trim());
        continue;
      }
      const unordered = line.match(/^\s*[-*]\s+(.+)$/);
      const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
      if (unordered || ordered) {
        const nextType = ordered ? "ol" : "ul";
        if (listType !== nextType) {
          closeList();
          output.push(`<${nextType}>`);
          listType = nextType;
        }
        output.push(`<li>${unordered?.[1] || ordered?.[1]}</li>`);
        continue;
      }
      closeList();
      if (!line.trim()) {
        output.push("");
      } else if (/^###\s+/.test(line)) {
        output.push(`<h3>${line.replace(/^###\s+/, "")}</h3>`);
      } else if (/^##\s+/.test(line)) {
        output.push(`<h2>${line.replace(/^##\s+/, "")}</h2>`);
      } else if (/^#\s+/.test(line)) {
        output.push(`<h1>${line.replace(/^#\s+/, "")}</h1>`);
      } else if (/^&gt;\s?/.test(line)) {
        output.push(`<blockquote>${line.replace(/^&gt;\s?/, "")}</blockquote>`);
      } else {
        output.push(`<p>${line}</p>`);
      }
    }
    closeList();

    let html = output.join("").replace(/<p><\/p>/g, "");
    codeBlocks.forEach((block, index) => {
      html = html.replace(`@@SAM_CODE_${index}@@`, block);
    });
    return html;
  }

  function formatTime(value, includeDate = false) {
    if (!value) return "now";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return safeText(value, "now");
    const today = new Date();
    const sameDay = date.toDateString() === today.toDateString();
    const options = includeDate && !sameDay
      ? { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }
      : { hour: "numeric", minute: "2-digit" };
    return new Intl.DateTimeFormat(undefined, options).format(date);
  }

  function relativeTime(value) {
    if (!value) return "Just now";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "Recently";
    const delta = Date.now() - date.getTime();
    const minutes = Math.floor(delta / 60_000);
    if (minutes < 1) return "Just now";
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    if (days < 7) return `${days}d ago`;
    return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(date);
  }

  function truncate(value, length = 84) {
    const text = safeText(value).replace(/\s+/g, " ").trim();
    return text.length > length ? `${text.slice(0, length - 1)}…` : text;
  }

  function showToast(title, message = "", type = "info", duration = 4200) {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.innerHTML = `
      <span class="toast-icon" aria-hidden="true">${type === "success" ? "✓" : type === "error" ? "!" : type === "warning" ? "◆" : "i"}</span>
      <span class="toast-copy"><strong></strong><span></span></span>
      <button type="button" aria-label="Dismiss notification">×</button>`;
    $("strong", toast).textContent = title;
    $(".toast-copy span", toast).textContent = message;
    const remove = () => toast.remove();
    $("button", toast).addEventListener("click", remove);
    dom.toastStack.appendChild(toast);
    window.setTimeout(remove, duration);
  }

  function setConnectivity(status, health = null, latencyMs = null) {
    state.connectivity = status;
    dom.connectionPill.classList.toggle("offline", status === "offline");
    dom.connectionPill.classList.toggle("connecting", status === "connecting");
    dom.sidebarStatusDot.classList.toggle("offline", status === "offline");
    dom.sidebarStatusDot.classList.toggle("connecting", status === "connecting");

    const pillText = $(".connection-text", dom.connectionPill);
    if (status === "online") {
      const provider = health?.provider || state.settings.provider;
      const modelReady = state.models.some((item) => item.available !== false);
      const modelMissing = state.modelsLoaded && !modelReady;
      pillText.textContent = modelMissing
        ? t("status.needsModel", "SAM needs a chat model")
        : provider ? `${titleCase(provider)} ${t("status.ready", "ready")}` : t("status.pill.ready", "Local service ready");
      dom.sidebarStatusLabel.textContent = modelMissing ? t("status.needsModel", "SAM needs a chat model") : t("status.ready", "SAM is ready");
      dom.sidebarProviderLabel.textContent = modelMissing
        ? t("status.needsModel", "Local service online · chat model unavailable")
        : provider ? `${titleCase(provider)} · ${health?.status || "online"}` : t("status.checking", "Local service online");
      dom.systemMeter.style.width = `${Math.max(12, Math.min(100, Number(health?.load_percent ?? 38)))}%`;
    } else if (status === "offline") {
      pillText.textContent = t("status.pill.offline", "Service offline");
      dom.sidebarStatusLabel.textContent = t("status.offline", "Service offline");
      dom.sidebarProviderLabel.textContent = t("status.startSam", "Start SAM to connect");
      dom.systemMeter.style.width = "0%";
    } else {
      pillText.textContent = t("status.pill.connecting", "Connecting");
      dom.sidebarStatusLabel.textContent = t("status.connecting", "Connecting…");
      dom.sidebarProviderLabel.textContent = t("status.checking", "Checking local service");
      dom.systemMeter.style.width = "14%";
    }

    const availableModel = state.models.find((item) => item.available !== false && item.default)
      || state.models.find((item) => item.available !== false);
    const model = health?.model || availableModel?.name;
    dom.sidebarModelLabel.textContent = state.modelsLoaded && !availableModel
      ? t("status.noChatModel", "No chat model available")
      : model || state.settings.model || t("status.noModel", "No model selected");
    dom.sidebarLatencyLabel.textContent = Number.isFinite(latencyMs) ? `${Math.round(latencyMs)} ms` : "—";
  }

  function titleCase(value) {
    return safeText(value)
      .replaceAll("_", " ")
      .replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  async function loadHealth({ quiet = false } = {}) {
    const started = performance.now();
    if (!quiet) setConnectivity("connecting");
    try {
      const payload = await getJson("/api/health");
      const health = payload?.health || payload || {};
      state.health = health;
      setConnectivity("online", health, performance.now() - started);
      return health;
    } catch (error) {
      state.health = null;
      setConnectivity("offline");
      if (!quiet) showToast(t("toast.offline", "SAM is offline"), t("toast.offline.detail", "Start the local service, then refresh the connection."), "warning");
      throw error;
    }
  }

  function normalizeConversation(item) {
    return {
      id: item?.id ?? item?.conversation_id ?? item?.uuid,
      title: item?.title || item?.name || "Untitled conversation",
      preview: item?.preview || item?.last_message || `${item?.message_count ?? 0} messages`,
      updated_at: item?.updated_at || item?.modified_at || item?.created_at,
      provider: item?.provider || "",
      model: item?.model || "",
      message_count: item?.message_count ?? 0,
    };
  }

  async function loadConversations({ selectFirst = false } = {}) {
    try {
      const payload = await getJson("/api/conversations");
      state.conversations = asArray(payload, "conversations", "items").map(normalizeConversation).filter((item) => item.id);
      state.conversations.sort((a, b) => new Date(b.updated_at || 0) - new Date(a.updated_at || 0));
      renderSessions();
      if (selectFirst && !state.conversationId && state.conversations.length) {
        await openConversation(state.conversations[0].id, { quiet: true });
      }
      return state.conversations;
    } catch (error) {
      state.conversations = [];
      renderSessions();
      return [];
    }
  }

  function renderSessions() {
    dom.sessionList.replaceChildren();
    dom.sessionsEmpty.hidden = state.conversations.length > 0;
    const template = $("#session-template");
    for (const conversation of state.conversations) {
      const node = template.content.firstElementChild.cloneNode(true);
      node.dataset.sessionId = conversation.id;
      node.classList.toggle("active", String(conversation.id) === String(state.conversationId));
      $("strong", node).textContent = conversation.title;
      $("small", node).textContent = conversation.updated_at
        ? `${relativeTime(conversation.updated_at)} · ${truncate(conversation.preview, 26)}`
        : truncate(conversation.preview, 34);
      node.setAttribute("aria-current", node.classList.contains("active") ? "page" : "false");
      dom.sessionList.appendChild(node);
    }
  }

  function normalizeMessage(item) {
    const role = ["user", "assistant", "system", "tool"].includes(item?.role) ? item.role : "assistant";
    const metadata = item?.metadata && typeof item.metadata === "object" ? item.metadata : {};
    return {
      id: item?.id || crypto.randomUUID(),
      role,
      content: safeText(item?.content ?? item?.message ?? item?.text),
      created_at: item?.created_at || item?.timestamp || new Date().toISOString(),
      tool_name: item?.tool_name || metadata?.tool_name || "",
      metadata,
      tools: asArray(item?.tools || metadata?.tools || metadata?.tool_calls, "tools"),
    };
  }

  async function openConversation(id, { quiet = false } = {}) {
    if (!id) return;
    state.conversationId = id;
    state.messages = [];
    state.plan = [];
    state.tools = [];
    const conversation = state.conversations.find((item) => String(item.id) === String(id));
    dom.title.textContent = conversation?.title || "Conversation";
    renderSessions();
    renderMessages({ loading: true });
    closeMobilePanels();
    try {
      const payload = await getJson(`/api/conversations/${encodeURIComponent(id)}/messages`);
      state.messages = asArray(payload, "messages", "items").map(normalizeMessage);
      const lastPlan = [...state.messages].reverse().find((message) => Array.isArray(message.metadata?.plan))?.metadata?.plan;
      if (lastPlan) updatePlan(lastPlan);
      renderMessages();
    } catch (error) {
      state.messages = [];
      renderMessages();
      if (!quiet) showToast("Could not load conversation", error.message, "error");
    }
  }

  function renderMessages({ loading = false } = {}) {
    dom.feed.replaceChildren();
    if (loading) {
      const pending = makeMessageElement({ role: "assistant", content: "", pending: true });
      dom.feed.appendChild(pending);
      return;
    }
    if (!state.messages.length) {
      dom.feed.appendChild(dom.welcome);
      dom.welcome.hidden = false;
      return;
    }
    dom.welcome.hidden = true;
    for (const message of state.messages) dom.feed.appendChild(makeMessageElement(message));
    requestAnimationFrame(scrollToBottom);
  }

  function makeMessageElement(message) {
    const template = $("#message-template");
    const node = template.content.firstElementChild.cloneNode(true);
    const role = message.role === "user" ? "user" : "assistant";
    node.classList.add(role);
    node.dataset.messageId = message.id || crypto.randomUUID();
    node.classList.toggle("pending", Boolean(message.pending));
    const avatar = $(".message-avatar", node);
    avatar.textContent = role === "user" ? "Y" : "S";
    $(".message-meta strong", node).textContent = role === "user" ? "You" : state.settings.agent_name || "SAM";
    const time = $("time", node);
    time.textContent = message.pending ? "working" : formatTime(message.created_at);
    time.dateTime = message.created_at || new Date().toISOString();
    const content = $(".message-content", node);
    content.dir = "auto";
    if (message.pending) {
      content.innerHTML = '<span class="thinking-dots" aria-label="SAM is thinking"><i></i><i></i><i></i></span>';
      $(".message-actions", node).hidden = true;
    } else {
      content.innerHTML = renderMarkdown(message.content);
    }

    const toolItems = Array.isArray(message.tools) ? message.tools : [];
    if (toolItems.length) {
      const tools = $(".message-tools", node);
      tools.hidden = false;
      tools.replaceChildren(...toolItems.map((tool) => {
        const chip = document.createElement("span");
        chip.className = "tool-chip";
        const name = tool?.name || tool?.tool_name || tool?.type || safeText(tool);
        chip.innerHTML = '<span class="tool-status-dot" aria-hidden="true"></span><span></span>';
        $("span:last-child", chip).textContent = titleCase(name);
        return chip;
      }));
    }
    return node;
  }

  function scrollToBottom() {
    dom.chatScroll.scrollTop = dom.chatScroll.scrollHeight;
  }

  async function createConversation() {
    state.conversationId = null;
    state.messages = [];
    state.plan = [];
    state.tools = [];
    state.pendingFiles = [];
    dom.title.textContent = "New conversation";
    renderMessages();
    renderSessions();
    renderPlan();
    renderAttachments();
    closeMobilePanels();
    dom.input.focus();

    try {
      const payload = await getJson("/api/conversations", {
        method: "POST",
        body: JSON.stringify({
          title: "New conversation",
          provider: state.settings.provider || "auto",
          model: state.settings.model || undefined,
        }),
      });
      const conversation = normalizeConversation(payload?.conversation || payload);
      if (conversation.id) {
        state.conversationId = conversation.id;
        state.conversations.unshift(conversation);
        renderSessions();
      }
    } catch {
      // Conversation creation is also supported implicitly by /api/chat.
    }
  }

  async function sendMessage(text) {
    const clean = text.trim();
    if ((!clean && !state.pendingFiles.length) || state.sending) return;
    const provider = (state.settings.provider || state.health?.provider || "auto").toLowerCase();
    if (state.pendingFiles.length && provider === "openai") {
      const confirmed = window.confirm(
        "The selected file contents will be included in a request to the configured cloud model. Continue?"
      );
      if (!confirmed) return;
    }
    let outgoing;
    try {
      outgoing = await buildOutgoingMessage(clean);
    } catch (error) {
      showToast("Attachment cancelled", error.message || "The selected file was not attached.", "warning", 6000);
      return;
    }
    const visibleContent = clean || `Shared ${state.pendingFiles.length} file${state.pendingFiles.length === 1 ? "" : "s"}`;
    const attachmentNames = state.pendingFiles.map((file) => file.name);
    state.pendingFiles = [];
    renderAttachments();
    dom.input.value = "";
    resizeComposer();
    updateSendButton();

    const userMessage = normalizeMessage({
      role: "user",
      content: attachmentNames.length ? `${visibleContent}\n\n_Attached: ${attachmentNames.join(", ")}_` : visibleContent,
    });
    state.messages.push(userMessage);
    if (dom.welcome?.isConnected) dom.welcome.remove();
    dom.feed.appendChild(makeMessageElement(userMessage));
    const pending = normalizeMessage({ role: "assistant", content: "" });
    pending.pending = true;
    const pendingNode = makeMessageElement(pending);
    dom.feed.appendChild(pendingNode);
    scrollToBottom();

    state.sending = true;
    state.abortController = new AbortController();
    updateSendButton();
    clearToolActivity();

    const payload = {
      message: outgoing,
      provider: state.settings.provider || state.health?.provider || undefined,
      model: state.settings.model || state.health?.model || undefined,
      conversation_id: state.conversationId || undefined,
    };

    const speakReply = Boolean(state.settings.voice_output) || Boolean(state.voiceTurn);
    state.voiceTurn = false;
    const speakSorani = speakReply && (isSoraniVoice() || looksSoraniText(outgoing));
    if (speakReply) {
      stopSpeakingReply();
      state.ttsCursor = 0;
    }

    let assistant = { id: pending.id, role: "assistant", content: "", created_at: new Date().toISOString(), tools: [] };
    try {
      try {
        assistant = await sendStreamingChat(
          payload,
          pendingNode,
          assistant,
          state.abortController.signal,
          speakSorani ? (text) => enqueueSpokenReply(text, false) : null,
        );
      } catch (error) {
        if (![404, 405, 415].includes(error.status)) throw error;
        assistant = await sendStandardChat(payload, pendingNode, assistant, state.abortController.signal);
      }
      if (!assistant.content.trim()) assistant.content = "Done.";
      finalizeAssistantNode(pendingNode, assistant);
      state.messages.push(normalizeMessage(assistant));
      if (assistant.conversation_id && !state.conversationId) state.conversationId = assistant.conversation_id;
      if (speakSorani) enqueueSpokenReply(assistant.content, true);
      else if (speakReply) speakText(assistant.content);
      await Promise.allSettled([loadConversations(), loadApprovals(), loadAudit(), loadMemories()]);
    } catch (error) {
      if (error?.name === "AbortError" || state.abortController?.signal.aborted) {
        assistant.content = assistant.content.trim() || "Response stopped.";
        finalizeAssistantNode(pendingNode, assistant, "stopped");
        state.messages.push(normalizeMessage(assistant));
      } else {
        pendingNode.remove();
        showToast("Message failed", error.message || "SAM could not complete the request.", "error", 6000);
        const errorMessage = normalizeMessage({
          role: "assistant",
          content: "I couldn’t reach the local agent service. Start SAM or refresh the connection, then try again.",
        });
        state.messages.push(errorMessage);
        dom.feed.appendChild(makeMessageElement(errorMessage));
      }
    } finally {
      state.sending = false;
      state.abortController = null;
      updateSendButton();
      scrollToBottom();
    }
  }

  async function buildOutgoingMessage(text) {
    if (!state.pendingFiles.length) return text;
    const maxTotal = 20_000;
    const parts = [text];
    let used = text.length;
    for (const file of state.pendingFiles.slice(0, 5)) {
      if (used >= maxTotal) break;
      const header = `\n\n--- Attached file: ${file.name} (${file.type || "unknown type"}) ---\n`;
      if (!looksTextual(file)) {
        parts.push(`${header}[Binary content is not embedded. Ask SAM to read it from the workspace if available.]`);
        continue;
      }
      let content = "";
      try {
        content = await file.text();
      } catch {
        content = "[Could not read this file in the browser.]";
      }
      if (looksSensitiveAttachment(file.name, content)) {
        const confirmed = window.confirm(
          `“${file.name}” may contain credentials or private key material. Include its text in this one message?`
        );
        if (!confirmed) throw new Error(`Potentially sensitive attachment “${file.name}” was not sent.`);
      }
      const remaining = Math.max(0, maxTotal - used - header.length);
      if (content.length > remaining) content = `${content.slice(0, Math.max(0, remaining - 30))}\n[Content truncated by the UI]`;
      parts.push(header + content);
      used += header.length + content.length;
    }
    return parts.join("");
  }

  function looksTextual(file) {
    const type = file.type || "";
    const ext = file.name.split(".").pop()?.toLowerCase();
    return type.startsWith("text/") || ["json", "js", "jsx", "ts", "tsx", "py", "md", "txt", "csv", "yaml", "yml", "toml", "ini", "html", "css", "xml", "sql", "log", "sh", "ps1"].includes(ext);
  }

  function looksSensitiveAttachment(name, content) {
    const lowerName = String(name || "").toLowerCase();
    const sensitiveName = /(^|[._-])(env|credentials?|secrets?|passwords?)([._-]|$)|^id_(rsa|ed25519)$|\.pem$|\.pfx$|\.key$/.test(lowerName);
    const sensitiveContent = /-----BEGIN [A-Z ]*PRIVATE KEY-----|(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*[^\s]{6,}/i.test(String(content || ""));
    return sensitiveName || sensitiveContent;
  }

  async function sendStreamingChat(payload, pendingNode, assistant, signal, onContent) {
    const response = await apiFetch("/api/chat/stream", {
      method: "POST",
      headers: { Accept: "text/event-stream" },
      body: JSON.stringify(payload),
      signal,
    }, 180_000);
    const contentType = response.headers.get("content-type") || "";
    if (!response.body || !contentType.includes("text/event-stream")) {
      const result = await response.json();
      const applied = applyChatPayload(result, assistant, pendingNode);
      onContent?.(applied.content);
      return applied;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replaceAll("\r\n", "\n");
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const event = parseSseBlock(block);
        if (event) {
          assistant = applyStreamEvent(event, assistant, pendingNode);
          onContent?.(assistant.content);
        }
      }
    }
    if (buffer.trim()) {
      const event = parseSseBlock(buffer);
      if (event) {
        assistant = applyStreamEvent(event, assistant, pendingNode);
        onContent?.(assistant.content);
      }
    }
    return assistant;
  }

  function parseSseBlock(block) {
    if (!block.trim() || block.trim().startsWith(":")) return null;
    let eventName = "message";
    const dataLines = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) eventName = line.slice(6).trim();
      if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
    }
    if (!dataLines.length) return null;
    const raw = dataLines.join("\n");
    if (raw === "[DONE]") return { type: "done" };
    try {
      const data = JSON.parse(raw);
      return { type: data.type || eventName, ...data };
    } catch {
      return { type: eventName, token: raw };
    }
  }

  function applyStreamEvent(event, assistant, pendingNode) {
    const type = safeText(event.type || "message").toLowerCase();
    if (type === "error" || event.error) {
      throw new ApiError(safeText(event.error?.message || event.error || event.message, "Streaming response failed."), 500, event);
    }
    if (type === "result") {
      assistant = applyChatPayload(event, assistant, pendingNode);
    }
    const token = event.token ?? event.delta ?? (type === "token" ? event.content : undefined);
    if (token !== undefined) {
      assistant.content += safeText(token);
      updatePendingAssistant(pendingNode, assistant.content);
    }
    if (type === "message" && event.message) {
      if (typeof event.message === "string") {
        if (!assistant.content || event.final) assistant.content = event.message;
      } else {
        assistant = applyChatPayload({ message: event.message, ...event }, assistant, pendingNode);
      }
      updatePendingAssistant(pendingNode, assistant.content);
    }
    if (event.content && !token && ["response", "assistant"].includes(type)) {
      assistant.content = safeText(event.content);
      updatePendingAssistant(pendingNode, assistant.content);
    }
    if (type === "tool" || type === "tool_call" || event.tool_call) {
      const tool = event.tool_call || event.tool || event;
      assistant.tools = [...(assistant.tools || []), tool];
      addToolActivity(tool);
    }
    if (type === "plan" || event.plan) updatePlan(event.plan || event.steps || []);
    if (type === "approval" || event.approval || event.pending_approvals) {
      mergeApprovals(event.pending_approvals || [event.approval || event]);
      showApprovalNotification();
    }
    if (event.conversation_id) assistant.conversation_id = event.conversation_id;
    return assistant;
  }

  async function sendStandardChat(payload, pendingNode, assistant, signal) {
    const response = await apiFetch("/api/chat", {
      method: "POST",
      body: JSON.stringify(payload),
      signal,
    }, 180_000);
    const result = await response.json();
    return applyChatPayload(result, assistant, pendingNode);
  }

  function applyChatPayload(payload, assistant, pendingNode) {
    const message = payload?.message || payload?.response || payload?.assistant || payload;
    if (typeof message === "string") assistant.content = message;
    else if (message && typeof message === "object") {
      assistant = {
        ...assistant,
        ...normalizeMessage(message),
        conversation_id: payload?.conversation_id || message?.conversation_id || assistant.conversation_id,
        tools: asArray(message?.tools || message?.tool_calls || message?.metadata?.tool_calls, "tools"),
      };
    }
    if (payload?.conversation_id) assistant.conversation_id = payload.conversation_id;
    if (payload?.plan) updatePlan(payload.plan);
    const approvals = payload?.pending_approvals || payload?.approvals;
    if (approvals?.length) {
      mergeApprovals(approvals);
      showApprovalNotification();
    }
    const toolCalls = [
      ...asArray(payload?.tool_calls, "tools"),
      ...asArray(payload?.tool_events, "events"),
    ];
    if (toolCalls.length) {
      assistant.tools = [...(assistant.tools || []), ...toolCalls];
      toolCalls.forEach(addToolActivity);
    }
    updatePendingAssistant(pendingNode, assistant.content);
    return assistant;
  }

  function updatePendingAssistant(node, content) {
    const contentNode = $(".message-content", node);
    if (!contentNode) return;
    node.classList.add("pending");
    contentNode.innerHTML = content ? renderMarkdown(content) : '<span class="thinking-dots" aria-label="SAM is thinking"><i></i><i></i><i></i></span>';
    requestAnimationFrame(scrollToBottom);
  }

  function finalizeAssistantNode(node, assistant, status = "") {
    node.classList.remove("pending");
    const contentNode = $(".message-content", node);
    contentNode.innerHTML = renderMarkdown(assistant.content);
    $("time", node).textContent = formatTime(assistant.created_at || new Date().toISOString());
    const stateLabel = $(".message-state", node);
    stateLabel.textContent = status;
    $(".message-actions", node).hidden = false;
    if (assistant.tools?.length) {
      const tools = $(".message-tools", node);
      tools.hidden = false;
      tools.replaceChildren(...assistant.tools.map((tool) => {
        const chip = document.createElement("span");
        chip.className = "tool-chip";
        chip.innerHTML = '<span class="tool-status-dot" aria-hidden="true"></span><span></span>';
        $("span:last-child", chip).textContent = titleCase(tool?.name || tool?.tool_name || tool?.type || safeText(tool));
        return chip;
      }));
    }
  }

  function stopCurrentResponse() {
    state.abortController?.abort("user");
  }

  function updateSendButton() {
    const hasInput = Boolean(dom.input.value.trim() || state.pendingFiles.length);
    dom.send.disabled = !hasInput && !state.sending;
    dom.send.classList.toggle("stop", state.sending);
    dom.send.setAttribute("aria-label", state.sending ? "Stop response" : "Send message");
    $("span", dom.send).textContent = state.sending ? "■" : "↑";
  }

  function resizeComposer() {
    dom.input.style.height = "auto";
    dom.input.style.height = `${Math.min(180, dom.input.scrollHeight)}px`;
  }

  function handleFiles(files) {
    const accepted = [...files].slice(0, Math.max(0, 5 - state.pendingFiles.length));
    const tooLarge = accepted.filter((file) => file.size > 3 * 1024 * 1024);
    state.pendingFiles.push(...accepted.filter((file) => file.size <= 3 * 1024 * 1024));
    if (tooLarge.length) showToast("Some files were skipped", "Individual attachments must be smaller than 3 MB.", "warning");
    renderAttachments();
    updateSendButton();
  }

  function renderAttachments() {
    dom.attachments.replaceChildren();
    dom.attachments.hidden = state.pendingFiles.length === 0;
    state.pendingFiles.forEach((file, index) => {
      const chip = document.createElement("span");
      chip.className = "attachment-chip";
      chip.innerHTML = '<span aria-hidden="true">◇</span><strong></strong><button type="button" aria-label="Remove attachment">×</button>';
      $("strong", chip).textContent = file.name;
      $("button", chip).addEventListener("click", () => {
        state.pendingFiles.splice(index, 1);
        renderAttachments();
        updateSendButton();
      });
      dom.attachments.appendChild(chip);
    });
  }

  function normalizeApproval(item) {
    const args = item?.arguments || item?.args || item?.parameters || {};
    const command = item?.command || args?.command || args?.path || args?.url || (Object.keys(args).length ? JSON.stringify(args, null, 2) : "");
    return {
      ...item,
      id: item?.id || item?.approval_id,
      tool_name: item?.tool_name || item?.tool || item?.action || "Sensitive action",
      risk_level: item?.risk_level || item?.risk || "high",
      reason: item?.reason || item?.description || "This action requires explicit approval.",
      command: safeText(command),
      status: item?.status || "pending",
      requested_at: item?.requested_at || item?.created_at || new Date().toISOString(),
      arguments: args,
    };
  }

  async function loadApprovals() {
    try {
      const payload = await getJson("/api/approvals?status=pending");
      state.approvals = asArray(payload, "approvals", "items")
        .map(normalizeApproval)
        .filter((item) => item.id && !["approved", "denied", "expired", "completed"].includes(item.status));
      renderApprovals();
      return state.approvals;
    } catch {
      return state.approvals;
    }
  }

  function mergeApprovals(items) {
    for (const raw of asArray(items, "approvals")) {
      const item = normalizeApproval(raw);
      if (!item.id) continue;
      const index = state.approvals.findIndex((approval) => String(approval.id) === String(item.id));
      if (index >= 0) state.approvals[index] = item;
      else state.approvals.unshift(item);
    }
    renderApprovals();
  }

  function renderApprovals() {
    for (const emptyState of dom.approvalsEmptyStates) emptyState.hidden = state.approvals.length > 0;
    dom.approvalCount.hidden = state.approvals.length === 0;
    dom.approvalCount.textContent = String(state.approvals.length);
    if (dom.sidebarApprovalBadge) dom.sidebarApprovalBadge.textContent = state.approvals.length ? String(state.approvals.length) : "";
    const template = $("#approval-template");
    for (const list of dom.approvalLists) {
      list.replaceChildren();
      for (const approval of state.approvals) {
        const node = template.content.firstElementChild.cloneNode(true);
        node.dataset.approvalId = approval.id;
        $(".approval-type", node).textContent = `${titleCase(approval.risk_level)} risk`;
        $("strong", node).textContent = titleCase(approval.tool_name);
        $("small", node).textContent = truncate(approval.reason, 55);
        list.appendChild(node);
      }
    }
  }

  function openApproval(id) {
    const approval = state.approvals.find((item) => String(item.id) === String(id));
    if (!approval) return;
    state.currentApproval = approval;
    $("#approval-description").textContent = approval.reason;
    $("#approval-risk").textContent = titleCase(approval.risk_level);
    $("#approval-command").textContent = approval.command || JSON.stringify(approval.arguments || {}, null, 2) || "No command details provided.";
    $("#approval-command-wrap").hidden = !approval.command && !Object.keys(approval.arguments || {}).length;
    $("#approval-scope").textContent = titleCase(approval.scope || "This action only");
    $("#approval-warning-copy").textContent = approval.warning || "Only approve if you understand and expect this action.";
    if (!dom.approvalModal.open) dom.approvalModal.showModal();
  }

  async function decideApproval(decision) {
    const approval = state.currentApproval;
    if (!approval) return;
    const button = decision === "approve" ? $("[data-action='approve-action']") : $("[data-action='deny-approval']");
    button.disabled = true;
    try {
      await getJson(`/api/approvals/${encodeURIComponent(approval.id)}/decision`, {
        method: "POST",
        body: JSON.stringify({
          decision: decision === "approve" ? "approved" : "denied",
          note: "Decision made from the SAM UI.",
        }),
      });
      state.approvals = state.approvals.filter((item) => String(item.id) !== String(approval.id));
      renderApprovals();
      dom.approvalModal.close();
      state.currentApproval = null;
      showToast(decision === "approve" ? "Action approved" : "Action denied", titleCase(approval.tool_name), decision === "approve" ? "success" : "info");
      await loadAudit();
    } catch (error) {
      showToast("Decision failed", error.message, "error");
    } finally {
      button.disabled = false;
    }
  }

  function showApprovalNotification() {
    if (!state.approvals.length) return;
    showToast("Approval required", `${titleCase(state.approvals[0].tool_name)} is waiting for your review.`, "warning", 6500);
    if (document.visibilityState === "hidden" && state.settings.notifications && "Notification" in window && Notification.permission === "granted") {
      new Notification("SAM needs your approval", { body: titleCase(state.approvals[0].tool_name), tag: `sam-approval-${state.approvals[0].id}` });
    }
  }

  function normalizeMemory(item) {
    return {
      id: item?.id || item?.memory_id,
      category: item?.category || item?.type || "context",
      content: safeText(item?.content ?? item?.text ?? item?.value),
      source: item?.source || "User approved",
      created_at: item?.created_at || item?.updated_at,
    };
  }

  async function loadMemories(query = "") {
    try {
      const suffix = query ? `?q=${encodeURIComponent(query)}` : "";
      const payload = await getJson(`/api/memories${suffix}`);
      state.memories = asArray(payload, "memories", "items").map(normalizeMemory).filter((item) => item.content);
      renderMemories();
      return state.memories;
    } catch {
      return state.memories;
    }
  }

  function renderMemories() {
    const query = state.memoryQuery.trim().toLowerCase();
    const items = state.memories.filter((memory) => !query || `${memory.category} ${memory.content}`.toLowerCase().includes(query));
    for (const emptyState of dom.memoryEmptyStates) emptyState.hidden = items.length > 0;
    for (const list of dom.memoryLists) {
      list.replaceChildren();
      for (const memory of items) {
        const card = document.createElement("article");
        card.className = "memory-card";
        card.innerHTML = `
          <div class="memory-card-head"><span></span><time></time></div>
          <p dir="auto"></p>
          <footer><span></span></footer>`;
        $(".memory-card-head span", card).textContent = titleCase(memory.category);
        $("time", card).textContent = relativeTime(memory.created_at);
        $("p", card).textContent = memory.content;
        $("footer span", card).textContent = memory.source;
        list.appendChild(card);
      }
    }
  }

  function normalizeAudit(item) {
    return {
      id: item?.id || crypto.randomUUID(),
      category: safeText(item?.category || item?.type || item?.event_type || "tool").toLowerCase(),
      title: item?.title || item?.action || item?.tool_name || titleCase(item?.event_type || "Activity"),
      description: safeText(item?.description || item?.message || item?.summary),
      detail: safeText(item?.command || item?.detail || item?.details || item?.arguments),
      timestamp: item?.timestamp || item?.created_at || item?.time,
      status: item?.status || "",
    };
  }

  async function loadAudit() {
    try {
      const payload = await getJson("/api/audit?limit=100");
      state.audit = asArray(payload, "events", "audit", "items", "entries").map(normalizeAudit);
      renderAudit();
      return state.audit;
    } catch {
      return state.audit;
    }
  }

  function renderAudit() {
    const filter = state.auditFilter;
    const items = state.audit.filter((item) => filter === "all" || item.category.includes(filter));
    for (const emptyState of dom.auditEmptyStates) emptyState.hidden = items.length > 0;
    const template = $("#audit-template");
    for (const list of dom.auditLists) {
      list.replaceChildren();
      for (const event of items) {
        const node = template.content.firstElementChild.cloneNode(true);
        node.classList.add(event.category.includes("error") ? "error" : event.category.includes("approval") ? "approval" : event.category.includes("security") ? "security" : "tool");
        $("strong", node).textContent = event.title;
        const time = $("time", node);
        time.textContent = formatTime(event.timestamp, true);
        time.dateTime = event.timestamp || "";
        $("p", node).textContent = event.description || titleCase(event.status || event.category);
        const code = $("code", node);
        code.hidden = !event.detail;
        code.textContent = truncate(event.detail, 180);
        list.appendChild(node);
      }
    }
  }

  function exportAudit() {
    if (!state.audit.length) {
      showToast("Nothing to export", "The audit trail is currently empty.", "info");
      return;
    }
    const blob = new Blob([JSON.stringify(state.audit, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `sam-audit-${new Date().toISOString().slice(0, 10)}.json`;
    link.click();
    URL.revokeObjectURL(url);
    showToast("Audit exported", "Saved as a local JSON file.", "success");
  }

  function normalizePlanStep(item, index) {
    if (typeof item === "string") return { title: item, status: "pending", description: "" };
    const rawStatus = safeText(item?.status || (item?.complete ? "completed" : "pending")).toLowerCase();
    const status = ["done", "complete", "completed", "success"].includes(rawStatus)
      ? "complete"
      : ["active", "running", "in_progress", "in-progress", "current"].includes(rawStatus)
        ? "in-progress"
        : "pending";
    return {
      title: item?.title || item?.step || item?.task || `Step ${index + 1}`,
      description: item?.description || item?.detail || "",
      status,
    };
  }

  function updatePlan(plan) {
    const steps = Array.isArray(plan) ? plan : asArray(plan, "steps", "items", "plan");
    state.plan = steps.map(normalizePlanStep);
    renderPlan();
  }

  function renderPlan() {
    for (const list of dom.planLists) list.replaceChildren();
    if (!state.plan.length) {
      for (const list of dom.planLists) {
        const li = document.createElement("li");
        li.className = "plan-placeholder";
        li.innerHTML = '<span class="step-marker">1</span><div><strong>Waiting for a task</strong><small>Tell SAM what outcome you want.</small></div>';
        list.appendChild(li);
      }
      setPlanProgress(0);
      for (const label of dom.planProgressLabels) label.textContent = "0%";
      for (const title of dom.planStatusTitles) title.textContent = "No active plan";
      for (const copy of dom.planStatusCopies) copy.textContent = "A task plan will appear here when SAM starts multi-step work.";
      return;
    }

    const complete = state.plan.filter((step) => step.status === "complete").length;
    const progress = Math.round((complete / state.plan.length) * 100);
    setPlanProgress(progress);
    for (const label of dom.planProgressLabels) label.textContent = `${progress}%`;
    const running = state.plan.find((step) => step.status === "in-progress");
    for (const title of dom.planStatusTitles) title.textContent = progress === 100 ? "Plan complete" : running ? "Work in progress" : "Plan ready";
    for (const copy of dom.planStatusCopies) copy.textContent = `${complete} of ${state.plan.length} steps complete${running ? ` · ${truncate(running.title, 35)}` : ""}`;

    for (const list of dom.planLists) {
      state.plan.forEach((step, index) => {
        const li = document.createElement("li");
        li.className = step.status;
        const marker = step.status === "complete" ? "✓" : String(index + 1);
        li.innerHTML = '<span class="step-marker"></span><div><strong></strong><small></small></div>';
        $(".step-marker", li).textContent = marker;
        $("strong", li).textContent = step.title;
        $("small", li).textContent = step.description || titleCase(step.status);
        list.appendChild(li);
      });
    }
  }

  function setPlanProgress(progress) {
    const normalized = Math.max(0, Math.min(100, Number(progress) || 0));
    for (const indicator of dom.planProgressIndicators) {
      indicator.style.setProperty("--progress", `${normalized}%`);
      if (indicator.dataset.progressKind === "bar") {
        indicator.style.width = `${normalized}%`;
        continue;
      }
      const circle = $(".plan-ring-fill", indicator);
      if (circle) circle.style.strokeDashoffset = String(113.1 * (1 - normalized / 100));
    }
  }

  function addToolActivity(tool) {
    const normalized = {
      name: tool?.name || tool?.tool_name || tool?.type || "tool",
      detail: safeText(tool?.detail || tool?.command || tool?.arguments || "Running"),
    };
    state.tools.push(normalized);
    state.tools = state.tools.slice(-5);
    for (const activity of dom.toolActivities) activity.hidden = false;
    for (const items of dom.toolActivityItems) {
      items.replaceChildren(...state.tools.map((item) => {
        const row = document.createElement("div");
        row.className = "tool-activity-row";
        row.innerHTML = "<strong></strong><code></code>";
        $("strong", row).textContent = titleCase(item.name);
        $("code", row).textContent = truncate(item.detail, 30);
        return row;
      }));
    }
  }

  function clearToolActivity() {
    state.tools = [];
    for (const activity of dom.toolActivities) activity.hidden = true;
    for (const items of dom.toolActivityItems) items.replaceChildren();
  }

  async function loadModels() {
    try {
      const payload = await getJson("/api/models");
      const raw = asArray(payload, "models", "items");
      state.models = raw.map((item) => typeof item === "string"
        ? { id: item, name: item, provider: "ollama", available: true }
        : {
            id: item?.id || item?.name || item?.model,
            name: item?.display_name || item?.name || item?.model || item?.id,
            provider: item?.provider || "ollama",
            available: item?.available !== false,
            default: Boolean(item?.default),
            local: item?.local === true || item?.provider === "ollama",
          }).filter((item) => item.id);
      state.modelsLoaded = true;
      renderModelSettings();
      if (state.health) setConnectivity("online", state.health);
      return state.models;
    } catch {
      state.models = [];
      state.modelsLoaded = true;
      renderModelSettings();
      if (state.health) setConnectivity("online", state.health);
      return [];
    }
  }

  function renderModelSettings() {
    const providerSelect = $("#settings-provider");
    const modelSelect = $("#settings-model");
    const providers = [...new Set(["auto", "ollama", "litellm", "openrouter", "openai", ...state.models.map((item) => item.provider), state.settings.provider].filter(Boolean))];
    providerSelect.replaceChildren(...providers.map((provider) => {
      const option = document.createElement("option");
      option.value = provider;
      option.textContent = provider === "auto" ? "Auto router" : provider === "ollama" ? "Ollama (local)" : titleCase(provider);
      return option;
    }));
    providerSelect.value = state.settings.provider || providers[0] || "auto";
    renderModelOptions();
  }

  function renderModelOptions() {
    const modelSelect = $("#settings-model");
    const provider = $("#settings-provider").value || state.settings.provider;
    let models = state.models.filter((item) => item.provider === provider);
    if (provider === "openrouter") {
      const sonnet = { id: "anthropic/claude-sonnet-4.5", name: "Claude Sonnet 4.5", provider: "openrouter", available: true };
      if (!models.some((item) => item.id === sonnet.id)) models = [sonnet, ...models];
    }
    modelSelect.replaceChildren();
    if (!models.length) {
      const option = document.createElement("option");
      option.value = state.settings.model || "";
      option.textContent = state.settings.model || "No models reported";
      modelSelect.appendChild(option);
      return;
    }
    for (const model of models) {
      const option = document.createElement("option");
      option.value = model.id;
      option.textContent = `${model.name}${model.available ? "" : " · unavailable"}${model.local ? " · local" : ""}`;
      option.disabled = !model.available;
      modelSelect.appendChild(option);
    }
    const preferred = state.settings.model || models.find((item) => item.default)?.id || models.find((item) => item.available)?.id;
    if (preferred && ![...modelSelect.options].some((item) => item.value === preferred)) {
      const option = document.createElement("option");
      option.value = preferred;
      option.textContent = preferred;
      modelSelect.appendChild(option);
    }
    if (preferred) modelSelect.value = preferred;
  }

  async function loadSettings() {
    try {
      const payload = await getJson("/api/settings");
      const runtime = { ...(payload?.runtime || {}), ...(payload?.overrides || {}) };
      const local = readLocalUiSettings();
      state.settings = {
        ...state.settings,
        ...local,
        provider: runtime.default_provider || local.provider || state.settings.provider,
        model: (runtime.default_provider || local.provider) === "openai"
          ? runtime.openai_model || local.model || state.settings.model
          : runtime.default_model || local.model || state.settings.model,
        permission_mode: runtime.permission_mode || local.permission_mode || state.settings.permission_mode,
        model_mode: runtime.model_mode || local.model_mode || state.settings.model_mode,
        daily_budget_usd: runtime.daily_budget_usd ?? local.daily_budget_usd ?? state.settings.daily_budget_usd,
        monthly_budget_usd: runtime.monthly_budget_usd ?? local.monthly_budget_usd ?? state.settings.monthly_budget_usd,
        computer_control_enabled: runtime.computer_control_enabled ?? local.computer_control_enabled ?? false,
        screen_access_enabled: runtime.screen_access_enabled ?? local.screen_access_enabled ?? false,
        voice_mode: runtime.voice_mode || local.voice_mode || state.settings.voice_mode,
        voice_language: runtime.voice_language || local.voice_language || state.settings.voice_language,
        voice_wake_word: runtime.voice_wake_word || local.voice_wake_word || state.settings.voice_wake_word,
        voice_vad_threshold: runtime.voice_vad_threshold ?? local.voice_vad_threshold ?? state.settings.voice_vad_threshold,
        voice_silence_ms: runtime.voice_silence_ms ?? local.voice_silence_ms ?? state.settings.voice_silence_ms,
        workspace_path: runtime.workspace_root || local.workspace_path || "",
      };
      applySettingsToForm();
      applyUiLanguage();
      renderModelSettings();
      applyPermissionMode();
      if (state.connectivity === "online") setConnectivity("online", state.health);
      return state.settings;
    } catch {
      applySettingsToForm();
      return state.settings;
    }
  }

  function applySettingsToForm() {
    for (const [key, value] of Object.entries(state.settings)) {
      const field = dom.settingsForm.elements.namedItem(key);
      if (!field) continue;
      if (field.type === "checkbox") field.checked = Boolean(value);
      else field.value = value ?? "";
    }
    $("#temperature-output").textContent = Number(state.settings.temperature ?? 0.2).toFixed(1);
    $("#speech-rate-output").textContent = `${Number(state.settings.speech_rate ?? 1).toFixed(1)}×`;
    populateVoices();
    const supported = supportsVoiceInput();
    $("#voice-compatibility-note").textContent = supported
      ? t("voice.supported", "Voice input is supported by this browser. Speech recognition may use your browser vendor’s service.")
      : t("voice.unsupported", "Voice input is not available in this browser. Voice output may still be supported.");
    dom.voiceButton.hidden = !state.settings.voice_input;
  }

  function formToSettings() {
    const data = new FormData(dom.settingsForm);
    const next = { ...state.settings };
    for (const [key, value] of data.entries()) next[key] = value;
    for (const input of $$('input[type="checkbox"]', dom.settingsForm)) next[input.name] = input.checked;
    next.temperature = Number(next.temperature);
    next.speech_rate = Number(next.speech_rate);
    next.daily_budget_usd = Number(next.daily_budget_usd);
    next.monthly_budget_usd = Number(next.monthly_budget_usd);
    return next;
  }

  function readLocalUiSettings() {
    try {
      const value = JSON.parse(localStorage.getItem("sam-ui-settings") || "{}");
      return value && typeof value === "object" ? value : {};
    } catch {
      return {};
    }
  }

  async function saveSettings(event) {
    event.preventDefault();
    const next = formToSettings();
    dom.settingsStatus.textContent = t("settings.saving", "Saving…");
    try {
      const payload = await getJson("/api/settings", {
        method: "PUT",
        body: JSON.stringify({
          default_provider: next.provider,
          ...(next.provider === "openai" ? { openai_model: next.model } : { default_model: next.model }),
          ...(next.provider === "openrouter" ? {
            openrouter_fast_model: next.model,
            openrouter_strong_model: next.model,
            openrouter_vision_model: next.model,
          } : {}),
          model_mode: next.model_mode,
          permission_mode: next.permission_mode,
          daily_budget_usd: next.daily_budget_usd,
          monthly_budget_usd: next.monthly_budget_usd,
          computer_control_enabled: next.computer_control_enabled,
          screen_access_enabled: next.screen_access_enabled,
          voice_mode: next.voice_mode,
          voice_language: next.voice_language,
          voice_wake_word: next.voice_wake_word,
        }),
      });
      localStorage.setItem("sam-ui-settings", JSON.stringify(next));
      const runtime = { ...(payload?.runtime || {}), ...(payload?.overrides || {}) };
      state.settings = {
        ...next,
        provider: runtime.default_provider || next.provider,
        model: next.provider === "openai"
          ? runtime.openai_model || next.model
          : runtime.default_model || next.model,
      };
      dom.settingsStatus.textContent = t("settings.saved", "Saved");
      applyPermissionMode();
      applyUiLanguage();
      dom.voiceButton.hidden = !state.settings.voice_input;
      window.setTimeout(() => {
        if (dom.settingsModal.open) dom.settingsModal.close();
        dom.settingsStatus.textContent = "";
      }, 450);
      showToast(t("toast.saved", "Settings saved"), t("toast.saved.detail", "SAM will use your updated preferences."), "success");
      await loadHealth({ quiet: true }).catch(() => null);
    } catch (error) {
      dom.settingsStatus.textContent = t("settings.saveFailed", "Could not save");
      showToast("Settings were not saved", error.message, "error");
    }
  }

  function applyPermissionMode() {
    const labels = {
      guarded: t("mode.guarded", "Guarded mode"),
      strict: t("mode.strict", "Strict mode"),
      trusted: t("mode.trusted", "Trusted workspace"),
    };
    const label = labels[state.settings.permission_mode] || labels.guarded;
    $$('[data-sam-role="permission-mode-label"]').forEach((node) => { node.textContent = label; });
    if (dom.permissionModeLabel) dom.permissionModeLabel.textContent = label;
  }

  async function cyclePermissionMode() {
    const order = ["guarded", "strict", "trusted"];
    const next = order[(order.indexOf(state.settings.permission_mode) + 1) % order.length];
    state.settings.permission_mode = next;
    applyPermissionMode();
    try {
      await getJson("/api/settings", {
        method: "PUT",
        body: JSON.stringify({ permission_mode: next }),
      });
      localStorage.setItem("sam-ui-settings", JSON.stringify(state.settings));
      showToast("Permission mode updated", dom.permissionModeLabel.textContent, "success");
    } catch (error) {
      showToast("Could not update permission mode", error.message, "error");
    }
  }

  async function loadWorkspace() {
    try {
      const payload = await getJson("/api/workspace/tree");
      state.workspace = payload?.workspace || payload || { root: "", entries: [] };
      const root = state.workspace.root || state.workspace.path || state.workspace.workspace_root || "";
      state.workspace.root = root;
      dom.workspaceName.textContent = root ? compactPath(root) : "No workspace selected";
      $("#workspace-path-input").value = root;
      return state.workspace;
    } catch {
      dom.workspaceName.textContent = "Workspace unavailable";
      return state.workspace;
    }
  }

  function compactPath(path) {
    const normalized = safeText(path).replaceAll("/", "\\");
    const pieces = normalized.split("\\").filter(Boolean);
    return pieces.length > 2 ? `…\\${pieces.slice(-2).join("\\")}` : normalized;
  }

  async function saveWorkspace(event) {
    event.preventDefault();
    dom.workspaceModal.close();
  }

  function isSoraniVoice() {
    const code = String(state.settings.voice_language || state.settings.language || "").trim().toLowerCase();
    return ["ckb", "ckb-iq", "ku", "kur", "sorani"].includes(code) || code.split("-")[0] === "ckb";
  }

  function looksSoraniText(text) {
    const value = String(text || "");
    let arabic = 0;
    let letters = 0;
    for (const character of value) {
      if (character >= "\u0600" && character <= "\u06FF") arabic += 1;
      if (/\p{L}/u.test(character)) letters += 1;
    }
    return letters > 0 && arabic / letters > 0.5;
  }

  function supportsVoiceInput() {
    if (isSoraniVoice()) return Boolean(navigator.mediaDevices?.getUserMedia);
    return Boolean(window.SpeechRecognition || window.webkitSpeechRecognition);
  }

  async function toggleVoiceInput() {
    if (state.listening) {
      stopListening(true);
      return;
    }
    if (state.transcribing) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      showToast(
        t("voice.micUnavailable", "Microphone unavailable"),
        t("voice.micUnavailable.detail", "This browser cannot request microphone access on this page."),
        "warning",
      );
      return;
    }
    const sorani = isSoraniVoice();
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!sorani && !Recognition) {
      showToast(
        t("voice.inputUnavailable", "Voice input unavailable"),
        t("voice.inputUnavailable.detail", "Use a browser with Web Speech API support, such as Microsoft Edge or Chrome."),
        "warning",
      );
      return;
    }
    try {
      await startMicrophone(sorani);
    } catch (error) {
      showToast(
        t("voice.micPermission", "Microphone permission needed"),
        error.message || t("voice.micPermission.detail", "Allow microphone access and try again."),
        "warning",
        6000,
      );
      cleanupVoiceAudio();
      return;
    }
    if (sorani) {
      beginSoraniListen();
      return;
    }
    startBrowserRecognition(Recognition);
  }

  async function startMicrophone(capturePcm) {
    state.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      video: false,
    });
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return;
    state.audioContext = new AudioContext();
    if (state.audioContext.state === "suspended") await state.audioContext.resume();
    state.analyser = state.audioContext.createAnalyser();
    state.analyser.fftSize = 1024;
    state.voiceSource = state.audioContext.createMediaStreamSource(state.mediaStream);
    state.voiceSource.connect(state.analyser);
    state.capturePcm = Boolean(capturePcm);
    state.voicePcm = [];
    state.voiceSampleRate = state.audioContext.sampleRate || 48000;
    if (!capturePcm) return;
    if (typeof state.audioContext.createScriptProcessor !== "function") {
      throw new Error(t("voice.recordUnsupported", "This browser cannot record microphone audio for Sorani recognition."));
    }
    const processor = state.audioContext.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (event) => {
      if (!state.listening || !state.capturePcm) return;
      state.voicePcm.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    };
    const mute = state.audioContext.createGain();
    mute.gain.value = 0;
    state.voiceSource.connect(processor);
    processor.connect(mute);
    mute.connect(state.audioContext.destination);
    state.voiceProcessor = processor;
    state.voiceMute = mute;
  }

  function beginSoraniListen() {
    const mode = state.settings.voice_mode || "PUSH_TO_TALK";
    state.listening = true;
    state.voiceHeard = false;
    state.voiceManualStop = false;
    dom.voiceButton.classList.add("listening");
    dom.voiceButton.setAttribute("aria-label", t("composer.stop", "Stop"));
    dom.voiceWave.hidden = false;
    const wake = state.settings.voice_wake_word || "سام";
    dom.voiceStatus.textContent = mode === "WAKE_WORD"
      ? t("composer.wake", `Say ${wake}…`).replace("{wake}", wake)
      : t("composer.listen", "گوێدەگرێت…");
    if (state.voiceDeadline) window.clearTimeout(state.voiceDeadline);
    state.voiceDeadline = window.setTimeout(() => {
      if (state.listening) stopListening(false);
    }, 30_000);
    startVoiceActivityMonitor();
  }

  function startBrowserRecognition(Recognition) {
    const recognition = new Recognition();
    const mode = state.settings.voice_mode || "PUSH_TO_TALK";
    recognition.continuous = mode !== "PUSH_TO_TALK";
    recognition.interimResults = true;
    recognition.lang = state.settings.voice_language ||
      (state.settings.language && !["auto", "ku"].includes(state.settings.language) ? state.settings.language : navigator.language || "en-US");
    state.voiceTranscript = "";
    state.voiceHeard = false;
    state.voiceManualStop = false;
    recognition.onstart = () => {
      state.listening = true;
      dom.voiceButton.classList.add("listening");
      dom.voiceButton.setAttribute("aria-label", t("composer.stop", "Stop voice input"));
      dom.voiceWave.hidden = false;
      dom.voiceStatus.textContent = mode === "WAKE_WORD"
        ? `Say ${state.settings.voice_wake_word || "SAM"}…`
        : t("composer.listen", "Listening…");
      startVoiceActivityMonitor();
    };
    recognition.onresult = (event) => {
      let interim = "";
      let receivedFinal = false;
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const transcript = event.results[index][0].transcript;
        if (event.results[index].isFinal) {
          state.voiceTranscript += `${transcript} `;
          receivedFinal = true;
        }
        else interim += transcript;
      }
      dom.voiceStatus.textContent = interim ? truncate(interim, 48) : t("composer.listen", "Listening…");
      if (receivedFinal && mode !== "PUSH_TO_TALK") recognition.stop();
    };
    recognition.onerror = (event) => {
      if (!["aborted", "no-speech"].includes(event.error)) {
        showToast(t("voice.inputStopped", "Voice input stopped"), titleCase(event.error || "Speech recognition error"), "warning");
      }
    };
    recognition.onend = () => {
      const transcript = state.voiceTranscript.trim();
      const wasManual = state.voiceManualStop;
      state.listening = false;
      resetVoiceChrome();
      state.recognition = null;
      cleanupVoiceAudio();
      submitVoiceTranscript(transcript, wasManual, mode);
    };
    state.recognition = recognition;
    try {
      recognition.start();
    } catch (error) {
      showToast(t("voice.startFailed", "Microphone could not start"), error.message, "error");
      cleanupVoiceAudio();
    }
  }

  function stopListening(manual = false) {
    state.voiceManualStop = manual;
    if (state.recognition) {
      state.recognition.stop();
      return;
    }
    if (state.listening && isSoraniVoice()) finishSoraniListen();
  }

  async function finishSoraniListen() {
    if (state.transcribing) return;
    const chunks = state.voicePcm;
    const sampleRate = state.voiceSampleRate || 48000;
    const wasManual = state.voiceManualStop;
    const mode = state.settings.voice_mode || "PUSH_TO_TALK";
    state.listening = false;
    state.capturePcm = false;
    state.transcribing = true;
    resetVoiceChrome(false);
    cleanupVoiceAudio();
    const merged = concatFloat32(chunks);
    const maxSamples = Math.floor(sampleRate * 30);
    const clipped = merged.length > maxSamples ? merged.slice(merged.length - maxSamples) : merged;
    const seconds = clipped.length / sampleRate;
    if (seconds < 0.35) {
      state.transcribing = false;
      hideVoiceWave();
      showToast(t("voice.tooShort", "قسەکە زۆر کورت بوو"), t("voice.tooShort.detail", "تکایە دووبارە بڵێوە."), "warning");
      afterVoiceTurn("", wasManual, mode);
      return;
    }
    dom.voiceWave.hidden = false;
    dom.voiceStatus.textContent = t("composer.transcribing", "ناسینەوە…");
    try {
      const wav = encodeWav(downsampleBuffer(clipped, sampleRate, 16000), 16000);
      const form = new FormData();
      form.append("file", wav, "utterance.wav");
      form.append("language", state.settings.voice_language || "ckb-IQ");
      const response = await apiFetch("/api/voice/transcribe", { method: "POST", body: form }, 120_000);
      const result = await response.json();
      const transcript = String(result.text || "").trim();
      if (result.error) {
        showToast(t("voice.recognizeFailed", "ناسینەوە سەری نەگرت"), result.error, "warning", 7000);
      } else if (!transcript) {
        showToast(
          t("voice.noSpeech", "هیچ قسەیەک نەبیسترا"),
          result.reason || result.error || t("voice.tooShort.detail", "تکایە دووبارە بڵێوە."),
          "warning",
        );
      }
      submitVoiceTranscript(transcript, wasManual, mode);
    } catch (error) {
      showToast(t("voice.recognizeFailed", "ناسینەوە سەری نەگرت"), error.message || "", "error", 7000);
      afterVoiceTurn("", wasManual, mode);
    } finally {
      state.transcribing = false;
      hideVoiceWave();
    }
  }

  function submitVoiceTranscript(transcript, wasManual, mode) {
    const message = stripWakeWord(transcript, mode);
    if (message) {
      state.voiceTurn = true;
      dom.input.value = message;
      resizeComposer();
      updateSendButton();
      dom.composer.requestSubmit();
    }
    afterVoiceTurn(message, wasManual, mode);
  }

  function afterVoiceTurn(message, wasManual, mode) {
    const shouldRestart = !wasManual && ["CONVERSATION", "ALWAYS_LISTENING", "WAKE_WORD"].includes(mode);
    if (shouldRestart) scheduleVoiceRestart();
    else if (!message) dom.input.focus();
  }

  function stripWakeWord(message, mode) {
    const text = String(message || "").trim();
    if (!text) return "";
    if (mode !== "WAKE_WORD") return text;
    const configured = String(state.settings.voice_wake_word || "سام").trim();
    const needles = [configured, "سام", "SAM", "sam"].filter(Boolean);
    const lowered = text.toLocaleLowerCase();
    for (const word of needles) {
      const index = lowered.indexOf(word.toLocaleLowerCase());
      if (index >= 0) return text.slice(index + word.length).trim();
    }
    return "";
  }

  function resetVoiceChrome(hideWave = true) {
    dom.voiceButton.classList.remove("listening");
    dom.voiceButton.setAttribute("aria-label", t("composer.mic", "Start voice input"));
    if (hideWave) hideVoiceWave();
  }

  function hideVoiceWave() {
    if (!state.listening && !state.transcribing) dom.voiceWave.hidden = true;
  }

  function cleanupVoiceAudio() {
    if (state.vadFrame) cancelAnimationFrame(state.vadFrame);
    state.vadFrame = 0;
    if (state.voiceDeadline) {
      window.clearTimeout(state.voiceDeadline);
      state.voiceDeadline = 0;
    }
    try { state.voiceProcessor?.disconnect(); } catch { /* already gone */ }
    try { state.voiceMute?.disconnect(); } catch { /* already gone */ }
    try { state.voiceSource?.disconnect(); } catch { /* already gone */ }
    state.voiceProcessor = null;
    state.voiceMute = null;
    state.voiceSource = null;
    state.capturePcm = false;
    for (const track of state.mediaStream?.getTracks?.() || []) track.stop();
    state.mediaStream = null;
    state.analyser = null;
    if (state.audioContext && state.audioContext.state !== "closed") state.audioContext.close().catch(() => null);
    state.audioContext = null;
  }

  function concatFloat32(chunks) {
    const list = Array.isArray(chunks) ? chunks : [];
    const length = list.reduce((sum, chunk) => sum + chunk.length, 0);
    const output = new Float32Array(length);
    let offset = 0;
    for (const chunk of list) {
      output.set(chunk, offset);
      offset += chunk.length;
    }
    return output;
  }

  function downsampleBuffer(buffer, fromRate, toRate) {
    if (fromRate === toRate) return buffer;
    const ratio = fromRate / toRate;
    const outLength = Math.max(1, Math.round(buffer.length / ratio));
    const output = new Float32Array(outLength);
    for (let index = 0; index < outLength; index += 1) {
      const start = Math.floor(index * ratio);
      const end = Math.min(buffer.length, Math.floor((index + 1) * ratio));
      let sum = 0;
      const count = Math.max(1, end - start);
      for (let sample = start; sample < end; sample += 1) sum += buffer[sample];
      output[index] = sum / count;
    }
    return output;
  }

  function encodeWav(float32, sampleRate) {
    const samples = float32.length;
    const bytes = new ArrayBuffer(44 + samples * 2);
    const view = new DataView(bytes);
    const writeString = (offset, value) => {
      for (let index = 0; index < value.length; index += 1) view.setUint8(offset + index, value.charCodeAt(index));
    };
    writeString(0, "RIFF");
    view.setUint32(4, 36 + samples * 2, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, "data");
    view.setUint32(40, samples * 2, true);
    let offset = 44;
    for (let index = 0; index < samples; index += 1) {
      const clipped = Math.max(-1, Math.min(1, float32[index]));
      view.setInt16(offset, clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff, true);
      offset += 2;
    }
    return new Blob([bytes], { type: "audio/wav" });
  }

  function startVoiceActivityMonitor() {
    if (!state.analyser) return;
    const samples = new Float32Array(state.analyser.fftSize);
    let lastVoiceAt = performance.now();
    let bargeInTriggered = false;
    const frame = () => {
      if (!state.listening || !state.analyser) return;
      state.analyser.getFloatTimeDomainData(samples);
      const rms = Math.sqrt(samples.reduce((sum, value) => sum + value * value, 0) / samples.length);
      const speaking = rms >= Number(state.settings.voice_vad_threshold || 0.035);
      if (speaking) {
        state.voiceHeard = true;
        lastVoiceAt = performance.now();
        if (!bargeInTriggered && isSpeakingReply()) {
          bargeInTriggered = true;
          stopSpeakingReply();
          state.abortController?.abort("voice-barge-in");
          fetch("/api/abort", { method: "POST", cache: "no-store" }).catch(() => null);
          dom.voiceStatus.textContent = t("composer.bargeIn", "Barge-in · listening…");
        }
      }
      const mode = state.settings.voice_mode || "PUSH_TO_TALK";
      const silenceLimit = Number(state.settings.voice_silence_ms || 800);
      if (state.voiceHeard && ["PUSH_TO_TALK", "CONVERSATION"].includes(mode) && performance.now() - lastVoiceAt > silenceLimit) {
        stopListening(false);
        return;
      }
      state.vadFrame = requestAnimationFrame(frame);
    };
    state.vadFrame = requestAnimationFrame(frame);
  }

  function isSpeakingReply() {
    if (state.speakingReply) return true;
    if (state.ttsAudio && !state.ttsAudio.paused && !state.ttsAudio.ended) return true;
    return Boolean(window.speechSynthesis?.speaking);
  }

  function scheduleVoiceRestart() {
    const retry = () => {
      if (state.voiceManualStop || state.listening || state.transcribing) return;
      if (state.sending || isSpeakingReply()) {
        window.setTimeout(retry, 400);
        return;
      }
      toggleVoiceInput();
    };
    window.setTimeout(retry, 650);
  }

  function populateVoices() {
    if (!("speechSynthesis" in window)) return;
    const select = $("#settings-voice");
    const voices = speechSynthesis.getVoices();
    const current = state.settings.voice || select.value;
    select.replaceChildren();
    const defaultOption = document.createElement("option");
    defaultOption.value = "";
    defaultOption.textContent = "System default";
    select.appendChild(defaultOption);
    for (const voice of voices) {
      const option = document.createElement("option");
      option.value = voice.name;
      option.textContent = `${voice.name} · ${voice.lang}`;
      select.appendChild(option);
    }
    select.value = current;
  }

  function plainSpeechText(text) {
    return safeText(text)
      .replace(/```[\s\S]*?```/g, " Code block omitted. ")
      .replace(/[*_`#>\[\]()]/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function stopSpeakingReply() {
    state.ttsGeneration += 1;
    state.ttsQueue = [];
    state.ttsPrefetch = null;
    state.ttsBusy = false;
    state.ttsCursor = 0;
    state.speakingReply = false;
    try { window.speechSynthesis?.cancel(); } catch { /* ignore */ }
    if (state.ttsAudio) {
      try {
        state.ttsAudio.pause();
        state.ttsAudio.removeAttribute("src");
        state.ttsAudio.load();
      } catch { /* ignore */ }
      state.ttsAudio = null;
    }
    fetch("/api/voice/interrupt", { method: "POST", cache: "no-store" }).catch(() => null);
  }

  function takeSpeakableChunks(fullText, flush) {
    const plain = plainSpeechText(fullText);
    if (plain.length <= state.ttsCursor) return [];
    const pending = plain.slice(state.ttsCursor);
    const chunks = [];
    const sentence = /[^.!?؟…]+[.!?؟…]+(?:["»”’']*)(?:\s+|$)/gu;
    let consumed = 0;
    let match = sentence.exec(pending);
    while (match) {
      const piece = match[0].trim();
      if (piece) chunks.push(piece);
      consumed = match.index + match[0].length;
      match = sentence.exec(pending);
    }
    const rest = pending.slice(consumed);
    if (flush && rest.trim()) {
      chunks.push(rest.trim());
      consumed = pending.length;
    } else if (rest.length >= 180) {
      const cut = rest.lastIndexOf(" ", 180);
      const at = cut >= 40 ? cut : 180;
      const piece = rest.slice(0, at).trim();
      if (piece) chunks.push(piece);
      consumed += at;
    }
    state.ttsCursor += consumed;
    return chunks.filter(Boolean);
  }

  function enqueueSpokenReply(fullText, flush) {
    const chunks = takeSpeakableChunks(fullText, flush);
    if (!chunks.length) return;
    state.ttsQueue.push(...chunks);
    pumpTtsQueue();
  }

  async function fetchSoraniTtsBlob(text) {
    const response = await apiFetch("/api/voice/tts", {
      method: "POST",
      body: JSON.stringify({ text, language: "ckb" }),
    }, 120_000);
    return response.blob();
  }

  function playAudioBlob(blob) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      state.ttsAudio = audio;
      const finish = () => {
        URL.revokeObjectURL(url);
        if (state.ttsAudio === audio) state.ttsAudio = null;
        resolve();
      };
      audio.onended = finish;
      audio.onerror = finish;
      audio.play().then(() => null, reject);
    });
  }

  async function pumpTtsQueue() {
    if (state.ttsBusy) return;
    state.ttsBusy = true;
    const generation = state.ttsGeneration;
    state.speakingReply = true;
    try {
      while (state.ttsQueue.length && generation === state.ttsGeneration) {
        const text = state.ttsQueue.shift();
        let blob = null;
        if (state.ttsPrefetch && state.ttsPrefetch.text === text) {
          blob = await state.ttsPrefetch.promise;
          state.ttsPrefetch = null;
        } else {
          blob = await fetchSoraniTtsBlob(text);
        }
        if (generation !== state.ttsGeneration) return;
        const upcoming = state.ttsQueue[0];
        if (upcoming) {
          state.ttsPrefetch = { text: upcoming, promise: fetchSoraniTtsBlob(upcoming) };
        }
        try {
          await playAudioBlob(blob);
        } catch {
          if (generation !== state.ttsGeneration) return;
          await getJson("/api/voice/speak", {
            method: "POST",
            body: JSON.stringify({ text, language: "ckb" }),
          }, 120_000);
        }
      }
    } catch (error) {
      if (generation === state.ttsGeneration) {
        showToast(t("voice.speakFailed", "وەڵامەکە نەخوێندرایەوە"), error.message || "", "warning", 6000);
      }
    } finally {
      if (generation === state.ttsGeneration) {
        state.ttsBusy = false;
        if (!state.ttsQueue.length) state.speakingReply = false;
        else pumpTtsQueue();
      }
    }
  }

  async function speakText(text) {
    const plain = plainSpeechText(text);
    if (!plain) return;
    stopSpeakingReply();
    if (isSoraniVoice() || looksSoraniText(plain)) {
      enqueueSpokenReply(plain, true);
      return;
    }
    if (!("speechSynthesis" in window)) {
      showToast(
        t("voice.outputUnavailable", "Voice output unavailable"),
        t("voice.outputUnavailable.detail", "This browser does not support speech synthesis."),
        "warning",
      );
      return;
    }
    const voices = speechSynthesis.getVoices();
    const voice = voices.find((item) => item.name === state.settings.voice);
    const chunks = (plain.match(/[^.!?؟\n]+[.!?؟]?/g) || [plain])
      .flatMap((sentence) => sentence.length <= 240 ? [sentence] : sentence.match(/.{1,240}(?:\s|$)/g) || [sentence])
      .map((chunk) => chunk.trim())
      .filter(Boolean);
    state.speakingReply = true;
    for (const chunk of chunks) {
      const utterance = new SpeechSynthesisUtterance(chunk);
      if (voice) utterance.voice = voice;
      utterance.lang = state.settings.voice_language || voice?.lang || navigator.language;
      utterance.rate = Number(state.settings.speech_rate || 1);
      utterance.onend = () => {
        if (!window.speechSynthesis?.speaking) state.speakingReply = false;
      };
      utterance.onerror = () => { state.speakingReply = false; };
      speechSynthesis.speak(utterance);
    }
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      showToast("Copied", "Text is ready on your clipboard.", "success", 2200);
    } catch {
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
      showToast("Copied", "Text is ready on your clipboard.", "success", 2200);
    }
  }

  function selectView(name) {
    // The markup declares `.content-panel.active { display:flex }` and
    // `.nav-item.active`, but nothing was toggling either, so every view other
    // than the default chat was unreachable. This binds the two together.
    let matched = false;
    for (const panel of $$(".content-panel")) {
      const active = panel.id === `panel-${name}`;
      panel.classList.toggle("active", active);
      matched = matched || active;
    }
    if (!matched) return false;
    for (const button of $$("[data-nav]")) {
      button.classList.toggle("active", button.dataset.nav === name);
      button.setAttribute("aria-current", button.dataset.nav === name ? "page" : "false");
    }
    state.activePanel = name;
    try {
      window.localStorage.setItem("sam.activePanel", name);
    } catch {
      // Private windows and blocked site data are fine; the view still switches.
    }
    // Give a newly revealed panel a chance to refresh what it shows.
    window.dispatchEvent(new CustomEvent("sam:view-changed", { detail: { view: name } }));
    return true;
  }

  function selectContextPanel(name) {
    for (const tab of $$(".context-tabs [role='tab']")) {
      const active = tab.dataset.panel === name;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
    }
    for (const panel of $$(".context-panel")) {
      const active = panel.id === `panel-${name}`;
      panel.hidden = !active;
      panel.classList.toggle("active", active);
    }
  }

  function selectSettingsPanel(name) {
    for (const tab of $$("[data-settings-panel]")) {
      const active = tab.dataset.settingsPanel === name;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
    }
    for (const panel of $$("[data-settings-section]")) {
      const active = panel.dataset.settingsSection === name;
      panel.hidden = !active;
      panel.classList.toggle("active", active);
    }
  }

  function openContext(name = "plan") {
    selectContextPanel(name);
    if (window.matchMedia("(max-width: 1180px)").matches) {
      dom.app.classList.add("context-open");
      dom.scrim.hidden = false;
    } else {
      dom.app.classList.remove("context-collapsed");
      localStorage.setItem("sam-context-collapsed", "false");
    }
    $("[data-action='toggle-context']")?.setAttribute("aria-expanded", "true");
  }

  function toggleContext() {
    if (window.matchMedia("(max-width: 1180px)").matches) {
      const open = dom.app.classList.toggle("context-open");
      dom.scrim.hidden = !open && !dom.app.classList.contains("sidebar-open");
      $("[data-action='toggle-context']")?.setAttribute("aria-expanded", String(open));
    } else {
      const collapsed = dom.app.classList.toggle("context-collapsed");
      localStorage.setItem("sam-context-collapsed", String(collapsed));
      $("[data-action='toggle-context']")?.setAttribute("aria-expanded", String(!collapsed));
    }
  }

  function openSidebar() {
    dom.app.classList.add("sidebar-open");
    dom.scrim.hidden = false;
  }

  function closeMobilePanels() {
    dom.app.classList.remove("sidebar-open", "context-open");
    dom.scrim.hidden = true;
  }

  function openSettings(section = "general") {
    applySettingsToForm();
    selectSettingsPanel(section);
    if (!dom.settingsModal.open) dom.settingsModal.showModal();
  }

  function openWorkspace() {
    $("#workspace-path-input").value = state.workspace.root || state.settings.workspace_path || "";
    if (!dom.workspaceModal.open) dom.workspaceModal.showModal();
  }

  async function refreshAll() {
    setConnectivity("connecting");
    const results = await Promise.allSettled([
      loadHealth(),
      loadConversations(),
      loadModels(),
      loadApprovals(),
      loadMemories(),
      loadAudit(),
      loadWorkspace(),
    ]);
    if (results.some((result) => result.status === "fulfilled")) showToast("SAM refreshed", "Local status and activity are up to date.", "success", 2200);
  }

  async function renameSession() {
    if (!state.conversationId) return;
    const conversation = state.conversations.find((item) => String(item.id) === String(state.conversationId));
    const current = conversation?.title || dom.title.textContent;
    const next = window.prompt("Conversation name", current)?.trim();
    if (!next || next === current) return;
    try {
      await getJson(`/api/conversations/${encodeURIComponent(state.conversationId)}`, {
        method: "PATCH",
        body: JSON.stringify({ title: next }),
      });
      if (conversation) conversation.title = next;
      dom.title.textContent = next;
      renderSessions();
    } catch {
      showToast("Rename unavailable", "This SAM backend does not currently expose conversation renaming.", "warning");
    }
  }

  function bindEvents() {
    dom.composer.addEventListener("submit", (event) => {
      event.preventDefault();
      if (state.sending) stopCurrentResponse();
      else sendMessage(dom.input.value);
    });
    dom.input.addEventListener("input", () => {
      resizeComposer();
      updateSendButton();
    });
    dom.input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        dom.composer.requestSubmit();
      }
    });
    dom.fileInput.addEventListener("change", () => {
      handleFiles(dom.fileInput.files);
      dom.fileInput.value = "";
    });
    dom.memorySearches.forEach((element) => element.addEventListener("input", renderMemories));
    dom.auditFilters.forEach((element) => element.addEventListener("change", renderAudit));
    dom.settingsForm.addEventListener("submit", saveSettings);
    dom.workspaceForm.addEventListener("submit", saveWorkspace);
    $("#settings-provider").addEventListener("change", renderModelOptions);
    $("#settings-temperature").addEventListener("input", (event) => {
      $("#temperature-output").textContent = Number(event.target.value).toFixed(1);
    });
    $("#settings-speech-rate").addEventListener("input", (event) => {
      $("#speech-rate-output").textContent = `${Number(event.target.value).toFixed(1)}×`;
    });
    dom.scrim.addEventListener("click", closeMobilePanels);

    document.addEventListener("click", (event) => {
      const navTarget = event.target.closest("[data-nav]");
      if (navTarget?.dataset.nav) {
        event.preventDefault();
        selectView(navTarget.dataset.nav);
        closeMobilePanels();
      }
    });

    document.addEventListener("click", async (event) => {
      const actionTarget = event.target.closest("[data-action]");
      const action = actionTarget?.dataset.action;
      if (action) {
        const handlers = {
          "new-chat": createConversation,
          "open-sidebar": openSidebar,
          "close-sidebar": closeMobilePanels,
          "toggle-context": toggleContext,
          "open-settings": () => openSettings("general"),
          "open-workspace": openWorkspace,
          "refresh-all": refreshAll,
          "refresh-sessions": loadConversations,
          "refresh-plan": () => state.conversationId ? openConversation(state.conversationId, { quiet: true }) : renderPlan(),
          "refresh-approvals": loadApprovals,
          "refresh-memory": () => loadMemories(dom.memorySearch.value),
          "refresh-audit": loadAudit,
          "export-audit": exportAudit,
          "attach-file": () => dom.fileInput.click(),
          "toggle-voice-input": toggleVoiceInput,
          "stop-listening": stopListening,
          "cycle-mode": cyclePermissionMode,
          "rename-session": renameSession,
          "copy-approval-command": () => copyText($("#approval-command").textContent),
          "approve-action": () => decideApproval("approve"),
          "deny-approval": () => decideApproval("deny"),
        };
        if (handlers[action]) {
          event.preventDefault();
          handlers[action]();
        }
      }

      const suggestion = event.target.closest("[data-prompt]");
      if (suggestion) {
        dom.input.value = suggestion.dataset.prompt;
        resizeComposer();
        updateSendButton();
        dom.input.focus();
      }

      const session = event.target.closest("[data-session-id]");
      if (session) openConversation(session.dataset.sessionId);

      const approval = event.target.closest("[data-approval-id]");
      if (approval) openApproval(approval.dataset.approvalId);

      const contextTab = event.target.closest("[data-panel]");
      if (contextTab) selectContextPanel(contextTab.dataset.panel);

      const settingsTab = event.target.closest("[data-settings-panel]");
      if (settingsTab) selectSettingsPanel(settingsTab.dataset.settingsPanel);

      const messageAction = event.target.closest("[data-message-action]");
      if (messageAction) {
        const messageNode = messageAction.closest(".message");
        const content = $(".message-content", messageNode)?.innerText || "";
        if (messageAction.dataset.messageAction === "copy") copyText(content);
        if (messageAction.dataset.messageAction === "speak") speakText(content);
      }
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        closeMobilePanels();
        stopListening();
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        createConversation();
      }
      if ((event.ctrlKey || event.metaKey) && event.key === ",") {
        event.preventDefault();
        openSettings();
      }
    });

    window.addEventListener("resize", () => {
      if (!window.matchMedia("(max-width: 1180px)").matches) {
        dom.app.classList.remove("context-open");
        dom.scrim.hidden = !dom.app.classList.contains("sidebar-open");
      }
    });

    if ("speechSynthesis" in window) {
      speechSynthesis.addEventListener?.("voiceschanged", populateVoices);
      window.speechSynthesis.onvoiceschanged = populateVoices;
    }
  }

  function restoreUiState() {
    const collapsed = localStorage.getItem("sam-context-collapsed") === "true";
    if (collapsed && !window.matchMedia("(max-width: 1180px)").matches) dom.app.classList.add("context-collapsed");
  }

  function startPolling() {
    window.setInterval(() => {
      if (document.visibilityState === "visible") loadHealth({ quiet: true }).catch(() => null);
    }, HEALTH_POLL_MS);
    window.setInterval(async () => {
      if (document.visibilityState !== "visible") return;
      const before = state.approvals.length;
      await loadApprovals();
      if (state.approvals.length > before) showApprovalNotification();
    }, APPROVAL_POLL_MS);
  }

  async function init() {
    bindEvents();
    restoreUiState();
    applyUiLanguage();
    renderPlan();
    updateSendButton();
    populateVoices();
    setConnectivity("connecting");

    await Promise.allSettled([
      loadHealth(),
      loadSettings(),
      loadModels(),
      loadApprovals(),
      loadMemories(),
      loadAudit(),
      loadWorkspace(),
    ]);
    await loadConversations({ selectFirst: true });
    startPolling();
  }

  init();
})();
