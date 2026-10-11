// Voice commands for the robot: "hey rover, <command>", heard by the browser's speech recognition (Chrome, Edge or
// Safari; Chrome sends the audio to Google, so it needs internet). The microphone works only on https:// or
// localhost: open the dashboard through `ssh -L 8080:localhost:8080 <jetson>` as http://localhost:8080.

// "hey rover" as recognizers tend to hear it
const WAKE = /\b(?:hey|hay|hi|okay|ok|a)[\s,]+(?:rover|rovers|robert|rober|roper|robot)\b[\s,.:;!?-]*/i;
const ARMED_MS = 7000;          // "hey rover" alone: the next thing said within this is the command

/** What follows the wake word ("" for the wake word alone), or null without it. */
export function afterWake(text) {
  const m = WAKE.exec(text);
  return m ? text.slice(m.index + m[0].length).replace(/^(?:please|can you|could you)\s+/i, '').trim() : null;
}

export class Voice {
  /** onCommand(text): a command to send; onHeard(text, final): what it hears; onState(state, detail): its state. */
  constructor({ onCommand, onHeard = () => {}, onState = () => {} }) {
    this.onCommand = onCommand; this.onHeard = onHeard; this.onState = onState;
    this.enabled = false; this.armedUntil = 0; this.recognition = null;
  }

  static supported() { return typeof window !== 'undefined' && !!(window.SpeechRecognition || window.webkitSpeechRecognition); }

  start() {
    if (!Voice.supported()) { this.onState('unsupported'); return; }
    if (!window.isSecureContext) { this.onState('insecure'); return; }
    this.enabled = true;
    this.listen();
  }

  stop() {
    this.enabled = false;
    this.armedUntil = 0;
    try { this.recognition?.abort(); } catch { /* already stopped */ }
    this.recognition = null;
    this.onState('off');
  }

  listen() {
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const r = this.recognition = new Recognition();
    r.lang = 'en-US'; r.continuous = true; r.interimResults = true; r.maxAlternatives = 1;
    r.onstart = () => this.onState(this.armed() ? 'armed' : 'listening');
    r.onresult = event => {
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const text = event.results[i][0].transcript.trim();
        if (!text) continue;
        this.onHeard(text, event.results[i].isFinal);
        if (event.results[i].isFinal) this.handle(text);
      }
    };
    r.onerror = event => {
      if (event.error === 'not-allowed' || event.error === 'service-not-allowed') { this.enabled = false; this.onState('blocked'); }
      else if (event.error === 'network') this.onState('network');
      // 'no-speech' and 'aborted' just end this session; onend starts the next
    };
    r.onend = () => {                      // Chrome ends a session after a silence or a minute: keep listening
      if (this.enabled && this.recognition === r) setTimeout(() => { if (this.enabled && this.recognition === r) this.listen(); }, 250);
    };
    try { r.start(); } catch { /* starting twice: the running one carries on */ }
  }

  armed() { return Date.now() < this.armedUntil; }

  handle(text) {
    const rest = afterWake(text), wasArmed = this.armed();
    if (rest === '') {                     // "hey rover" alone: listen for the command
      this.armedUntil = Date.now() + ARMED_MS;
      this.onState('armed');
      return;
    }
    if (rest === null && !wasArmed) return;          // talk without the wake word: not for the robot
    this.armedUntil = 0;
    if (wasArmed) this.onState('listening');         // (before the command, so its "sent" note stays)
    this.onCommand(rest ?? text);
  }
}
