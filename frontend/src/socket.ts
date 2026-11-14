import { decodeFrame, type Frame } from "./protocol";

export type ServerMessage =
  | { type: "started"; n_prompt: number; expected_tokens: number }
  | { type: "done"; frames: number; stopped: boolean }
  | { type: "error"; message: string };

export interface SocketHandlers {
  onOpen(): void;
  onClose(): void;
  onFrame(frame: Frame): void;
  onMessage(msg: ServerMessage): void;
}

const RETRY_MS = 3000;

export class Socket {
  private ws: WebSocket | null = null;
  private retry: number | undefined;

  constructor(private readonly handlers: SocketHandlers) {}

  get open(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  connect(): void {
    window.clearTimeout(this.retry);
    if (this.ws && this.ws.readyState <= WebSocket.OPEN) return;
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => this.handlers.onOpen();
    ws.onmessage = (ev) => {
      if (ev.data instanceof ArrayBuffer) this.handlers.onFrame(decodeFrame(ev.data));
      else this.handlers.onMessage(JSON.parse(ev.data as string) as ServerMessage);
    };
    ws.onclose = () => {
      if (this.ws === ws) this.ws = null;
      this.handlers.onClose();
      this.retry = window.setTimeout(() => this.connect(), RETRY_MS);
    };
    this.ws = ws;
  }

  send(msg: object): boolean {
    if (!this.open) return false;
    this.ws!.send(JSON.stringify(msg));
    return true;
  }
}
