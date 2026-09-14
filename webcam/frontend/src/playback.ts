export class PlaybackClock {
  interval = 1000 / 15;
  private lastBatch: number | null = null;
  private next = 0;

  observe(now: number, frames: number, queued: number) {
    if (this.lastBatch !== null && now > this.lastBatch) {
      const sample = (now - this.lastBatch) / (frames + queued);
      this.interval = Math.min(1000, Math.max(1000 / 30, .8 * this.interval + .2 * sample));
    }
    this.lastBatch = now;
  }

  ready(now: number) {
    if (now < this.next) return false;
    this.next = Math.max(this.next + this.interval, now + this.interval * .1);
    return true;
  }
}
